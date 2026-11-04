"""Silver -> gold: de-identified, analysis-ready resident tables.

De-identification happens HERE, on the way into gold, by construction:
  * Only living residents (deathdate IS NULL): the "current residents" of the home.
  * Synthea's patient UUID is replaced by a short ``resident_id`` (101, 102, ...).
    The UUID <-> resident_id map is kept in the restricted silver layer only.
  * Gold columns are an allow-list: nothing is copied over unless named below.
    Names, SSN, driver's licence, passport, address, city, county, ZIP, lat/lon,
    birthplace, income, and encounter/provider/organization/payer IDs never reach gold.
  * Exact birth date -> age in years, capped at 90 ("90+"), as HIPAA Safe Harbor does
    for ages over 89, plus a 5-year age band.
Clinical event dates (encounters, diagnoses, medication starts) are kept: staff need
"when". In real data they are identifiers too, and per-resident date shifting would be
the next step. See README "Privacy".
"""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

AGE_CAP = 90
FALL_LEVEL = "Fall risk level [Morse Fall Scale]"
FALL_SCORE = "Fall risk total [Morse Fall Scale]"
# Injury-type diagnoses: the closest thing to falls that Synthea records.
INJURY_PATTERN = r"(?i)(fracture|injury|sprain|laceration|concussion|contusion|dislocation|whiplash|burn)"
NOT_INJURY_PATTERN = r"(?i)(dental|filling)"


def resident_key_map(patients: DataFrame, salt: str) -> DataFrame:
    """Living patients -> stable short ids (101..). Ordered by a salted hash, so the
    numbering does not leak anything (e.g. alphabetical order or age) and is stable
    across reruns. RESTRICTED: stays in silver."""
    w = Window.orderBy(F.sha2(F.concat(F.lit(salt), F.col("id")), 256))
    return (patients.where(F.col("deathdate").isNull())
            .select(F.col("id").alias("patient"), (F.row_number().over(w) + 100).alias("resident_id")))


def _with_resident(df: DataFrame, keymap: DataFrame) -> DataFrame:
    """Swap the patient UUID for resident_id. Inner join also drops deceased patients."""
    return df.join(keymap, "patient", "inner").drop("patient")


def _status(stop_col: str = "stop"):
    return F.when(F.col(stop_col).isNull(), "active").otherwise("resolved")


def _tag():
    # "Hypertension (disorder)" -> "disorder"; descriptions without a tag -> NULL
    return F.nullif(F.regexp_extract("description", r"\(([^()]+)\)\s*$", 1), F.lit(""))


def residents(patients: DataFrame, keymap: DataFrame, as_of: str) -> DataFrame:
    age = F.floor(F.months_between(F.to_date(F.lit(as_of)), F.col("birthdate")) / 12).cast("int")
    capped = F.least(age, F.lit(AGE_CAP))
    band = (F.when(age >= AGE_CAP, F.lit(f"{AGE_CAP}+"))
            .otherwise(F.concat((F.floor(age / 5) * 5).cast("string"), F.lit("-"),
                                (F.floor(age / 5) * 5 + 4).cast("string"))))
    return (_with_resident(patients.withColumnRenamed("id", "patient"), keymap)
            .select("resident_id", F.col("gender").alias("sex"), capped.alias("age"), band.alias("age_band")))


def conditions(df: DataFrame, keymap: DataFrame) -> DataFrame:
    is_injury = F.col("description").rlike(INJURY_PATTERN) & ~F.col("description").rlike(NOT_INJURY_PATTERN)
    return _with_resident(df, keymap).select(
        "resident_id", "code", "description", _tag().alias("semantic_tag"),
        F.col("start").alias("onset_date"), F.col("stop").alias("resolved_date"),
        _status().alias("status"), is_injury.alias("is_injury"))


def medications(df: DataFrame, keymap: DataFrame) -> DataFrame:
    return _with_resident(df, keymap).select(
        "resident_id", "code", "description",
        F.to_date("start").alias("start_date"), F.to_date("stop").alias("stop_date"),
        _status().alias("status"), F.col("reasondescription").alias("reason"), "dispenses")


def encounters(df: DataFrame, keymap: DataFrame) -> DataFrame:
    return _with_resident(df, keymap).select(
        "resident_id", F.to_date("start").alias("encounter_date"),
        F.col("encounterclass").alias("encounter_class"), "description",
        F.col("reasondescription").alias("reason"))


def allergies(df: DataFrame, keymap: DataFrame) -> DataFrame:
    return _with_resident(df, keymap).select(
        "resident_id", "description", "type", "category",
        F.col("description1").alias("reaction"), F.col("severity1").alias("severity"),
        F.col("start").alias("start_date"), _status().alias("status"))


def fall_risk(observations: DataFrame, keymap: DataFrame) -> DataFrame:
    """One row per Morse Fall Scale assessment: level text + total score."""
    mfs = observations.where(F.col("description").isin(FALL_LEVEL, FALL_SCORE))
    return (_with_resident(mfs, keymap)
            .groupBy("resident_id", F.to_date("date").alias("assessed_date"))
            .agg(F.max(F.when(F.col("description") == FALL_LEVEL, F.col("value"))).alias("risk_level"),
                 F.max(F.when(F.col("description") == FALL_SCORE, F.col("value_num"))).alias("score")))


def profiles(res: DataFrame, cond: DataFrame, meds: DataFrame, enc: DataFrame,
             falls: DataFrame, allg: DataFrame, recent_n: int = 5) -> DataFrame:
    """One row per resident with everything a care-staff question needs, as sorted arrays.
    sort_array on structs sorts by the first field, so the date goes first."""
    def latest(date_col, fields, n=None):
        arr = F.sort_array(F.collect_list(F.struct(date_col, *fields)), asc=False)
        return F.slice(arr, 1, n) if n else arr

    active_conditions = (cond.where("status = 'active' AND semantic_tag = 'disorder' AND NOT is_injury")
                         .groupBy("resident_id")
                         .agg(latest("onset_date", ["description"]).alias("active_conditions")))
    injuries = (cond.where("is_injury").groupBy("resident_id")
                .agg(latest("onset_date", ["description", "resolved_date"]).alias("injuries")))
    active_meds = (meds.where("status = 'active'").groupBy("resident_id")
                   .agg(latest("start_date", ["description", "reason"]).alias("active_medications")))
    recent_enc = (enc.groupBy("resident_id")
                  .agg(latest("encounter_date", ["encounter_class", "description", "reason"], recent_n)
                       .alias("recent_encounters"), F.count("*").alias("n_encounters")))
    fall_hist = (falls.groupBy("resident_id")
                 .agg(latest("assessed_date", ["risk_level", "score"], recent_n).alias("fall_risk_history")))
    active_allergies = (allg.where("status = 'active'").groupBy("resident_id")
                        .agg(latest("start_date", ["description", "reaction", "severity"]).alias("allergies")))

    out = res
    for part in [active_conditions, injuries, active_meds, recent_enc, fall_hist, active_allergies]:
        out = out.join(part, "resident_id", "left")
    return out.withColumn("latest_fall_risk", F.col("fall_risk_history")[0])
