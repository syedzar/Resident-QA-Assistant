# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold: de-identified resident profiles
# MAGIC * **Who:** living residents only (1,000), each given a short `resident_id` (101–1100).
# MAGIC   The UUID ↔ resident_id map stays in the restricted `silver_resident_key_map`.
# MAGIC * **What:** allow-listed columns only (`rcqa.gold`). Names, SSN, licence, passport, address,
# MAGIC   ZIP, lat/lon and all record UUIDs are never copied. Birth date → age (capped at 90) + 5-year band.
# MAGIC * **Profile:** one row per resident joining conditions, medications, encounters, allergies,
# MAGIC   fall-risk assessments and injuries, plus a rendered `profile_text` for chunking (stage 5).
# MAGIC * **Privacy gate:** every row of every gold table is scanned for identifiers. Any leak = job fails.

# COMMAND ----------

import os, sys, uuid
sys.path.insert(0, os.path.abspath("../../src"))

import pandas as pd
from pyspark.sql import functions as F
from rcqa import config, dq, gold, privacy, privacy_spark
from rcqa.profile_text import render

dbutils.widgets.text("run_id", "")
RUN_ID = dbutils.widgets.get("run_id") or str(uuid.uuid4())

def silver(t):
    return spark.table(config.table(f"silver_{t}"))

def write(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.table(name))
    return spark.table(config.table(name))

AS_OF = silver("encounters").agg(F.max(F.to_date("start"))).first()[0].isoformat()
print("Profiles as of", AS_OF)

# COMMAND ----------

# MAGIC %md ## Key map (restricted) and gold fact tables

# COMMAND ----------

keymap = write(gold.resident_key_map(silver("patients"), config.RESIDENT_ID_SALT), "silver_resident_key_map")

tables = {
    "gold_residents":   gold.residents(silver("patients"), keymap, AS_OF),
    "gold_conditions":  gold.conditions(silver("conditions"), keymap),
    "gold_medications": gold.medications(silver("medications"), keymap),
    "gold_encounters":  gold.encounters(silver("encounters"), keymap),
    "gold_allergies":   gold.allergies(silver("allergies"), keymap),
    "gold_fall_risk":   gold.fall_risk(silver("observations"), keymap),
}
tables = {name: write(df, name) for name, df in tables.items()}

for name, df in tables.items():
    print(f"{name:<18} {df.count():>9,} rows")

# COMMAND ----------

# MAGIC %md ## Resident profiles
# MAGIC Structured arrays are built in Spark; the text rendering is plain Python on the driver
# MAGIC (1,000 rows). `rcqa.profile_text.render` is unit-tested locally.

# COMMAND ----------

profiles = gold.profiles(tables["gold_residents"], tables["gold_conditions"], tables["gold_medications"],
                         tables["gold_encounters"], tables["gold_fall_risk"], tables["gold_allergies"])
profiles = write(profiles, "gold_resident_profiles_tmp")

texts = pd.DataFrame([(r.resident_id, render(r.asDict(recursive=True), AS_OF)) for r in profiles.collect()],
                     columns=["resident_id", "profile_text"])
profiles = write(profiles.join(spark.createDataFrame(texts), "resident_id")
                 .withColumn("profile_as_of", F.to_date(F.lit(AS_OF))), "gold_resident_profiles")
spark.sql(f"DROP TABLE IF EXISTS {config.table('gold_resident_profiles_tmp')}")
tables["gold_resident_profiles"] = profiles

example = profiles.where("resident_id = 204").select("profile_text").first()
print(example.profile_text if example else "(no resident 204)")

# COMMAND ----------

# MAGIC %md ## Privacy gate
# MAGIC Known identifiers come from the restricted silver layer. Every gold row is serialised to JSON
# MAGIC and scanned in Spark; the rendered profile texts (what gets embedded later) are also scanned
# MAGIC with the plain-Python scanner, which additionally checks every resident's street address.

# COMMAND ----------

patient_rows = [r.asDict() for r in silver("patients").collect()]
key = {r.patient: r.resident_id for r in keymap.collect()}
known = privacy.KnownIdentifiers.from_patient_rows(patient_rows, key)
own_addresses = {key[r["id"]]: r["address"] for r in patient_rows if r["id"] in key and r["address"]}
print(f"known identifiers: {len(known.tokens):,} tokens, {len(known.phrases):,} addresses, "
      f"{len(known.birthdates):,} birth dates")

results, leak_samples = [], []
for name, df in tables.items():
    leaks = privacy_spark.scan(spark, privacy_spark.row_text(df), known, own_addresses)  # no .cache() on serverless
    n = leaks.count()
    if n:
        leak_samples += [(name, *r) for r in leaks.limit(5).collect()]
    results.append(dq.CheckResult("gold", name, "privacy_leaks", "*", n, 0, n == 0, "error"))

text_leaks = [(int(r.resident_id), l.kind, l.value) for r in profiles.select("resident_id", "profile_text").collect()
              for l in privacy.find_leaks(r.profile_text, known, resident_id=int(r.resident_id))]
results.append(dq.CheckResult("gold", "gold_resident_profiles", "privacy_leaks_text", "profile_text",
                              len(text_leaks), 0, not text_leaks, "error"))
leak_samples += [("profile_text", *t) for t in text_leaks[:5]]

# Canary: prove the gate can fail. 20 rows of real identifiers from silver go through the SAME
# row_text() -> scan() path as the gold tables, and EVERY detector kind must fire at least once.
# If a refactor breaks any detector (or row_text), "0 leaks" above would be meaningless.
canary_rows = (silver("patients").join(keymap.withColumnRenamed("patient", "id"), "id").limit(20)
               .select("resident_id", "first", "last", "ssn", "address", "birthdate"))
canary_kinds = {r.kind for r in privacy_spark.scan(spark, privacy_spark.row_text(canary_rows), known, own_addresses)
                .select("kind").distinct().collect()}
missing = privacy.canary_missing(canary_kinds)
results.append(dq.CheckResult("gold", "privacy_gate", "canary_detectors_missing", ",".join(sorted(missing)) or "-",
                              len(missing), 0, not missing, "error"))

for r in results:
    print(f"{r.table:<24} {r.check:<22} value={r.value}")
if leak_samples:
    display(spark.createDataFrame(pd.DataFrame(leak_samples, columns=["table", "resident_id", "kind", "value"]).astype(str)))

# COMMAND ----------

# MAGIC %md ## Gold DQ + results

# COMMAND ----------

gold_rules = {
    "gold_residents": dq.TableRules("gold_residents", key=["resident_id"], not_null=["sex", "age", "age_band"],
                                    min_rows=1000),
    "gold_resident_profiles": dq.TableRules("gold_resident_profiles", key=["resident_id"],
                                            not_null=["profile_text"], min_rows=1000,
                                            foreign_keys={"resident_id": ("gold_residents", "resident_id")}),
}
for name in ["gold_conditions", "gold_medications", "gold_encounters", "gold_allergies", "gold_fall_risk"]:
    gold_rules[name] = dq.TableRules(name, not_null=["resident_id"],
                                     foreign_keys={"resident_id": ("gold_residents", "resident_id")})
for name, rules in gold_rules.items():
    results += dq.evaluate("gold", rules, dq.profile(tables[name], rules, parents=tables))

dq_df = (spark.createDataFrame(pd.DataFrame(dq.to_rows(results)))
         .withColumn("run_id", F.lit(RUN_ID)).withColumn("checked_at", F.current_timestamp()))
dq_df.write.mode("append").option("mergeSchema", "true").saveAsTable(config.table("dq_results"))
display(dq_df.orderBy("passed", "table", "check"))
dq.raise_on_failure(results)
