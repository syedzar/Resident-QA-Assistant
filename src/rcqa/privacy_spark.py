"""Distributed version of rcqa.privacy for scanning whole gold tables in Spark.

Same rules, same regex patterns (imported from rcqa.privacy, so they can't drift):
  * format patterns          -> regexp_extract_all on each row's JSON
  * known name/ID tokens      -> tokenize, then join against the known-token list
  * the resident's own birth date and street address -> join on resident_id
Each gold row is serialised with to_json(struct(*)) so every column is scanned.
"""
from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from rcqa.privacy import PATTERNS, KnownIdentifiers


def row_text(df: DataFrame) -> DataFrame:
    """(resident_id, text) where text is the whole row as JSON."""
    rid = F.col("resident_id") if "resident_id" in df.columns else F.lit(None).cast("int")
    return df.select(rid.alias("resident_id"), F.to_json(F.struct(*df.columns)).alias("text"))


def scan(spark: SparkSession, texts: DataFrame, known: KnownIdentifiers,
         own_addresses: dict[int, str] | None = None) -> DataFrame:
    """Return one row per leak: (resident_id, kind, value). Empty DataFrame = clean."""
    leaks = []
    for kind, rx in PATTERNS.items():
        leaks.append(texts.select("resident_id", F.lit(kind).alias("kind"),
                                  F.explode(F.regexp_extract_all("text", F.lit(rx.pattern), F.lit(0))).alias("value")))

    # small lookup tables are built from pandas (Arrow path), which also works on local Windows Spark
    tokens = spark.createDataFrame(pd.DataFrame({"value": sorted(known.tokens) or [""]}), "value string")
    leaks.append(texts.select("resident_id", F.explode(F.split("text", r"[^A-Za-z0-9'-]+")).alias("value"))
                 .join(F.broadcast(tokens), "value").select("resident_id", F.lit("known_identifier").alias("kind"),
                                                              "value"))

    own = [(rid, bd.isoformat(), (own_addresses or {}).get(rid)) for rid, bd in known.birthdates.items()]
    if own:
        own_df = spark.createDataFrame(pd.DataFrame(own, columns=["resident_id", "birthdate", "address"]),
                                       "resident_id int, birthdate string, address string")
        joined = texts.join(F.broadcast(own_df), "resident_id")
        leaks.append(joined.where(F.instr("text", F.col("birthdate")) > 0)
                     .select("resident_id", F.lit("birthdate").alias("kind"), F.col("birthdate").alias("value")))
        leaks.append(joined.where(F.col("address").isNotNull() & (F.instr("text", F.col("address")) > 0))
                     .select("resident_id", F.lit("address").alias("kind"), F.col("address").alias("value")))

    out = leaks[0]
    for part in leaks[1:]:
        out = out.unionByName(part)
    return out
