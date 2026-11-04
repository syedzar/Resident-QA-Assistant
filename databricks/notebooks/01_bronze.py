# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze: raw files → Delta
# MAGIC Land every raw input **exactly as it arrived**, plus two lineage columns
# MAGIC (`_source_file`, `_ingested_at`). Every CSV column stays a string: if a value
# MAGIC is malformed we still keep it, and silver decides what to do with it.
# MAGIC
# MAGIC | Input (Volume) | Bronze table |
# MAGIC |---|---|
# MAGIC | `synthea/*.csv.gz` (9 tables) | `bronze_<table>` |
# MAGIC | `docs/*.pdf`, `docs/*.txt` | `bronze_documents` (binary) |
# MAGIC | `drug_labels/*.json` | `bronze_drug_labels` (raw JSON text) |
# MAGIC | `sources/sources.csv` | `bronze_sources` |

# COMMAND ----------

import os, sys, uuid
sys.path.insert(0, os.path.abspath("../../src"))   # repo's src/ (notebook runs from databricks/notebooks)

from pyspark.sql import functions as F
from rcqa import config, dq

dbutils.widgets.text("run_id", "")                 # the scheduled job passes its run id (stage 9)
RUN_ID = dbutils.widgets.get("run_id") or str(uuid.uuid4())
print("run_id:", RUN_ID)

def lineage(df):
    return (df.withColumn("_source_file", F.col("_metadata.file_path"))
              .withColumn("_ingested_at", F.current_timestamp()))

def write(df, name):
    (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.table(name)))
    return spark.table(config.table(name))

# COMMAND ----------

# MAGIC %md ## Synthea tables

# COMMAND ----------

results = []
for t in config.SYNTHEA_TABLES:
    raw = (spark.read
           .option("header", True).option("inferSchema", False)   # all strings in bronze
           .option("quote", '"').option("escape", '"')
           .option("mode", "PERMISSIVE")
           .csv(f"{config.VOLUME}/synthea/{t}.csv.gz"))
    bronze = write(lineage(raw), f"bronze_{t}")
    rules = dq.TableRules(f"bronze_{t}", min_rows=1, exact_duplicates=True)
    results += dq.evaluate("bronze", rules, dq.profile(bronze, rules))
    print(f"bronze_{t:<14} {bronze.count():>10,} rows")

# COMMAND ----------

# MAGIC %md ## Documents, drug labels, source list

# COMMAND ----------

docs = write(lineage(spark.read.format("binaryFile")
                     .option("pathGlobFilter", "*.{pdf,txt}")
                     .load(f"{config.VOLUME}/docs")), "bronze_documents")

labels = write(lineage(spark.read.option("wholetext", True)
                       .text(f"{config.VOLUME}/drug_labels/*.json")), "bronze_drug_labels")

sources = write(lineage(spark.read.option("header", True).option("multiLine", True)
                        .option("quote", '"').option("escape", '"')
                        .csv(f"{config.VOLUME}/sources/sources.csv")), "bronze_sources")

for name, df in [("bronze_documents", docs), ("bronze_drug_labels", labels), ("bronze_sources", sources)]:
    rules = dq.TableRules(name, min_rows=1)
    results += dq.evaluate("bronze", rules, dq.profile(df, rules))
    print(f"{name:<22} {df.count():>4} rows")

# COMMAND ----------

# MAGIC %md ## Data-quality results
# MAGIC Appended to `dq_results` so every run is kept. Errors stop the job here,
# MAGIC before silver reads anything.

# COMMAND ----------

dq_df = (spark.createDataFrame(dq.to_rows(results))
         .withColumn("run_id", F.lit(RUN_ID)).withColumn("checked_at", F.current_timestamp()))
dq_df.write.mode("append").option("mergeSchema", "true").saveAsTable(config.table("dq_results"))
display(dq_df.orderBy("passed", "table", "check"))
dq.raise_on_failure(results)
