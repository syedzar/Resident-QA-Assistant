# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup check (Databricks Free Edition)
# MAGIC Run this once after importing the repo as a Git folder. It answers:
# MAGIC 1. Does Spark + Delta work, and can we create our schema and Volume?
# MAGIC 2. Does MLflow tracking work?
# MAGIC 3. **Which outside hosts can serverless compute reach?** Free Edition restricts
# MAGIC    outbound internet, so we test instead of guessing. The result decides where
# MAGIC    downloading, embedding and Qdrant upserts run.

# COMMAND ----------

CATALOG = "workspace"   # default catalog in Free Edition
SCHEMA = "rcqa"
VOLUME = "raw"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
print(f"Volume path: /Volumes/{CATALOG}/{SCHEMA}/{VOLUME}")

# COMMAND ----------

# MAGIC %md ## 1. Spark + Delta round trip

# COMMAND ----------

df = spark.range(5).withColumnRenamed("id", "n")
df.write.format("delta").mode("overwrite").saveAsTable(f"{CATALOG}.{SCHEMA}._setup_check")
assert spark.table(f"{CATALOG}.{SCHEMA}._setup_check").count() == 5
display(spark.sql(f"DESCRIBE HISTORY {CATALOG}.{SCHEMA}._setup_check"))
spark.sql(f"DROP TABLE {CATALOG}.{SCHEMA}._setup_check")
print("Delta OK")

# COMMAND ----------

# MAGIC %md ## 2. MLflow

# COMMAND ----------

import mlflow

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/rcqa-setup-check")
with mlflow.start_run(run_name="setup_check"):
    mlflow.log_param("stage", 1)
    mlflow.log_metric("ok", 1.0)
print("MLflow OK — see Experiments in the left sidebar")

# COMMAND ----------

# MAGIC %md ## 3. Outbound network access
# MAGIC Put your Qdrant Cloud URL below if you created one (leave blank otherwise).

# COMMAND ----------

import requests

QDRANT_CLOUD_URL = ""   # e.g. "https://xxxx.us-east4-0.gcp.cloud.qdrant.io:6333"

targets = {
    "GitHub (Synthea releases)": "https://github.com",
    "Hugging Face (embedding models)": "https://huggingface.co",
    "PyPI (pip install)": "https://pypi.org/simple/qdrant-client/",
    "openFDA (drug labels)": "https://api.fda.gov/drug/label.json?limit=1",
    "Gemini API": "https://generativelanguage.googleapis.com",
}
if QDRANT_CLOUD_URL:
    targets["Qdrant Cloud"] = QDRANT_CLOUD_URL

rows = []
for name, url in targets.items():
    try:
        r = requests.get(url, timeout=10)
        rows.append((name, "reachable", str(r.status_code)))
    except Exception as e:
        rows.append((name, "BLOCKED", type(e).__name__))

display(spark.createDataFrame(rows, ["target", "result", "detail"]))
