# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Embeddings: gold_chunks → vectors (Delta) → export for Qdrant
# MAGIC * **Model:** `BAAI/bge-small-en-v1.5` (384-d, free) via fastembed/ONNX. The same model embeds queries in the app.
# MAGIC * **Where:** on the driver in batches (see the Free Edition note below for why not `mapInPandas`).
# MAGIC * **Why Delta, then Qdrant:** Free Edition serverless can't reach a Qdrant running on a laptop. Vectors are
# MAGIC   stored in `gold_chunk_embeddings` (source of truth, versioned) and exported as one Parquet file to the
# MAGIC   Volume; `scripts/load_qdrant.py` loads it into any Qdrant (local Docker now, Qdrant Cloud the same way).

# COMMAND ----------

# MAGIC %pip install -q fastembed==0.9.1

# COMMAND ----------

import os, sys, time, uuid
sys.path.insert(0, os.path.abspath("../../src"))

import pandas as pd
from pyspark.sql import functions as F, types as T
from rcqa import config, dq
from rcqa.embed import DIM, MODEL

dbutils.widgets.text("config_id", "w200_o40")
dbutils.widgets.text("run_id", "")
CONFIG_ID = dbutils.widgets.get("config_id")
RUN_ID = dbutils.widgets.get("run_id") or str(uuid.uuid4())

chunks = spark.table(config.table("gold_chunks")).where(F.col("config_id") == CONFIG_ID)
n_chunks = chunks.count()
print(f"{CONFIG_ID}: {n_chunks:,} chunks to embed with {MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Free Edition limitation → workaround.** The first design embedded with `mapInPandas` on workers.
# MAGIC On Free Edition serverless that failed twice: workers can't download the model (no internet), and after
# MAGIC caching it in the Volume the UDF tasks died with an opaque `INTERNAL_ERROR` after ~20 min (Python UDF
# MAGIC resources are tightly limited). 10k short texts take ~1-2 min on one CPU, so we embed **on the driver**
# MAGIC in batches. Same model, same Delta output; on a paid cluster this cell goes back to `mapInPandas`.

# COMMAND ----------

from rcqa.embed import Embedder

model = Embedder(cache_dir="/tmp/fastembed_cache")      # driver has internet; downloads once per session
rows = chunks.select("chunk_id", "embed_text").orderBy("chunk_id").toPandas()

t0, vecs, BATCH = time.time(), [], 512
for i in range(0, len(rows), BATCH):
    vecs += model.embed_docs(rows["embed_text"].iloc[i:i + BATCH].tolist())
    if (i // BATCH) % 5 == 0:
        print(f"{i + BATCH:>6,}/{len(rows):,}  {time.time() - t0:.0f}s")
embed_seconds = time.time() - t0
print(f"embedded {len(vecs):,} chunks in {embed_seconds:.0f}s ({len(vecs) / embed_seconds:.0f}/s)")

schema = T.StructType([T.StructField("chunk_id", T.StringType()),
                       T.StructField("embedding", T.ArrayType(T.FloatType()))])
vectors = (spark.createDataFrame(pd.DataFrame({"chunk_id": rows["chunk_id"], "embedding": vecs}), schema)
           .withColumn("config_id", F.lit(CONFIG_ID)).withColumn("model", F.lit(MODEL))
           .withColumn("embedded_at", F.current_timestamp()))

target = config.table("gold_chunk_embeddings")
if spark.catalog.tableExists(target):
    vectors.write.mode("overwrite").option("replaceWhere", f"config_id = '{CONFIG_ID}'").saveAsTable(target)
else:
    vectors.write.partitionBy("config_id").saveAsTable(target)
emb = spark.table(target).where(F.col("config_id") == CONFIG_ID)
seconds = embed_seconds

# COMMAND ----------

# MAGIC %md ## Embedding DQ

# COMMAND ----------

stats = emb.select(
    F.count("*").alias("n"),
    F.sum(F.when(F.size("embedding") != DIM, 1).otherwise(0)).alias("wrong_dim"),
    F.sum(F.when(F.abs(F.sqrt(F.aggregate("embedding", F.lit(0.0), lambda a, x: a + x * x)) - 1) > 0.01, 1)
          .otherwise(0)).alias("not_unit_norm"),
).first()
missing = chunks.join(emb, "chunk_id", "left_anti").count()

results = []
def check(name, value, threshold, passed):
    results.append(dq.CheckResult("embeddings", "gold_chunk_embeddings", name, CONFIG_ID, int(value), int(threshold),
                                  bool(passed), "error"))
check("chunks_without_vector", missing, 0, missing == 0)
check("vectors_vs_chunks", stats.n, n_chunks, stats.n == n_chunks)
check("wrong_dimension", stats.wrong_dim, 0, stats.wrong_dim == 0)
check("not_unit_norm", stats.not_unit_norm, 0, stats.not_unit_norm == 0)

dq_df = (spark.createDataFrame(pd.DataFrame(dq.to_rows(results)))
         .withColumn("run_id", F.lit(RUN_ID)).withColumn("checked_at", F.current_timestamp()))
dq_df.write.mode("append").option("mergeSchema", "true").saveAsTable(config.table("dq_results"))
display(dq_df)
dq.raise_on_failure(results)

# COMMAND ----------

# MAGIC %md ## Export one Parquet file (chunk payload + vector) for the Qdrant loader

# COMMAND ----------

export_dir = f"{config.VOLUME}/exports"
os.makedirs(export_dir, exist_ok=True)
path = f"{export_dir}/chunk_embeddings_{CONFIG_ID}.parquet"
export = (chunks.join(emb.select("chunk_id", "embedding"), "chunk_id")
          .select("chunk_id", "source_id", "document_type", "title", "section", "page", "resident_id",
                  "source_date", "citation", "citation_url", "text", "config_id",
                  F.col("embedding").alias("vector")))
export.toPandas().to_parquet(path, index=False)
print(path, f"{os.path.getsize(path) / 1e6:.1f} MB", f"model={MODEL}", f"seconds={seconds:.0f}")
dbutils.notebook.exit(f"{stats.n} vectors in {seconds:.0f}s -> {path}")
