# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver: clean, typed, de-duplicated
# MAGIC * **Synthea:** snake_case names, blanks → NULL, real dates/numbers, exact duplicates removed
# MAGIC   (`rcqa.silver.clean`, unit-tested locally). Then DQ: unique keys, required fields,
# MAGIC   and **referential integrity**, so every condition/medication points at a real patient and encounter.
# MAGIC * **Documents:** PDFs → one row per page; e-Laws text → one row per document;
# MAGIC   drug labels → one row per label section. Each row carries its `source_id` and citation URL.
# MAGIC
# MAGIC Silver still holds names, SSNs and addresses. It is a restricted layer, and
# MAGIC de-identification happens on the way to gold (stage 4).

# COMMAND ----------

# MAGIC %pip install -q pypdf>=5.0

# COMMAND ----------

import dataclasses, os, sys, uuid
sys.path.insert(0, os.path.abspath("../../src"))

from pyspark.sql import functions as F, types as T
from rcqa import config, dq, documents
from rcqa.silver import RULES, clean

dbutils.widgets.text("run_id", "")
RUN_ID = dbutils.widgets.get("run_id") or str(uuid.uuid4())

def write(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(config.table(name))
    return spark.table(config.table(name))

# COMMAND ----------

# MAGIC %md ## Synthea tables

# COMMAND ----------

silver = {}
for t in config.SYNTHEA_TABLES:            # patients and encounters come first: others reference them
    silver[t] = write(clean(spark.table(config.table(f"bronze_{t}")), t), f"silver_{t}")

results = []
for t in config.SYNTHEA_TABLES:
    rules = dataclasses.replace(RULES[t], table=f"silver_{t}")
    m = dq.profile(silver[t], rules, parents=silver)
    results += dq.evaluate("silver", rules, m)
    b = spark.table(config.table(f"bronze_{t}")).count()
    print(f"silver_{t:<14} {m['rows']:>10,} rows  (bronze {b:,}, removed {b - m['rows']:,})")

# COMMAND ----------

# MAGIC %md ## Documents
# MAGIC 4 PDFs + 2 law texts + 25 labels: small enough to parse on the driver with plain Python.
# MAGIC At thousands of documents this would become a `mapInPandas` job that runs on the workers.

# COMMAND ----------

src = {r.local_path.split("/")[-1].rsplit(".", 1)[0]: r.asDict()
       for r in spark.table(config.table("bronze_sources")).collect()}

def meta(source_id):
    s = src[source_id]
    return (source_id, s["title"], s["document_type"], s["citation_url"], s["edition"])

page_rows = []
for r in spark.table(config.table("bronze_documents")).select("path", "content").collect():
    sid, ext = r.path.split("/")[-1].rsplit(".", 1)
    if ext == "pdf":
        page_rows += [(*meta(sid), page, text) for page, text in documents.pdf_pages(bytes(r.content))]
    else:  # e-Laws text: kept whole here, split into numbered sections when chunking (stage 5)
        page_rows.append((*meta(sid), None, documents.clean_text(bytes(r.content).decode("utf-8-sig"))))

doc_schema = T.StructType([T.StructField(c, T.StringType()) for c in
                           ["source_id", "title", "document_type", "citation_url", "edition"]]
                          + [T.StructField("page", T.IntegerType()), T.StructField("text", T.StringType())])
silver_docs = write(spark.createDataFrame(page_rows, doc_schema), "silver_document_pages")

label_rows = []
for r in spark.table(config.table("bronze_drug_labels")).select("_source_file", "value").collect():
    sid = r._source_file.split("/")[-1].rsplit(".", 1)[0]
    label_rows += [(*meta(sid), i, section, text)
                   for i, (section, text) in enumerate(documents.label_sections(r.value))]

label_schema = T.StructType([T.StructField(c, T.StringType()) for c in
                             ["source_id", "title", "document_type", "citation_url", "edition"]]
                            + [T.StructField("section_order", T.IntegerType()),
                               T.StructField("section", T.StringType()), T.StructField("text", T.StringType())])
silver_labels = write(spark.createDataFrame(label_rows, label_schema), "silver_drug_label_sections")

display(silver_docs.groupBy("source_id", "document_type")
        .agg(F.count("*").alias("pages"), F.sum((F.length("text") == 0).cast("int")).alias("empty_pages"),
             F.sum(F.length("text")).alias("chars")).orderBy("source_id"))

# COMMAND ----------

# MAGIC %md ## Document DQ
# MAGIC Every source in `sources.csv` must produce text; empty PDF pages (images, covers) are a warning.

# COMMAND ----------

n_sources = spark.table(config.table("bronze_sources")).count()
n_with_text = (silver_docs.select("source_id").union(silver_labels.select("source_id"))
               .where("source_id IS NOT NULL").distinct().count())
results.append(dq.CheckResult("silver", "documents", "sources_with_text", "source_id",
                              n_with_text, n_sources, n_with_text == n_sources, "error"))
empty = silver_docs.where("length(text) = 0").count()
results.append(dq.CheckResult("silver", "silver_document_pages", "empty_pages", "text",
                              empty, 0, empty == 0, "warn"))
for name, df in [("silver_document_pages", silver_docs), ("silver_drug_label_sections", silver_labels)]:
    rules = dq.TableRules(name, not_null=["source_id", "citation_url", "text"])
    results += dq.evaluate("silver", rules, dq.profile(df, rules))

# COMMAND ----------

# MAGIC %md ## Data-quality results

# COMMAND ----------

dq_df = (spark.createDataFrame(dq.to_rows(results))
         .withColumn("run_id", F.lit(RUN_ID)).withColumn("checked_at", F.current_timestamp()))
dq_df.write.mode("append").option("mergeSchema", "true").saveAsTable(config.table("dq_results"))
display(dq_df.orderBy("passed", "table", "check"))
dq.raise_on_failure(results)
