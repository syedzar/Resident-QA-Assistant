# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Chunks: every source → citable retrieval units (Delta)
# MAGIC | Source | Split at | Citation example |
# MAGIC |---|---|---|
# MAGIC | e-Laws Act + regulation | numbered section | `O. Reg. 246/22, s. 54 (Falls prevention and management)` |
# MAGIC | RNAO / PHO PDFs | page (running headers stripped, section tab kept) | `RNAO falls guideline, PDF p. 37 (Recommendations)` |
# MAGIC | Drug labels | label section | `Drug label: warfarin sodium > Boxed Warning` |
# MAGIC | Resident profiles | `##` profile section | `Resident 204 > Active medications (as of 2026-11-04)` |
# MAGIC
# MAGIC Then each unit is packed into windows of ≤ `chunk_size` words with `overlap` words carried over.
# MAGIC Both are **job parameters**. Each configuration is written to its own slice of `gold_chunks`
# MAGIC (`replaceWhere config_id = ...`), so stage 8 can compare configurations side by side.

# COMMAND ----------

import os, sys, uuid
sys.path.insert(0, os.path.abspath("../../src"))

import pandas as pd
from pyspark.sql import functions as F
from rcqa import config, dq, privacy
from rcqa.chunking import ChunkConfig, chunk_label_section, chunk_law, chunk_pdf, chunk_profile

dbutils.widgets.text("chunk_size", "200")
dbutils.widgets.text("overlap", "40")
dbutils.widgets.text("run_id", "")
CFG = ChunkConfig(int(dbutils.widgets.get("chunk_size")), int(dbutils.widgets.get("overlap")))
RUN_ID = dbutils.widgets.get("run_id") or str(uuid.uuid4())
print("config:", CFG.config_id)

# COMMAND ----------

# MAGIC %md ## Build chunks
# MAGIC About 10k small text units: plain Python on the driver is simpler and fast enough here.
# MAGIC The chunkers are pure functions (`rcqa.chunking`, unit-tested), so moving this to
# MAGIC `mapInPandas` on workers later is a wrapper change, not a rewrite.

# COMMAND ----------

sources = {r.source_id: r.asDict() for r in spark.table(config.table("bronze_sources")).collect()}

def source_date(sid):
    """Edition date for citations: label effective date, guideline edition, or retrieval date for laws."""
    s = sources[sid]
    ed = s["edition"] or ""
    if ed.isdigit() and len(ed) == 8:
        return f"{ed[:4]}-{ed[4:6]}-{ed[6:]}"
    return ed if ed[:1].isdigit() else s["retrieved_at"]

chunks = []
pages = spark.table(config.table("silver_document_pages")).orderBy("source_id", "page").collect()
by_source = {}
for r in pages:
    by_source.setdefault(r.source_id, []).append(r)
for sid, rows in by_source.items():
    s = sources[sid]
    if rows[0].page is None:   # e-Laws text
        chunks += chunk_law(sid, s["title"], rows[0].text, s["citation_url"], source_date(sid), CFG)
    else:
        chunks += chunk_pdf(sid, s["document_type"], s["title"], [(r.page, r.text) for r in rows],
                            s["citation_url"], source_date(sid), CFG)

for r in spark.table(config.table("silver_drug_label_sections")).orderBy("source_id", "section_order").collect():
    chunks += chunk_label_section(r.source_id, r.title, r.section, r.text, r.citation_url, source_date(r.source_id), CFG)

profiles = spark.table(config.table("gold_resident_profiles")).select("resident_id", "profile_text", "profile_as_of")
for r in profiles.collect():
    chunks += chunk_profile(int(r.resident_id), r.profile_text, r.profile_as_of.isoformat(), CFG)

print(f"{len(chunks):,} chunks")

# COMMAND ----------

# MAGIC %md ## Write to Delta (one slice per configuration)

# COMMAND ----------

pdf = pd.DataFrame([c.to_dict() for c in chunks])
pdf["resident_id"] = pdf["resident_id"].astype("Int64")
pdf["page"] = pdf["page"].astype("Int64")
df = (spark.createDataFrame(pdf)
      .withColumn("config_id", F.lit(CFG.config_id))
      .withColumn("chunk_size", F.lit(CFG.chunk_size)).withColumn("overlap", F.lit(CFG.overlap))
      .withColumn("created_at", F.current_timestamp()))

target = config.table("gold_chunks")
if spark.catalog.tableExists(target):
    df.write.mode("overwrite").option("replaceWhere", f"config_id = '{CFG.config_id}'").saveAsTable(target)
else:
    df.write.partitionBy("config_id").saveAsTable(target)
gold_chunks = spark.table(target).where(F.col("config_id") == CFG.config_id)

display(gold_chunks.groupBy("document_type")
        .agg(F.count("*").alias("chunks"), F.countDistinct("source_id").alias("sources"),
             F.round(F.avg("n_words"), 1).alias("avg_words"), F.max("n_words").alias("max_words"))
        .orderBy("document_type"))

# COMMAND ----------

# MAGIC %md ## Chunk DQ

# COMMAND ----------

results = []
rules = dq.TableRules("gold_chunks", key=["chunk_id"],
                      not_null=["source_id", "document_type", "citation", "text", "embed_text"])
results += dq.evaluate("chunks", rules, dq.profile(gold_chunks, rules))

def check(name, value, threshold, passed):
    results.append(dq.CheckResult("chunks", "gold_chunks", name, CFG.config_id, int(value), int(threshold),
                                  bool(passed), "error"))

over = gold_chunks.where(F.col("n_words") > CFG.chunk_size).count()
check("chunks_over_size", over, 0, over == 0)
missing_sources = set(sources) - {r.source_id for r in gold_chunks.select("source_id").distinct().collect()}
check("sources_without_chunks", len(missing_sources), 0, not missing_sources)
n_res = gold_chunks.where("document_type = 'resident_profile'").select("resident_id").distinct().count()
check("residents_with_chunks", n_res, 1000, n_res == 1000)
bad_meta = gold_chunks.where("(document_type = 'resident_profile') != (resident_id IS NOT NULL) OR "
                             "(document_type != 'resident_profile' AND citation_url IS NULL)").count()
check("chunks_with_wrong_metadata", bad_meta, 0, bad_meta == 0)
print("sources without chunks:", missing_sources or "none")

# COMMAND ----------

# MAGIC %md ## Privacy gate on every chunk (+ canary through the same chunking path)

# COMMAND ----------

key = {r.patient: r.resident_id for r in spark.table(config.table("silver_resident_key_map")).collect()}
patient_rows = [r.asDict() for r in spark.table(config.table("silver_patients")).collect()]
known = privacy.KnownIdentifiers.from_patient_rows(patient_rows, key)

leaks = [(c.chunk_id, c.citation, l.kind, l.value) for c in chunks
         for l in privacy.scan_chunk(c.text, c.document_type, c.resident_id, known)]
check("privacy_leaks", len(leaks), 0, not leaks)

# Canary: a real resident's identifiers planted into a profile, chunked and scanned exactly like production.
p = next(r for r in patient_rows if r["id"] in key and r["address"])
planted = (f"# Resident {key[p['id']]}\n\n## Notes\n{p['first']} {p['last']} SSN {p['ssn']} "
           f"lives at {p['address']}, born {p['birthdate'].isoformat()}\n")
canary_found = {l.kind for c in chunk_profile(key[p["id"]], planted, "2026-01-01", CFG)
                for l in privacy.scan_chunk(c.text, c.document_type, c.resident_id, known)}
missing = privacy.canary_missing(canary_found)
check("canary_detectors_missing", len(missing), 0, not missing)

if leaks:
    display(pd.DataFrame(leaks[:20], columns=["chunk_id", "citation", "kind", "value"]))

# COMMAND ----------

dq_df = (spark.createDataFrame(pd.DataFrame(dq.to_rows(results)))
         .withColumn("run_id", F.lit(RUN_ID)).withColumn("checked_at", F.current_timestamp()))
dq_df.write.mode("append").option("mergeSchema", "true").saveAsTable(config.table("dq_results"))
display(dq_df.orderBy("passed", "check"))
dq.raise_on_failure(results)
