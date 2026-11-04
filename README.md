# Resident Care Q&A Assistant

> A cited question-answering assistant for long-term care staff, built on a
> **Databricks** medallion pipeline (PySpark + Delta Lake + MLflow + Jobs) and a
> **Qdrant** RAG pipeline with filtered vector search.
> **Synthetic data only. No real patient data is used anywhere in this project.**

<!-- Sections marked TODO get filled in as each stage is built. No number goes
     in this README unless it was measured and logged in MLflow. -->

## 1. What it does
Staff ask plain-English questions and get answers with citations to the exact
resident record or document section. If the sources don't support an answer,
it says **"I don't know."**

Example questions:
- "Has resident 204 had any falls, and what medications are they on?"
- "What does the RNAO guideline recommend after a resident falls?"
- "What does Ontario's long-term care regulation require for fall prevention?"
- "What are the main side effects listed on the label for warfarin?"

## 2. Architecture
TODO (stage 11): diagram.

```
Synthea CSVs + public docs
        │  (uploaded to a Unity Catalog Volume)
        ▼
Databricks: bronze ──► silver ──► gold (Delta tables, DQ checks, de-identification)
        │                              │
        │                        chunks + metadata (Delta)
        ▼                              ▼
  Databricks Job (scheduled)     embeddings ──► Qdrant (payload indexes:
                                                resident_id, document_type)
                                               │
                    question ──► filtered search ──► LLM with citation prompt
                                               │
                                     Streamlit app (answer + sources)
        MLflow: every retrieval experiment logged
```

## 3. Data sources
Every cited source is recorded in [`data/sources/sources.csv`](data/sources/sources.csv)
with its citation URL, SHA-256 and retrieval date. Raw files are git-ignored and
rebuilt by the scripts below. Publishers keep copyright; files are used for
non-commercial education and are not redistributed in this repo.

| Source | Type | What it is |
|---|---|---|
| [Synthea](https://github.com/synthetichealth/synthea) | residents | 1,000 living synthetic patients aged 65–95 (+773 deceased during simulated history), seed 42, CSV export |
| [RNAO: Preventing Falls and Reducing Injury from Falls, 4th ed. (2017)](https://rnao.ca/bpg/guidelines/prevention-falls-and-fall-injuries) | guideline | Fall prevention best practice |
| [RNAO: Delirium, Dementia, and Depression in Older Adults, 2nd ed. (2016)](https://rnao.ca/media/3485/download) | guideline | Assessment and care |
| [RNAO: Pressure Injury Management, 4th ed. (2024)](https://rnao.ca/bpg/guidelines/pressure-injuries) | guideline | Risk assessment, prevention, treatment |
| [PHO: IPAC for Long-Term Care Homes (2020)](https://www.publichealthontario.ca/en/Health-Topics/Infection-Prevention-Control/Long-Term-Care) | ipac_guidance | Infection prevention key principles |
| [Fixing Long-Term Care Act, 2021](https://www.ontario.ca/laws/statute/21f39) | legislation | e-Laws current consolidation |
| [O. Reg. 246/22: General](https://www.ontario.ca/laws/regulation/220246) | regulation | e-Laws current consolidation (falls program: s. 54) |
| openFDA / DailyMed drug labels ×25 | drug_label | Top 25 ongoing medications among living residents, matched by RxNorm code |

Reproduce:
```bash
bash scripts/generate_residents.sh          # needs Java 17+ and tools/synthea-with-dependencies.jar
python scripts/download_sources.py          # documents + drug labels -> sources.csv
powershell -ExecutionPolicy Bypass -File scripts/convert_elaws_docs.ps1   # e-Laws .doc -> .txt (needs Word)
```

## 4. Pipeline (Databricks)
All tables live in Unity Catalog under `workspace.rcqa`, stored as Delta.

| Layer | Notebook | Tables | What happens |
|---|---|---|---|
| Raw | `scripts/upload_to_volume.sh` | Volume `/Volumes/workspace/rcqa/raw` | 9 clinical Synthea tables (gzipped 900 MB → 58 MB), PDFs, law text, label JSON |
| Bronze | `01_bronze` | `bronze_<table>`, `bronze_documents`, `bronze_drug_labels`, `bronze_sources` | Raw as-is, every CSV column a string, plus `_source_file` and `_ingested_at` lineage |
| Silver | `02_silver` | `silver_<table>`, `silver_document_pages`, `silver_drug_label_sections` | snake_case, blanks → NULL, typed dates/numbers, exact duplicates removed; PDFs per page, labels per section |
| Gold | `03_gold` | `gold_residents`, `gold_conditions`, `gold_medications`, `gold_encounters`, `gold_allergies`, `gold_fall_risk`, `gold_resident_profiles` | Living residents only, de-identified (see Privacy); one profile row per resident with sorted arrays + rendered `profile_text` |

**Data quality.** Every layer writes to `dq_results` (one row per check per run):
row counts, null counts on required fields, duplicate keys, exact duplicate rows,
and **orphan foreign keys** (e.g. a medication pointing at a patient or encounter
that doesn't exist). `error` checks stop the job; `warn` checks are recorded only.
The measuring (`profile`, Spark) and judging (`evaluate`, plain Python) halves are
split so the rules are unit-tested without a cluster.

TODO (stages 5, 9): chunking, job schedule.

## 5. Retrieval + generation (Qdrant RAG)
TODO (stages 6–7): embedding model, collection schema, filters, prompt.

## 6. Evaluation and results
TODO (stage 8): 50-question eval set, metrics, baseline vs. best table.

| Run | Chunk size | Overlap | top-k | Filters | Hit@k | Unsupported-answer rate |
|-----|-----------|---------|-------|---------|-------|-------------------------|
| baseline | – | – | – | – | – | – |
| best     | – | – | – | – | – | – |

## 7. Privacy
The data is synthetic, but the pipeline is built as if it weren't.

**De-identification by construction (silver → gold).** Gold columns are an allow-list
(`src/rcqa/gold.py`), so nothing reaches gold unless it is named there.

| Identifier | In gold? |
|---|---|
| Names, maiden name | removed |
| SSN, driver's licence, passport | removed |
| Street address, city, county, ZIP, lat/lon, birthplace | removed |
| Synthea patient / encounter / provider / payer UUIDs | removed; patient → short `resident_id` (101–1100) |
| Exact birth date | removed → age in years, **capped at 90**, plus 5-year age band |
| Deceased patients | excluded (gold = current residents) |
| Clinical event dates | **kept** (staff need "when") — see limitation below |

The UUID ↔ `resident_id` map lives only in the restricted `silver_resident_key_map`.
Ids are assigned in salted-hash order, so the number reveals nothing about the person.

**Privacy gate (blocks the job).** `03_gold` scans every row of every gold table
(~298k rows, serialised to JSON) plus every rendered profile, using two independent nets:
format patterns (SSN, UUID, licence, passport, phone, Synthea-style names) and known
values from silver (every name, ID and address, plus each resident's *own* birth date).
A **canary** scans 20 deliberately leaky rows each run and must find leaks, which proves
the gate can still fail.

| Check (latest run) | Result |
|---|---|
| Leaks in 7 gold tables + 1,000 profile texts | **0** |
| Canary leaks detected (must be > 0) | 160 |

`pytest` re-runs the scan from a laptop against live gold (`tests/test_privacy_live.py`,
skipped in CI where there are no credentials) and unit-tests every identifier type.

**Known limitation.** Real-world Safe Harbor de-identification would also generalise or
shift event dates (e.g. a random per-resident offset). That was left out so staff answers
keep real dates, which is acceptable only because the data is synthetic.

## 8. How to run it
TODO (stages 1, 9, 10). Local services so far:

```bash
docker compose up -d            # Qdrant on http://localhost:6333
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python scripts/check_setup.py
pytest
```

## 9. Repo layout
```
databricks/notebooks/   PySpark notebooks (bronze, silver, gold, chunking)
databricks/jobs/        Job definition for the end-to-end pipeline
src/rcqa/               Shared Python: de-identification, chunking, retrieval, prompts
app/                    Streamlit app
data/raw/               Downloaded raw data (git-ignored)
data/sources/           sources.csv — every source URL used for citations
eval/                   50-question evaluation set + results
scripts/                Local helper scripts (setup checks, upload, indexing)
tests/                  pytest suite (privacy, pipeline, search)
docs/                   Architecture diagram, demo script
```

## 10. Limitations
TODO: honest list (synthetic data, free-tier limits, not clinical advice).
