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
| Chunks | `04_chunks` | `gold_chunks` (partitioned by `config_id`) | Every source split into citable units, then into word windows; privacy-scanned before anything is embedded |
| Embeddings | `05_embeddings` | `gold_chunk_embeddings` + Parquet export in the Volume | BGE-small vectors (384-d), checked: one per chunk, right dimension, unit length |

**Data quality.** Every layer writes to `dq_results` (one row per check per run):
row counts, null counts on required fields, duplicate keys, exact duplicate rows,
and **orphan foreign keys** (e.g. a medication pointing at a patient or encounter
that doesn't exist). `error` checks stop the job; `warn` checks are recorded only.
The measuring (`profile`, Spark) and judging (`evaluate`, plain Python) halves are
split so the rules are unit-tested without a cluster.

**Chunking (`src/rcqa/chunking.py`).** Split on the document's own structure first,
then pack each unit into windows of at most `chunk_size` words with `overlap` words carried over:

| Source | Unit | Citation stored on the chunk |
|---|---|---|
| Act + regulation (e-Laws) | numbered section | `O. Reg. 246/22, s. 54 (Falls prevention and management)` |
| RNAO / PHO PDFs | page (running headers removed, section tab kept, near-empty pages dropped) | `Preventing Falls… (4th ed.), PDF p. 37 (Recommendations)` |
| Drug labels (openFDA) | label section | `Drug label: warfarin sodium > Boxed Warning` |
| Resident profiles | profile section (`## Active medications`…) | `Resident 204 > Active medications (as of 2026-11-04)` |

Each chunk carries `source_id`, `document_type`, `section`, `page`, `resident_id`
(profiles only), `source_date`, `citation`, `citation_url`, the plain `text` shown to
staff, and an `embed_text` with a context header (e.g. `O. Reg. 246/22, s. 54: Falls
prevention and management`) so a window from the middle of a section still says what
it is about. `chunk_size` / `overlap` are job parameters; each configuration is its own
`config_id` slice (`w200_o40`), written with Delta `replaceWhere`, so stage 8 can compare
configurations without overwriting each other.

| Baseline `w200_o40` (latest run) | Chunks | Sources | Avg words |
|---|---|---|---|
| Resident profiles | 6,000 | 1,000 residents | 44.0 |
| Drug labels | 1,535 | 25 | 172.4 |
| Guidelines (RNAO) | 1,001 | 3 | 161.1 |
| Regulation (O. Reg. 246/22) | 791 | 1 | 141.7 |
| IPAC guidance (PHO) | 443 | 1 | 163.4 |
| Act (FLTCA 2021) | 385 | 1 | 139.2 |
| **Total** | **10,155** | 31 + 1,000 | |

Chunk DQ (all `error`): unique `chunk_id`, required fields not null, no chunk over
`chunk_size`, every source and all 1,000 residents have chunks, `resident_id` set if and
only if the chunk is a profile, every document chunk has a URL to cite.

Bugs this caught on the first Databricks run: the Act produced **0** chunks because
Windows `

` line endings became blank lines and broke heading detection (fixed in
`clean_text`, regression test added), and 17 pages that were just a page number and a
tab label were turned into useless chunks (now dropped).

TODO (stage 9): job schedule.

## 5. Retrieval + generation (Qdrant RAG)
**Embeddings.** `BAAI/bge-small-en-v1.5` (free, 384-d) through fastembed (ONNX, no PyTorch),
so Databricks and the app run the *same* model files. Documents are embedded from
`embed_text` (chunk + context header); queries get BGE's search instruction prefix.

**Databricks → Qdrant.** Free Edition serverless can't reach a Qdrant on a laptop, so
Delta (`gold_chunk_embeddings`) is the source of truth, and `05_embeddings` exports one
Parquet file to the Volume. `scripts/load_qdrant.py` downloads it and upserts into
whatever `QDRANT_URL` points at: local Docker now, and Qdrant Cloud the same way.

**Free Edition limitation, measured.** The first design embedded with `mapInPandas` on
workers. That failed: workers have no internet (the model can't be downloaded), and with
the model cached in the Volume, the UDF tasks died with an opaque `INTERNAL_ERROR` after
~20 min. The workaround embeds on the driver in batches:

| Where | Throughput (BGE-small, CPU) | 10,155 chunks |
|---|---|---|
| Databricks Free Edition serverless driver | ~3.5 chunks/s | 2,905 s (48 min) |
| Laptop (same model, fastembed) | ~163 chunks/s | ~1 min |

**Qdrant layout (`src/rcqa/index.py`).**
- One collection per chunking config (`chunks_w200_o40`). The app reads the alias
  `rcqa_chunks`, which is swapped atomically when a better config wins (stage 8).
- Cosine distance. Point id = UUIDv5(`chunk_id`), so reloads are idempotent.
- Payload indexes: `resident_id` (integer), `document_type` and `source_id` (keyword).
  The payload carries everything needed to cite: `citation`, `citation_url`, `section`, `page`, `text`.

**Filtered search: privacy by construction.**

| Question about… | Filter |
|---|---|
| A resident (`resident_id=204`) | `resident_id = 204`, optionally OR the public types asked for (e.g. `drug_label`) |
| Rules or guidelines (no resident) | `document_type IN (public types)`, never resident profiles |

A question can therefore never retrieve another resident's chunk, even when that chunk
is the closest vector. A unit test plants exactly that case, and a live test checks it
against the real collection.

**Sanity checks on the real index** (`tests/test_search_live.py`; scored evaluation is stage 8):
"post-fall assessment" with `regulation` → top hit `O. Reg. 246/22, s. 54 (Falls
prevention and management)`; resident 204 searches return only resident 204; public
searches return no resident chunks.

**Answers (`src/rcqa/rag.py`).** LLM: `qwen2.5:3b` running locally through Ollama on the
GPU (RTX 3060, about 3–5 s per answer), at temperature 0 so evaluation runs are repeatable.
It's local, so resident chunks never leave the machine. Set `OLLAMA_MODEL` to swap models.

| Question type | What is retrieved |
|---|---|
| No resident | top-5 public chunks (optionally limited to `document_types`) |
| Resident `204` | that resident's **whole record** (~6 chunks; ranking only 6 can only drop a section) |
| Resident + references | the whole record, plus top-5 chunks of the requested public types (e.g. `drug_label`) |

**"I don't know" is enforced in code, not just requested in the prompt:**

| Guard | Refuses when | Reason code |
|---|---|---|
| Retrieval gate | best chunk scores < 0.65 (measured: on-topic ≥ 0.72, off-topic ≤ 0.61); the LLM isn't called | `low_retrieval_score` |
| Prompt | the model replies "I don't know" | `llm_idk` |
| Citation check | no valid `[S#]` citation in the answer | `no_citations` |
| Content check | the answer is only citations (e.g. `[2]`) | `no_content` |
| Error | Ollama is down or times out: refuse, never guess | `llm_error` |

**Citations are checked, not trusted.** Sources are labelled `[S1]`, `[S2]`… because laws and
labels are full of their own numbers ("7. Every resident has the right…"). The 3B model cited
the Bill of Rights' paragraph numbers until the labels changed. After generation:
- citations to sources that weren't given are dropped and counted;
- each claim's words are compared with the source it cites. If another source clearly matches
  better, the citation is corrected and counted (`recited`). In testing, the model quoted s. 54
  word for word but labelled it s. 298;
- sentences without any citation are counted (`uncited_sentences`, a stage-8 metric).

**Summarize this resident.** Every section of the record goes to a handover prompt
(`- point [S#]` bullets), with the same citation checks.

Known weaknesses of the 3B model (stage 8 will measure them): it sometimes refuses when the
record says "No active allergies recorded", and it sometimes writes citations as sentence
subjects ("[2] recommends…").

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
A **canary** sends 20 rows of real identifiers through the same serialise-and-scan path
each run, and **every detector kind** (known value, Synthea name, SSN, address, own birth
date) must fire. Just "found some leaks" isn't enough, because one broken detector
could hide behind the others.

**Every chunk is scanned again** in `04_chunks` before embedding, with its own canary
pushed through `chunk_profile()`. Resident chunks use every detector. Public-document
chunks (laws, guidelines, labels) count only known patient values (names, IDs,
addresses); format patterns there are noise. The first run flagged 13 "leaks", and all
were footnote markers glued to words (`Learning45`), journal IDs and DOI UUIDs.

| Check (latest run) | Result |
|---|---|
| Leaks in 7 gold tables + 1,000 profile texts | **0** |
| Leaks in 10,155 chunks | **0** |
| Detector kinds the canaries missed (gold, chunks) | 0, 0 |

`pytest` re-runs the scan from a laptop against live gold profiles and every chunk (`tests/test_privacy_live.py`,
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
python scripts/load_qdrant.py --config w200_o40     # after 05_embeddings has run in Databricks
python scripts/search.py "post-fall assessment" --types regulation
python scripts/search.py "is she on a blood thinner?" --resident 204 --types drug_label
ollama pull qwen2.5:3b                               # local LLM (Ollama for Windows)
python scripts/ask.py "what does the regulation say about post-fall assessment?" --types regulation
python scripts/ask.py --summarize 204
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
