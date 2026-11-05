# Interview prep

Every number here comes from a run in this repo (Databricks job/notebook output, `dq_results`,
MLflow experiment `rcqa-retrieval-eval`, or `data/eval/results.csv`). If a number isn't in one
of those, it isn't on this page.

## 30-second pitch
"I built a Q&A assistant for long-term care staff that answers from three places: a resident's
record, Ontario's long-term care law and clinical guidelines, and it always cites the source or
says it doesn't know. The data side is a Databricks medallion pipeline (bronze, silver, gold) over
1,000 synthetic Synthea residents plus 31 public documents, with data-quality checks and a privacy
gate that fails the job if any identifier leaks. Chunks are embedded with BGE-small and served
from Qdrant with per-resident filters, and a local 3B model writes the answer. I built a 60-question
eval set and tracked every run in MLflow: retrieval finds a correct source in the top 5 for all 50 answerable questions, a prompt fix took grounded answers from 86% to 94–96%, and all 10 must-refuse questions are refused."

## Resume bullets (real numbers only)
- Built a Databricks (PySpark, Delta, Unity Catalog) medallion pipeline that turns 1,000 synthetic
  residents and 31 public documents into 10,155 citable chunks, run as a scheduled 5-task Databricks
  job with data-quality checks written to a Delta table on every run.
- Designed a de-identification step and privacy gate (pattern + known-identifier scans, plus a canary
  that must trigger every detector) that scans 7 gold tables, 1,000 profiles and every chunk; 0 leaks
  in the latest run, and the job fails on any leak.
- Built a RAG service with Qdrant (filtered vector search so a question can only see one resident's
  record) and a local Ollama LLM, with code-enforced "I don't know" (retrieval gate at 0.65 cosine)
  and a citation checker that corrects mislabelled citations.
- Built a 60-question evaluation set and MLflow tracking; compared 4 chunk sizes and filter on/off:
  200-word chunks with type filters scored MRR 0.967 (vs 0.906 for 100-word); a prompt fix raised grounded answers from 86% to 94–96% (resident questions 80% → 100%) with 0% unsupported answers and 10/10 must-refuse questions refused; turning the citation checker off raised unsupported answers to 4.2%, all tracked in MLflow (15 runs).
- Found and fixed a silent Spark-on-serverless bug (`wholetext` option ignored, each JSON file split
  into ~140 rows) and added a DQ check (`rows_per_file`) so the job now fails on it.

## Questions by stage, with short answers

### 1. Setup
**Why Databricks Free Edition + local Qdrant/Ollama instead of one cloud stack?**
Free tiers only; serverless can't reach a laptop, so Delta is the source of truth and Qdrant is
loaded from an exported Parquet file. Local LLM means resident text never leaves the machine.

**How do you keep secrets out of a public repo?**
`.env` is git-ignored, `.env.example` has placeholders, the CLI uses a named profile, and CI only
deploys when repo secrets exist.

### 2. Data
**Why synthetic data?** No real patient data, ever. Synthea gives realistic structure (conditions,
meds, encounters) with no consent or privacy risk; the pipeline still treats it as if it were real.

**How did you pick documents?** Sources staff actually use: the Fixing Long-Term Care Act and
O. Reg. 246/22 (e-Laws), RNAO guidelines, PHO IPAC guidance, and 25 drug labels for the most common
ongoing medications, every one with a URL in `sources.csv` for citations.

### 3. Bronze → silver
**Why keep every bronze column a string?** Bronze is "exactly what arrived". If a value is malformed
we still have it; silver decides how to type or reject it, and that decision is testable.

**What DQ checks run, and what happens on failure?** Row counts, nulls on required fields, duplicate
keys, exact duplicates, referential integrity, and file-vs-row counts; all go to `dq_results`, and
any error-level failure raises, stopping the job before the next layer reads bad data.

### 4. Gold, de-identification, privacy test
**What did you remove and why keep dates?** Names, SSN, licence/passport, address, lat/lon, UUIDs,
exact birth date (age capped at 90). Event dates are kept because staff need "when"; real Safe Harbor
would shift them, which is listed as a limitation.

**How do you know the privacy scan works?** The canary: real identifiers are planted and sent through
the same path; if any detector doesn't fire, the job fails. A scanner that finds 0 leaks is only
believable if it is proven to find planted ones.

### 5. Chunking
**Why split by section first, then by size?** Each chunk then has one citation (s. 54, PDF p. 37,
"Boxed Warning"), and windows never mix two sections.

**How would a privacy reviewer trust chunks?** Every chunk is scanned, not just a sample, plus the canary
through `chunk_profile`; results are in `dq_results` per run.

### 6. Embeddings + Qdrant
**Pre-filter vs post-filter?** Qdrant applies the payload filter inside the search, so another
resident's chunk is never a candidate. Post-filtering top-k can return too few results and touches data
it shouldn't.

**Why is Delta the source of truth, not Qdrant?** Delta is versioned and rebuildable; Qdrant is a serving
index that can be reloaded from the export at any time (UUIDv5 ids make reloads idempotent).

### 7. RAG answers
**Why a threshold and a prompt instruction for "I don't know"?** The threshold catches off-topic questions
cheaply (LLM never called); the prompt catches relevant-but-not-covered questions.

**Limits of the overlap citation checker?** Boilerplate, negation, short sentences. Measure it with the eval:
checker on vs off. I ran it: with the checker off, unsupported answers went from 0% to 4.2% (2 of 48), so it earns its place.

### 8. Evaluation + MLflow
**How did you pick the best configuration, and how do you know the gain is real?**
Retrieval grid first (4 chunk sizes × filter on/off): w200_o40 with filters had the best MRR (0.967). Retrieval
was already 100% hit@5, so the 7 baseline misses were all generation: the model answered yes/no questions with only
a citation. One prompt rule + tolerant citation parsing → grounded 86% → 94%/96% (two runs). I reran the same config to
measure noise (±1 question), so I only claim differences bigger than that; k=3 vs k=5 is a tie.

**Why are the labels "chunking-independent"?** Resident gold = (resident, profile section); document gold = (source, an
evidence regex the chunk must contain). Any chunk size can be scored against the same key, so the comparison is fair.

### 9. Job + CI
**What does the job guarantee?** Each task runs only if the previous one succeeded, every task writes its DQ
results with the job's run id, `max_concurrent_runs: 1`, email on failure, weekly schedule.

**What did CI catch / what does it do?** pytest on every push (live tests skip without services); on main it
pulls the Databricks Git folder and redeploys the job definition, so the job always runs reviewed code.

### 10. Streamlit
**Why show sources and refusal reasons in the UI?** Staff need to check the answer, and a refusal with a reason
("best source scored 0.47, below 0.65") is something they can act on.

**How do you keep it fast?** `st.cache_resource` for the embedder, Qdrant client and LLM; measured first answer
15.2 s (cold), resident summary 5.2 s.

### 11–12. README, demo, interview
**What's still weak?** The 3B model still sometimes cites the law's paragraph number (`[54 (2)]`) instead of `[S1]` and gets refused, and one answer named PHQ-9 when the guideline's answer was GDS: grounded but wrong, which citation checks can't catch. The eval set is small (one question = 2 points, run-to-run noise is about 1 question).

**What would you do next?** A bigger, independently written eval set; a stronger model or reranker for the
failures in `data/eval/runs`; date shifting for real de-identification; Qdrant Cloud + auth for a real deployment.
