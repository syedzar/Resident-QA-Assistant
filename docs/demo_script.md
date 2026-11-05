# Demo script (about 5 minutes)

Before the demo:
- Docker Desktop is running (`docker compose up -d`): Qdrant on :6333.
- Ollama is running from the Ollama app, so it uses the GPU (check with `ollama ps`: `100% GPU`).
- Run `streamlit run app/streamlit_app.py` and open http://localhost:8501.
- Have the Databricks workspace open on: Jobs → `rcqa-pipeline` (last run), Catalog → `workspace.rcqa`,
  and Experiments → `rcqa-retrieval-eval`.

## 1. The problem (30 s)
"Staff in a long-term care home need quick answers from three places: the resident's record, Ontario's
long-term care law, and clinical guidance. They need to trust the answer, so every answer must cite its
source, and when the sources don't cover a question, the tool must say 'I don't know' instead of guessing.
All resident data here is synthetic (Synthea) and de-identified anyway."

## 2. Ask a policy question (45 s)
Mode: *Policies & guidelines*. Ask: **"What does the regulation require after a resident falls?"**
- Point at the citation: O. Reg. 246/22, s. 54, plus the RNAO falls guideline page.
- Open the source expander, then click through to e-Laws: "every citation links to the real source."
- If the caption says "citation corrected": "the model labelled a quote with the wrong source number;
  a check compares each claim's words with the sources and fixes it. I don't trust the model's numbering."

## 3. Resident questions (60 s)
Mode: *A resident*, ID **204**.
- Ask **"What medications is this resident taking?"** → metformin and vitamin B12, cited to
  *Resident 204 > Active medications*.
- Click **Summarize this resident** → handover bullets, each citing its section.
- Say: "a resident question can only retrieve that resident's chunks; the filter is applied inside Qdrant,
  so another resident's record is never even fetched."

## 4. "I don't know" (30 s)
- Resident 204: **"Does this resident have a pet dog?"** → refused.
- Policies: **"What is the capital of France?"** → refused instantly; "the best source scored 0.47, below
  the 0.65 gate, so the LLM was never called."

## 5. The pipeline behind it (60 s)
In Databricks:
- Jobs → `rcqa-pipeline`: bronze → silver → gold → chunks → embeddings, scheduled weekly.
- `dq_results`: every run writes its checks, including the privacy gate (0 leaks) and the canary
  ("every detector must fire on planted identifiers, or the job fails").
- `gold_chunks` partitioned by `config_id`: "each chunking experiment is its own slice."

## 6. The numbers (60 s)
Experiments → `rcqa-retrieval-eval`: show the baseline vs best runs side by side, and read the
headline row from the README results table (hit@5, grounded-answer rate, unsupported-answer %,
refusal rate on the 10 must-refuse questions). Say what changed between baseline and best, and one
thing that is still weak (the 3B model sometimes cites the law's own paragraph number, e.g. `[54 (2)]`, and the answer gets refused).

## 7. Close (15 s)
"Everything runs on free tiers: Databricks Free Edition, local Qdrant and a local LLM, so resident data
never leaves the machine. CI runs the 90-test suite on every push."
