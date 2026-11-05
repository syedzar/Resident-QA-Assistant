"""Run the eval set against one configuration and log it to MLflow (Databricks).

    python scripts/evaluate.py --config w200_o40 --retrieval-only --k 5           # fast: hit@k, MRR
    python scripts/evaluate.py --config w200_o40 --k 5 --run-name baseline         # + LLM answers
    python scripts/evaluate.py --config w300_o60 --k 3 --no-filter --retrieval-only

Every run: params + metrics + per-question results (artifact) in the MLflow experiment
/Users/<you>/rcqa-retrieval-eval, and one summary row appended to data/eval/results.csv.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rcqa import evaluation as ev, index, rag
from rcqa.embed import MODEL, Embedder

QUESTIONS = ROOT / "data" / "eval" / "questions.jsonl"
RESULTS = ROOT / "data" / "eval" / "results.csv"
EXPERIMENT = "/Users/zaryansyed04@gmail.com/rcqa-retrieval-eval"


def _drop_placeholder_databricks_env() -> None:
    """.env may hold DATABRICKS_HOST=<your-workspace>... placeholders; env vars override the CLI
    profile, so a placeholder would send MLflow to a host that doesn't exist."""
    for k in ("DATABRICKS_HOST", "DATABRICKS_TOKEN"):
        if not os.environ.get(k) or "<" in os.environ[k] or "your" in os.environ[k].lower():
            os.environ.pop(k, None)


def log_to_mlflow(run_file: Path) -> str:
    """Log a saved run (params, metrics, per-question results) to Databricks MLflow."""
    import mlflow
    _drop_placeholder_databricks_env()
    d = json.loads(run_file.read_text(encoding="utf-8"))
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks://rcqa"))
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=d["run_name"]) as r:
            mlflow.log_params(d["params"])
            mlflow.log_metrics({k: float(v) for k, v in d["metrics"].items()})
            mlflow.log_dict({"per_question": d["per_question"]}, "per_question.json")
            return r.info.run_id
    except Exception as e:   # keep the local result; re-log later with --log-run
        print(f"MLflow logging failed ({type(e).__name__}); saved locally: {run_file}")
        return "-"


def run(args) -> dict:
    load_dotenv(ROOT / ".env")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    collection = index.collection_name(args.config)
    retriever = rag.qdrant_retriever(client, collection, Embedder())
    items = ev.load_questions(QUESTIONS)[: args.limit or None]
    positives = [q for q in items if q["kind"] != "negative"]

    t0 = time.time()
    ranks = [ev.first_relevant_rank(ev.ranked_hits(retriever, q, max(args.k, 10), not args.no_filter), q["expected"])
             for q in positives]
    per_q = [{"id": q["id"], "kind": q["kind"], "first_relevant_rank": r} for q, r in zip(positives, ranks)]
    metrics = ev.retrieval_metrics(ranks)
    for kind in ("resident", "document"):
        sub = [r for q, r in zip(positives, ranks) if q["kind"] == kind]
        metrics[f"hit_at_{args.k}_{kind}"] = ev.retrieval_metrics(sub, (args.k,))[f"hit_at_{args.k}"]
    metrics["retrieval_seconds"] = time.time() - t0

    llm = None
    if not args.retrieval_only:
        llm = rag.OllamaLLM(args.model)
        rows, lat = [], []
        for q in items:
            t = time.time()
            types = q["document_types"] if not args.no_filter else None
            a = rag.answer(q["question"], retriever, llm, q["resident_id"], types, k=args.k,
                           min_score=args.min_score, verify=not args.no_verify)
            lat.append(time.time() - t)
            rows.append(ev.score_answer(q, a))
        metrics.update(ev.answer_metrics(rows))
        metrics["mean_answer_seconds"] = sum(lat) / len(lat)
        per_q = rows
    params = {"config_id": args.config, "chunk_size": int(args.config.split("_")[0][1:]),
              "overlap": int(args.config.split("_")[1][1:]), "k": args.k, "filter": not args.no_filter,
              "embed_model": MODEL, "llm": llm.name if llm else "-", "min_score": args.min_score,
              "citation_check": not args.no_verify, "prompt_version": rag.PROMPT_VERSION if llm else "-", "n_questions": len(items),
              "eval_set_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest()[:12],
              "mode": "retrieval" if args.retrieval_only else "answers"}
    run_name = args.run_name or f"{args.config}_k{args.k}{'_nofilter' if args.no_filter else ''}_{params['mode']}"

    # Save locally FIRST: a network problem must never throw away a 5-minute LLM run.
    run_file = ROOT / "data" / "eval" / "runs" / f"{run_name}.json"
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps({"run_name": run_name, "params": params, "metrics": metrics,
                                    "per_question": per_q}, indent=1), encoding="utf-8")
    run_id = "-" if args.no_mlflow else log_to_mlflow(run_file)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS.exists()
    fields = ["run_name", "mlflow_run_id", *params, *sorted(metrics)]
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"run_name": run_name, "mlflow_run_id": run_id, **params,
                    **{k: round(v, 4) for k, v in metrics.items()}})
    print(run_name, json.dumps({k: round(v, 3) for k, v in metrics.items()}))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="w200_o40")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-filter", action="store_true", help="search all public types, not the question's category")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--min-score", type=float, default=rag.MIN_SCORE)
    ap.add_argument("--no-verify", action="store_true", help="turn the citation checker off")
    ap.add_argument("--model", default=None, help="Ollama model (default OLLAMA_MODEL or qwen2.5:3b)")
    ap.add_argument("--run-name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--log-run", help="only log an already-saved data/eval/runs/<name>.json to MLflow")
    sys.stdout.reconfigure(encoding="utf-8")
    args = ap.parse_args()
    if args.log_run:
        print(log_to_mlflow(Path(args.log_run)))
    else:
        run(args)


if __name__ == "__main__":
    main()
