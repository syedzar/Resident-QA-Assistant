"""Ask the assistant a question from the command line (retrieval + local LLM + citations).

    python scripts/ask.py "what does the regulation say about post-fall assessment?"
    python scripts/ask.py "is this resident on a blood thinner?" --resident 204 --types drug_label
    python scripts/ask.py --summarize 204
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rcqa import index, rag
from rcqa.embed import Embedder


def show(a: rag.Answer, seconds: float) -> None:
    print(f"\n{a.text}\n")
    print(f"[{a.reason}] top score {a.top_score:.3f}, {seconds:.1f}s, "
          f"invalid citations {a.invalid_citations}, re-cited {a.recited}, uncited sentences {a.uncited_sentences}")
    for s in a.cited:
        print(f"  [{s.n}] {s.citation}" + (f"  {s.citation_url}" if s.citation_url else ""))
    if not a.supported and a.raw:
        print(f"  (model said: {a.raw[:200]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--resident", type=int)
    ap.add_argument("--types", nargs="*")
    ap.add_argument("--summarize", type=int, metavar="RESIDENT_ID")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(ROOT / ".env")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    llm = rag.OllamaLLM()
    t0 = time.time()
    if args.summarize:
        a = rag.summarize_resident(args.summarize, index.resident_chunks(client, index.ALIAS, args.summarize), llm)
    else:
        retriever = rag.qdrant_retriever(client, index.ALIAS, Embedder())
        a = rag.answer(args.question, retriever, llm, args.resident, args.types, k=args.k)
    show(a, time.time() - t0)


if __name__ == "__main__":
    main()
