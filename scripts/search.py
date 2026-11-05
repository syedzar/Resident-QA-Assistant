"""Try filtered search from the command line.

    python scripts/search.py "what does the regulation say about post-fall assessment?"
    python scripts/search.py "is she on a blood thinner?" --resident 204 --types drug_label
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rcqa import index
from rcqa.embed import Embedder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--resident", type=int)
    ap.add_argument("--types", nargs="*", help="document types, e.g. regulation guideline drug_label")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--collection", default=index.ALIAS)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console default (cp1252) chokes on "≥", "µ"

    load_dotenv(ROOT / ".env")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    vec = Embedder().embed_query(args.question)
    for h in index.search(client, args.collection, vec, args.k, args.resident, args.types):
        print(f"{h['score']:.3f}  [{h['document_type']}] {h['citation']}")
        print(f"       {h['text'][:140].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
