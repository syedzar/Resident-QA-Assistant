"""Embed a TUNING chunk configuration locally and load it into Qdrant.

    python scripts/build_tuning_collection.py --config w300_o60

The chunks come from Databricks (`gold_chunks`, written and privacy-gated by 04_chunks with that
chunk_size/overlap). Only the embedding runs locally: the same BGE-small model takes ~48 min per
config on the Free Edition serverless driver vs ~1 min on a laptop CPU (measured; see README).
The baseline/production config (w200_o40) is embedded in Databricks by 05_embeddings.
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

from rcqa import index
from rcqa.dbsql import query
from rcqa.embed import DIM, Embedder

COLS = ["chunk_id", "source_id", "document_type", "title", "section", "page", "resident_id", "source_date",
        "citation", "citation_url", "text", "config_id", "embed_text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")

    t0 = time.time()
    cols, rows = query(f"SELECT {', '.join(COLS)} FROM workspace.rcqa.gold_chunks "
                       f"WHERE config_id = '{args.config}' ORDER BY chunk_id")
    recs = [dict(zip(cols, r)) for r in rows]
    if not recs:
        raise SystemExit(f"no chunks for {args.config}: run 04_chunks with that chunk_size/overlap first")
    print(f"{len(recs):,} chunks fetched in {time.time() - t0:.0f}s")

    t1 = time.time()
    vecs = Embedder().embed_docs([r["embed_text"] for r in recs])
    print(f"embedded in {time.time() - t1:.0f}s ({len(recs) / (time.time() - t1):.0f}/s)")
    for r, v in zip(recs, vecs):
        r["vector"] = v

    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    name = index.collection_name(args.config)
    index.ensure_collection(client, name, DIM, recreate=True)
    index.upsert(client, name, recs)
    print(f"{name}: {client.count(name, exact=True).count:,} points")


if __name__ == "__main__":
    main()
