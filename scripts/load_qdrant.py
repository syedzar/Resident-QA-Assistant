"""Load the embeddings exported by 05_embeddings into Qdrant, then point the app alias at them.

    python scripts/load_qdrant.py --config w200_o40            # download from the Volume, load, set alias
    python scripts/load_qdrant.py --config w200_o40 --no-alias # load only (e.g. a tuning config)

QDRANT_URL / QDRANT_API_KEY come from .env: local Docker by default, Qdrant Cloud works the same.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rcqa import config, index
from rcqa.dbsql import PROFILE, _dbx_path
from rcqa.embed import DIM

EXPORTS = ROOT / "data" / "exports"


def download(config_id: str) -> Path:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    name = f"chunk_embeddings_{config_id}.parquet"
    local = EXPORTS / name
    subprocess.run([_dbx_path(), "fs", "cp", "--overwrite", f"dbfs:{config.VOLUME}/exports/{name}", str(local),
                    "-p", PROFILE], check=True, env={**os.environ, "MSYS_NO_PATHCONV": "1"})
    return local


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="w200_o40")
    ap.add_argument("--file", help="use a local parquet instead of downloading")
    ap.add_argument("--no-alias", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    path = Path(args.file) if args.file else download(args.config)
    df = pd.read_parquet(path)
    print(f"{len(df):,} chunks from {path.name}")

    name = index.collection_name(args.config)
    t0 = time.time()
    index.ensure_collection(client, name, DIM, recreate=True)
    n = index.upsert(client, name, df.to_dict("records"))
    count = client.count(name, exact=True).count
    print(f"{name}: wrote {n:,}, collection has {count:,} points ({time.time() - t0:.0f}s)")
    if count != len(df):
        raise SystemExit("point count does not match the export")
    if not args.no_alias:
        index.point_alias(client, name)
        print(f"alias {index.ALIAS} -> {name}")


if __name__ == "__main__":
    main()
