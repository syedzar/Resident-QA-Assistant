"""Query Databricks SQL from a laptop or CI via the Statement Execution API.

Authenticates through the Databricks CLI's saved login (profile 'rcqa' by default),
so no token is stored in this repo.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

PROFILE = os.getenv("DATABRICKS_PROFILE", "rcqa")


def _dbx_path() -> str | None:
    return os.getenv("DBX") or shutil.which("databricks") or next(
        (p for p in [os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\databricks.exe")] if os.path.exists(p)),
        None)


def available() -> bool:
    """True if the CLI is installed and logged in (used to skip live tests in CI)."""
    dbx = _dbx_path()
    if not dbx:
        return False
    r = subprocess.run([dbx, "current-user", "me", "-p", PROFILE], capture_output=True, text=True)
    return r.returncode == 0


def _dbx(*args: str) -> dict:
    out = subprocess.run([_dbx_path(), *args, "-p", PROFILE, "-o", "json"],
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def query(sql: str) -> tuple[list[str], list[list]]:
    """Run ``sql`` on the first SQL warehouse; return (column names, rows). Rows are strings or None."""
    warehouse = _dbx("warehouses", "list")[0]["id"]
    body = json.dumps({"warehouse_id": warehouse, "statement": sql, "wait_timeout": "50s"})
    res = _dbx("api", "post", "/api/2.0/sql/statements", "--json", body)
    sid = res["statement_id"]
    while res["status"]["state"] in ("PENDING", "RUNNING"):
        res = _dbx("api", "get", f"/api/2.0/sql/statements/{sid}")
    if res["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(res["status"])
    cols = [c["name"] for c in res["manifest"]["schema"]["columns"]]
    chunk = res.get("result", {})
    rows = list(chunk.get("data_array", []))
    while chunk.get("next_chunk_index") is not None:   # large results come back in chunks
        chunk = _dbx("api", "get", f"/api/2.0/sql/statements/{sid}/result/chunks/{chunk['next_chunk_index']}")
        rows += chunk.get("data_array", [])
    return cols, rows
