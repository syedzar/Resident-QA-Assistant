"""Run a SQL query on the Databricks SQL warehouse and print the result as a table.

    python scripts/sql.py "SELECT * FROM workspace.rcqa.dq_results LIMIT 5"

Uses the Databricks CLI's saved login (profile 'rcqa' by default), so no token is
stored in this repo. Handy for checking pipeline output from the terminal.
"""
import json
import os
import shutil
import subprocess
import sys

PROFILE = os.getenv("DATABRICKS_PROFILE", "rcqa")
DBX = os.getenv("DBX") or shutil.which("databricks") or os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\databricks.exe")


def dbx(*args: str) -> dict:
    out = subprocess.run([DBX, *args, "-p", PROFILE, "-o", "json"], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def query(sql: str) -> tuple[list[str], list[list]]:
    warehouse = dbx("warehouses", "list")[0]["id"]
    body = json.dumps({"warehouse_id": warehouse, "statement": sql, "wait_timeout": "50s"})
    res = dbx("api", "post", "/api/2.0/sql/statements", "--json", body)
    while res["status"]["state"] in ("PENDING", "RUNNING"):
        res = dbx("api", "get", f"/api/2.0/sql/statements/{res['statement_id']}")
    if res["status"]["state"] != "SUCCEEDED":
        raise SystemExit(res["status"])
    cols = [c["name"] for c in res["manifest"]["schema"]["columns"]]
    return cols, res.get("result", {}).get("data_array", [])


if __name__ == "__main__":
    cols, rows = query(sys.argv[1])
    widths = [max(len(str(x)) for x in [c, *(r[i] for r in rows)]) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
