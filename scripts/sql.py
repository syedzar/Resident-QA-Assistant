"""Run a SQL query on Databricks and print the result as a table.

    python scripts/sql.py "SELECT * FROM workspace.rcqa.dq_results LIMIT 5"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcqa.dbsql import query  # noqa: E402

if __name__ == "__main__":
    cols, rows = query(sys.argv[1])
    widths = [max(len(str(x)) for x in [c, *(r[i] for r in rows)]) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(r, widths)))
