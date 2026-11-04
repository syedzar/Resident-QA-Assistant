"""Local environment check: Python version, .env, and Qdrant connectivity.

Run from the repo root:  python scripts/check_setup.py
"""
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK ' if ok else 'FAIL'}] {label}{' - ' + detail if detail else ''}")
    return ok


def main() -> int:
    results = [check("Python", sys.version_info >= (3, 10), sys.version.split()[0])]

    results.append(check(".env file present", os.path.exists(".env"),
                         "copy .env.example to .env" if not os.path.exists(".env") else ""))

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=5)
        names = [c.name for c in client.get_collections().collections]
        results.append(check("Qdrant reachable", True, f"{url} | collections={names}"))
    except Exception as e:  # noqa: BLE001 - we want to report any failure
        results.append(check("Qdrant reachable", False,
                             f"{url} | {type(e).__name__}. Is Docker running? `docker compose up -d`"))

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
