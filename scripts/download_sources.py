"""Download every public source document and record it in data/sources/sources.csv.

Two kinds of sources:
  1. Documents listed by hand in data/sources/documents_manifest.csv
     (RNAO guidelines, PHO guidance, Ontario Act + regulation).
  2. Drug labels from openFDA for the most common medications in the
     generated Synthea population. Looked up by the exact RxNorm code
     Synthea uses, so every label matches a drug a resident actually takes.

Every row in sources.csv has the URL used for citations, a SHA-256 of the file
and the retrieval date, so anyone can check they have the same documents.

Run from the repo root (after scripts/generate_residents.sh):
    python scripts/download_sources.py
Standard library only.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/sources/documents_manifest.csv"
OUT_CSV = ROOT / "data/sources/sources.csv"
DOCS_DIR = ROOT / "data/raw/docs"
LABELS_DIR = ROOT / "data/raw/drug_labels"
SYNTHEA = ROOT / "data/raw/synthea/csv"

USER_AGENT = "resident-care-qa/0.1 (educational portfolio project)"
N_DRUGS = 25
ALWAYS_INCLUDE = ["warfarin"]  # in the data and in the project's example questions
MAGIC = {"pdf": b"%PDF-", "doc": b"\xd0\xcf\x11\xe0"}  # PDF header / OLE2 (legacy Word)

FIELDS = ["source_id", "title", "publisher", "document_type", "edition", "citation_url",
          "download_url", "local_path", "bytes", "sha256", "retrieved_at", "notes"]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def today() -> str:
    return dt.date.today().isoformat()


# ---------------------------------------------------------------- documents
def download_documents() -> list[dict]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in csv.DictReader(MANIFEST.open(encoding="utf-8")):
        data = fetch(m["download_url"])
        if not data.startswith(MAGIC[m["format"]]):
            raise ValueError(f"{m['source_id']}: response is not a {m['format']} file")
        path = DOCS_DIR / f"{m['source_id']}.{m['format']}"
        path.write_bytes(data)
        rows.append({**{k: m[k] for k in FIELDS if k in m},
                     "local_path": path.relative_to(ROOT).as_posix(),
                     "bytes": len(data), "sha256": sha256(data),
                     "retrieved_at": today(),
                     "notes": "text extracted to .txt by scripts/convert_elaws_docs.ps1"
                              if m["format"] == "doc" else ""})
        print(f"  {m['source_id']:<20} {len(data):>10,} bytes")
    return rows


# ---------------------------------------------------------------- drug labels
def rank_resident_medications() -> list[tuple[str, str, int]]:
    """(rxnorm_code, description, n_living_residents), most common first.

    Living residents only. Drugs only ever dispensed once (anaesthetics,
    one-off surgical injections) are dropped: they are not part of a
    resident's ongoing medication list.
    """
    alive = {r["Id"] for r in csv.DictReader((SYNTHEA / "patients.csv").open(encoding="utf-8"))
             if not r["DEATHDATE"]}
    patients, max_dispenses, desc = defaultdict(set), defaultdict(int), {}
    for r in csv.DictReader((SYNTHEA / "medications.csv").open(encoding="utf-8")):
        if r["PATIENT"] in alive:
            patients[r["CODE"]].add(r["PATIENT"])
            max_dispenses[r["CODE"]] = max(max_dispenses[r["CODE"]], int(r["DISPENSES"] or 0))
            desc[r["CODE"]] = r["DESCRIPTION"]
    ranked = [(c, desc[c], len(p)) for c, p in patients.items() if max_dispenses[c] > 1]
    return sorted(ranked, key=lambda t: -t[2])


def best_label(rxcui: str) -> dict | None:
    """Most complete current label for an RxNorm code (prefers labels with warnings + adverse reactions)."""
    q = urllib.parse.quote(f'openfda.rxcui:"{rxcui}"')
    try:
        results = json.loads(fetch(f"https://api.fda.gov/drug/label.json?search={q}&limit=20"))["results"]
    except urllib.error.HTTPError as e:
        if e.code == 404:  # openFDA returns 404 for "no matches"
            return None
        raise
    def score(lbl):
        has = lambda *ks: any(k in lbl for k in ks)
        return (has("adverse_reactions"), has("warnings_and_cautions", "warnings", "boxed_warning"),
                lbl.get("effective_time", ""))
    return max(results, key=score)


def download_drug_labels() -> list[dict]:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    ranked = rank_resident_medications()
    chosen, seen_generics = [], set()

    def take(code, description, n, reason):
        label = best_label(code)
        time.sleep(0.3)  # stay well under openFDA's unauthenticated rate limit
        if label is None:
            print(f"  (no openFDA label for {code} {description})")
            return
        generic = "; ".join(label.get("openfda", {}).get("generic_name", [description])).lower()
        if generic in seen_generics:  # e.g. simvastatin 10 mg and 20 mg -> one label
            return
        seen_generics.add(generic)
        chosen.append((code, description, n, reason, generic, label))

    for code, description, n in ranked:
        if len(chosen) >= N_DRUGS:
            break
        take(code, description, n, "ranked by number of residents taking it")
    for name in ALWAYS_INCLUDE:
        if not any(name in g for *_, g, _ in chosen):
            for code, description, n in ranked:
                if name in description.lower():
                    take(code, description, n, "always included (example question)")
                    break

    rows = []
    for code, description, n, reason, generic, label in chosen:
        data = json.dumps(label, indent=1, ensure_ascii=False).encode("utf-8")
        sid = "label_" + generic.split(";")[0].split(" and ")[0].strip().replace(" ", "_")[:40]
        path = LABELS_DIR / f"{sid}.json"
        path.write_bytes(data)
        set_id = label["set_id"]
        rows.append({
            "source_id": sid,
            "title": f"Drug label: {generic} (RxNorm {code}: {description})",
            "publisher": "U.S. FDA via openFDA / DailyMed",
            "document_type": "drug_label",
            "edition": label.get("effective_time", ""),
            "citation_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
            "download_url": f'https://api.fda.gov/drug/label.json?search=set_id:"{set_id}"',
            "local_path": path.relative_to(ROOT).as_posix(),
            "bytes": len(data), "sha256": sha256(data), "retrieved_at": today(),
            "notes": f"{n} living residents; {reason}",
        })
        print(f"  {sid:<34} {n:>4} residents")
    return rows


def main() -> int:
    if not (SYNTHEA / "medications.csv").exists():
        print("Run scripts/generate_residents.sh first.")
        return 1
    print("Documents:")
    rows = download_documents()
    print("Drug labels:")
    rows += download_drug_labels()
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} sources to {OUT_CSV.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
