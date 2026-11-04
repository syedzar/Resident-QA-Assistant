"""Every source we cite must be recorded, unique, and match the file on disk."""
import csv
import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/sources/sources.csv"
DOC_TYPES = {"guideline", "ipac_guidance", "legislation", "regulation", "drug_label"}


@pytest.fixture(scope="module")
def rows():
    return list(csv.DictReader(SOURCES.open(encoding="utf-8")))


def test_source_ids_unique(rows):
    ids = [r["source_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_every_source_has_citation_metadata(rows):
    for r in rows:
        for field in ["title", "publisher", "citation_url", "retrieved_at", "sha256"]:
            assert r[field], f"{r['source_id']} missing {field}"
        assert r["citation_url"].startswith("https://"), r["source_id"]
        assert r["document_type"] in DOC_TYPES, r["source_id"]


def test_required_coverage(rows):
    types = [r["document_type"] for r in rows]
    assert types.count("drug_label") >= 20
    assert {"guideline", "ipac_guidance", "legislation", "regulation"} <= set(types)


def test_local_files_match_checksums(rows):
    """Raw files are git-ignored, so in CI this is skipped; locally it proves integrity."""
    present = [r for r in rows if (ROOT / r["local_path"]).exists()]
    if not present:
        pytest.skip("raw data not downloaded (expected in CI)")
    for r in present:
        digest = hashlib.sha256((ROOT / r["local_path"]).read_bytes()).hexdigest()
        assert digest == r["sha256"], f"{r['source_id']} changed since download"
