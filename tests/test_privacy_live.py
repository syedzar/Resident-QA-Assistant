"""LIVE privacy test: scan every de-identified profile in Databricks gold for identifiers.

Runs only when the Databricks CLI is logged in (your laptop). In CI there are no
credentials, so it is skipped; the same scan also runs inside the gold notebook,
where any leak fails the scheduled job.
It also scans every chunk in gold_chunks (what stage 6 embeds), for every chunking config.
"""
from datetime import date

import pytest

from rcqa import dbsql
from rcqa.privacy import KnownIdentifiers, find_leaks, scan_chunk

pytestmark = pytest.mark.skipif(not dbsql.available(), reason="Databricks CLI not logged in")

T = "workspace.rcqa"


@pytest.fixture(scope="module")
def known():
    cols, rows = dbsql.query(f"SELECT p.id, first, middle, last, maiden, ssn, drivers, passport, address, "
                             f"birthdate, k.resident_id FROM {T}.silver_patients p "
                             f"LEFT JOIN {T}.silver_resident_key_map k ON p.id = k.patient")
    patients = [dict(zip(cols, r)) for r in rows]
    for p in patients:
        p["birthdate"] = date.fromisoformat(p["birthdate"]) if p["birthdate"] else None
    keymap = {p["id"]: int(p["resident_id"]) for p in patients if p["resident_id"]}
    assert len(keymap) == 1000
    return KnownIdentifiers.from_patient_rows(patients, keymap)


def test_every_gold_profile_is_free_of_identifiers(known):
    _, rows = dbsql.query(f"SELECT resident_id, profile_text FROM {T}.gold_resident_profiles")
    assert len(rows) == 1000
    leaks = {int(rid): find_leaks(text, known, resident_id=int(rid)) for rid, text in rows}
    leaks = {rid: l for rid, l in leaks.items() if l}
    assert not leaks, f"{len(leaks)} profiles leak identifiers, e.g. {list(leaks.items())[:3]}"


def test_every_chunk_is_free_of_identifiers(known):
    _, rows = dbsql.query(f"SELECT chunk_id, document_type, resident_id, text FROM {T}.gold_chunks")
    assert len(rows) > 5000
    leaks = [(cid, l) for cid, dtype, rid, text in rows
             for l in scan_chunk(text, dtype, int(rid) if rid else None, known)]
    assert not leaks, f"{len(leaks)} chunk leaks, e.g. {leaks[:3]}"


def test_gold_tables_have_no_identifier_columns():
    banned = {"patient", "id", "first", "middle", "last", "maiden", "ssn", "drivers", "passport", "address",
              "city", "county", "zip", "lat", "lon", "birthdate", "birthplace", "encounter", "provider",
              "organization", "payer"}
    _, rows = dbsql.query(f"SELECT table_name, column_name FROM workspace.information_schema.columns "
                          f"WHERE table_schema = 'rcqa' AND table_name LIKE 'gold_%'")
    found = {(t, c) for t, c in rows if c.lower() in banned}
    assert not found, found
