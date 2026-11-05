"""The privacy scanner must catch every identifier type, and must not cry wolf on normal clinical text."""
from datetime import date

import pytest

from rcqa.privacy import KnownIdentifiers, find_leaks

PATIENT = {"id": "e74367bd-7385-97dc-2c7f-251b4cd9fe29", "first": "Tennille443", "middle": "Denna536",
           "last": "Kiehn525", "maiden": "Rosenbaum794", "ssn": "999-21-9587", "drivers": "S99929408",
           "passport": "X22883891X", "address": "911 Hauck Key Suite 34", "birthdate": date(1960, 3, 1)}
KNOWN = KnownIdentifiers.from_patient_rows([PATIENT], {PATIENT["id"]: 204})


@pytest.mark.parametrize("leaked, kind", [
    ("SSN 999-21-9587 on file", "ssn"),
    ("record e74367bd-7385-97dc-2c7f-251b4cd9fe29", "uuid"),
    ("licence S99929408", "drivers_licence"),
    ("passport X22883891X", "passport"),
    ("call 555-123-4567", "phone"),
    ("seen by Tennille443", "synthea_name"),
    ("lives at 911 Hauck Key Suite 34", "address"),
    ("born 1960-03-01", "birthdate"),
    ("born March 1, 1960", "birthdate"),
    ("DOB 03/01/1960", "birthdate"),
])
def test_each_identifier_type_is_caught(leaked, kind):
    kinds = {leak.kind for leak in find_leaks(leaked, KNOWN, resident_id=204)}
    assert kind in kinds


def test_name_caught_by_known_list_even_without_digits_pattern():
    known = KnownIdentifiers(tokens={"Kiehn"})
    assert {leak.kind for leak in find_leaks("Mrs Kiehn fell", known)} == {"known_identifier"}


CLEAN_PROFILE = """# Resident 204
Sex: F | Age: 66 (age band 65-69) | Profile as of 2026-11-04

## Fall risk and falls
Latest Morse Fall Scale assessment: High Risk (MFS Score 50+), total score 55, assessed 2026-03-01.

## Active medications
- Simvastatin 20 MG Oral Tablet (since 2015-06-22; for Hyperlipidemia)
- Vitamin B12 5 MG/ML Injectable Solution (since 2020-01-01)
- 24 HR metoprolol succinate 100 MG Extended Release Oral Tablet (since 2019-04-02)
"""


def test_clean_profile_has_no_false_positives():
    assert find_leaks(CLEAN_PROFILE, KNOWN, resident_id=204) == []


def test_other_residents_birthdate_is_not_flagged():
    # 2026-03-01 is an encounter date here; it must only count as a leak for the resident born on it.
    known = KnownIdentifiers(birthdates={204: date(1960, 3, 1), 305: date(2026, 3, 1)})
    assert find_leaks(CLEAN_PROFILE, known, resident_id=204) == []
    assert find_leaks(CLEAN_PROFILE, known, resident_id=305)  # same text, but it IS 305's birth date


def test_canary_reports_missing_detectors():
    from rcqa.privacy import canary_missing
    found = {l.kind for l in find_leaks("Tennille443 999-21-9587 911 Hauck Key Suite 34 1960-03-01", KNOWN, 204)}
    assert canary_missing(found) == set()
    assert canary_missing(found - {"address"}) == {"address"}


def test_public_documents_may_contain_phone_numbers_but_not_patients():
    from rcqa.privacy import scan_chunk
    doc = ("Call RNAO at 416-599-1925 for the falls guideline. Learning45 Settings176 "
           "J Hosp Infect S01959417 doi 5fa19e13-82fe-4601-bf83-d8e84e1cef8a")   # real false positives seen in PDFs
    assert scan_chunk(doc, "guideline", None, KNOWN) == []
    assert scan_chunk(doc + " Tennille443", "guideline", None, KNOWN)            # a patient name in a doc chunk
    assert scan_chunk("phone 416-599-1925", "resident_profile", 204, KNOWN)      # but not in resident chunks


def test_privacy_scan_of_chunked_profile_catches_planted_identifiers():
    """Same path as production: render-like text -> chunk_profile -> scan_chunk."""
    from rcqa.chunking import ChunkConfig, chunk_profile
    from rcqa.privacy import canary_missing, scan_chunk
    text = ("# Resident 204\n\n## Allergies\n- Peanut\n\n## Notes\n"
            "Tennille443 Kiehn525 SSN 999-21-9587 lives at 911 Hauck Key Suite 34, born 1960-03-01\n")
    found = {l.kind for c in chunk_profile(204, text, "2026-11-04", ChunkConfig(200, 40))
             for l in scan_chunk(c.text, c.document_type, c.resident_id, KNOWN)}
    assert canary_missing(found) == set()
