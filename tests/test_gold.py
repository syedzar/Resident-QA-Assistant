"""Gold transforms on tiny fixtures: de-identification by construction, ages, profiles, rendering."""
from datetime import date

import pytest

from rcqa import gold
from rcqa.privacy import KnownIdentifiers, find_leaks
from rcqa.profile_text import NO_FALLS_NOTE, render
from rcqa.silver import clean

P_COLS = ["Id", "BIRTHDATE", "DEATHDATE", "SSN", "DRIVERS", "PASSPORT", "FIRST", "LAST", "ADDRESS",
          "CITY", "ZIP", "GENDER"]
PATIENTS = [
    ("p-alive", "1960-03-01", "", "999-21-9587", "S99929408", "X22883891X", "Tennille443", "Kiehn525",
     "911 Hauck Key Suite 34", "Arlington", "02476", "F"),
    ("p-old", "1931-12-01", "", "999-11-1111", "S11111111", "", "Abe12", "Old34", "1 Main St", "Boston", "02100", "M"),
    ("p-dead", "1940-01-01", "2020-01-01", "999-22-2222", "", "", "Gone56", "Away78", "2 Elm St", "Boston", "02101", "M"),
]
AS_OF = "2026-11-04"


@pytest.fixture
def silver(make_df):
    t = lambda rows, cols, name: clean(make_df(rows, cols), name)
    patients = t(PATIENTS, P_COLS, "patients")
    conditions = t([("2019-03-02", "2019-05-01", "p-alive", "e1", "1", "Fracture of forearm (disorder)"),
                    ("1990-01-01", "", "p-alive", "e1", "2", "Essential hypertension (disorder)"),
                    ("2000-01-01", "", "p-alive", "e1", "3", "Fractured dental filling (finding)"),
                    ("2010-01-01", "", "p-dead", "e9", "2", "Essential hypertension (disorder)")],
                   ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"], "conditions")
    meds = t([("2015-06-22T02:23:04Z", "", "p-alive", "e1", "1", "Simvastatin 20 MG Oral Tablet", "Hyperlipidemia", "5"),
              ("2001-01-01T00:00:00Z", "2002-01-01T00:00:00Z", "p-alive", "e1", "2", "Old drug", "", "1")],
             ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "REASONDESCRIPTION", "DISPENSES"],
             "medications")
    enc = t([("e1", "2026-10-01T10:00:00Z", "p-alive", "org-uuid", "prov-uuid", "ambulatory", "Checkup", "")],
            ["Id", "START", "PATIENT", "ORGANIZATION", "PROVIDER", "ENCOUNTERCLASS", "DESCRIPTION",
             "REASONDESCRIPTION"], "encounters")
    obs = t([("2026-03-01T00:00:00Z", "p-alive", "e1", gold.FALL_LEVEL, "High Risk (MFS Score 50+)"),
             ("2026-03-01T00:00:00Z", "p-alive", "e1", gold.FALL_SCORE, "55"),
             ("2025-03-01T00:00:00Z", "p-alive", "e1", gold.FALL_LEVEL, "Low Risk (MFS Score 0 - 24)")],
            ["DATE", "PATIENT", "ENCOUNTER", "DESCRIPTION", "VALUE"], "observations")
    allg = t([("2000-01-01", "", "p-alive", "e1", "1", "Peanut (substance)", "allergy", "food", "Hives", "MILD")],
             ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION", "TYPE", "CATEGORY", "DESCRIPTION1",
              "SEVERITY1"], "allergies")
    return dict(patients=patients, conditions=conditions, meds=meds, enc=enc, obs=obs, allg=allg)


@pytest.fixture
def built(silver):
    keymap = gold.resident_key_map(silver["patients"], salt="test")
    res = gold.residents(silver["patients"], keymap, AS_OF)
    cond = gold.conditions(silver["conditions"], keymap)
    meds = gold.medications(silver["meds"], keymap)
    enc = gold.encounters(silver["enc"], keymap)
    falls = gold.fall_risk(silver["obs"], keymap)
    allg = gold.allergies(silver["allg"], keymap)
    prof = gold.profiles(res, cond, meds, enc, falls, allg)
    return dict(keymap=keymap, res=res, cond=cond, meds=meds, enc=enc, falls=falls, prof=prof)


def test_keymap_living_only_short_stable_ids(silver, built):
    ids = {r.patient: r.resident_id for r in built["keymap"].collect()}
    assert set(ids) == {"p-alive", "p-old"} and set(ids.values()) == {101, 102}
    again = {r.patient: r.resident_id for r in gold.resident_key_map(silver["patients"], "test").collect()}
    assert again == ids                                    # stable across reruns


def test_gold_tables_contain_no_identifier_columns(built):
    banned = {"patient", "id", "first", "last", "ssn", "drivers", "passport", "address", "city", "zip",
              "birthdate", "deathdate", "encounter", "organization", "provider"}
    for name in ["res", "cond", "meds", "enc", "falls", "prof"]:
        assert not banned & set(built[name].columns), name


def test_age_is_capped_at_90_and_banded(built):
    rows = {r.age_band: r.age for r in built["res"].collect()}
    assert rows == {"65-69": 66, "90+": 90}                # born 1931 -> 94, reported as 90


def test_profile_contents(built):
    p = [r.asDict(recursive=True) for r in built["prof"].collect() if r.age == 66][0]
    assert [c["description"] for c in p["active_conditions"]] == ["Essential hypertension (disorder)"]
    assert [i["description"] for i in p["injuries"]] == ["Fracture of forearm (disorder)"]  # not the dental filling
    assert [m["description"] for m in p["active_medications"]] == ["Simvastatin 20 MG Oral Tablet"]
    assert p["latest_fall_risk"]["risk_level"] == "High Risk (MFS Score 50+)"
    assert p["latest_fall_risk"]["score"] == 55
    assert len(p["fall_risk_history"]) == 2


def test_rendered_profile_is_readable_and_leak_free(silver, built):
    p = [r.asDict(recursive=True) for r in built["prof"].collect() if r.age == 66][0]
    text = render(p, AS_OF)
    for heading in ["## Fall risk and falls", "## Injuries", "## Active conditions", "## Active medications",
                    "## Allergies", "## Recent encounters"]:
        assert heading in text
    assert NO_FALLS_NOTE in text
    assert "High Risk (MFS Score 50+), total score 55, assessed 2026-03-01" in text

    patient_rows = [r.asDict() for r in silver["patients"].collect()]
    keymap = {r.patient: r.resident_id for r in built["keymap"].collect()}
    known = KnownIdentifiers.from_patient_rows(patient_rows, keymap)
    assert find_leaks(text, known, resident_id=p["resident_id"]) == []


def test_spark_scan_clean_gold_and_catches_planted_leaks(spark, silver, built, make_df):
    from rcqa import privacy_spark

    patient_rows = [r.asDict() for r in silver["patients"].collect()]
    keymap = {r.patient: r.resident_id for r in built["keymap"].collect()}
    known = KnownIdentifiers.from_patient_rows(patient_rows, keymap)
    addresses = {keymap[r["id"]]: r["address"] for r in patient_rows if r["id"] in keymap}

    for name in ["res", "cond", "meds", "enc", "falls", "prof"]:
        texts = privacy_spark.row_text(built[name])
        assert privacy_spark.scan(spark, texts, known, addresses).count() == 0, name

    rid = keymap["p-alive"]
    planted = make_df([(str(rid), "Tennille443 999-21-9587 born 1960-03-01 at 911 Hauck Key Suite 34")],
                      ["resident_id", "text"]).selectExpr("cast(resident_id as int) resident_id", "text")
    kinds = {r.kind for r in privacy_spark.scan(spark, planted, known, addresses).collect()}
    assert {"synthea_name", "ssn", "known_identifier", "birthdate", "address"} <= kinds
