from pyspark.sql import functions as F

from rcqa.silver import clean, snake


def test_snake_case():
    assert snake("BASE_ENCOUNTER_COST") == "base_encounter_cost"
    assert snake("ReasonCode") == "reason_code"
    assert snake(" Id ") == "id"


def test_clean_types_blanks_and_duplicates(make_df):
    bronze = make_df(
        [("2015-06-22T02:23:04Z", "", "p1", "849574", "Naproxen  sodium 220 MG", "138", "f1.csv"),
         ("2015-06-22T02:23:04Z", "", "p1", "849574", "Naproxen  sodium 220 MG", "138", "f2.csv"),  # dup
         ("not-a-date", "2020-01-01T00:00:00Z", "p2", "1", " Aspirin ", "x", "f1.csv")],
        ["START", "STOP", "PATIENT", "CODE", "DESCRIPTION", "DISPENSES", "_source_file"])
    silver = clean(bronze, "medications")
    assert dict(silver.dtypes)["start"] == "timestamp"
    # Compare as a UTC string: collect() would convert to the laptop's local time zone.
    silver = silver.withColumn("start_utc", F.date_format("start", "yyyy-MM-dd HH:mm:ss"))
    rows = sorted(silver.collect(), key=lambda r: r.patient)

    assert len(rows) == 2                                     # exact duplicate dropped despite different _source_file
    r1, r2 = rows
    assert r1.start_utc == "2015-06-22 02:23:04"              # 'Z' parsed as UTC
    assert r1.stop is None                                    # "" -> NULL
    assert r1.description == "Naproxen sodium 220 MG"         # double space collapsed
    assert r1.dispenses == 138
    assert r2.start is None and r2.dispenses is None          # bad values become NULL for DQ to catch
    assert r2.description == "Aspirin"                        # trimmed


def test_observation_value_kept_as_text_and_number(make_df):
    bronze = make_df(
        [("2020-01-01T00:00:00Z", "72.5"), ("2020-01-01T00:00:00Z", "High Risk (MFS Score 50+)")],
        ["DATE", "VALUE"])
    rows = {r.value: r.value_num for r in clean(bronze, "observations").collect()}
    assert rows == {"72.5": 72.5, "High Risk (MFS Score 50+)": None}
