import pytest

from rcqa.dq import TableRules, evaluate, profile, raise_on_failure

RULES = TableRules("t", key=["id"], not_null=["name"], foreign_keys={"parent_id": ("parents", "id")})


def test_evaluate_passes_clean_metrics():
    m = {"rows": 3, "nulls": {"id": 0, "name": 0}, "duplicate_keys": 0, "orphans": {"parent_id": 0}}
    results = evaluate("silver", RULES, m)
    assert all(r.passed for r in results)
    raise_on_failure(results)  # does not raise


def test_evaluate_flags_each_problem():
    m = {"rows": 0, "nulls": {"id": 1, "name": 2}, "duplicate_keys": 1, "orphans": {"parent_id": 4}}
    failed = {(r.check, r.column) for r in evaluate("silver", RULES, m) if not r.passed}
    assert failed == {("min_rows", "*"), ("null_count", "id"), ("null_count", "name"),
                      ("duplicate_keys", "id"), ("orphan_foreign_key", "parent_id")}
    with pytest.raises(ValueError, match="5 data-quality check"):
        raise_on_failure(evaluate("silver", RULES, m))


def test_warn_severity_does_not_stop_pipeline():
    m = {"rows": 3, "nulls": {"id": 0, "name": 2}, "duplicate_keys": 0, "orphans": {"parent_id": 0}}
    results = evaluate("bronze", RULES, m, null_severity="warn")
    assert not all(r.passed for r in results)
    raise_on_failure(results)  # only warnings -> no exception


def test_profile_measures_spark_dataframe(make_df):
    parents = make_df([("p1",), ("p2",)], ["id"])
    df = make_df(
        [("a", "x", "p1"), ("a", "y", "p2"), ("b", None, "p9"), (None, "z", None)],
        ["id", "name", "parent_id"])
    m = profile(df, RULES, {"parents": parents})
    assert m["rows"] == 4
    assert m["nulls"] == {"id": 1, "name": 1}
    assert m["duplicate_keys"] == 1      # "a" twice; the NULL id is not a duplicate
    assert m["orphans"] == {"parent_id": 1}  # p9 has no parent; NULL is not an orphan


def test_exact_duplicates_are_a_warning(make_df):
    df = make_df([("a", "x"), ("a", "x"), ("b", "y")], ["id", "name"])
    rules = TableRules("t", exact_duplicates=True)
    m = profile(df, rules)
    assert m["exact_duplicates"] == 1
    [dup] = [r for r in evaluate("bronze", rules, m) if r.check == "exact_duplicate_rows"]
    assert not dup.passed and dup.severity == "warn"
