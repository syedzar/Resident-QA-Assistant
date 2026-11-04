"""Data-quality checks shared by every pipeline layer.

Two halves, on purpose:
  * ``profile`` runs on Spark and *measures* a table (row count, nulls,
    duplicate keys, orphan foreign keys). It never decides pass/fail.
  * ``evaluate`` is plain Python and *judges* those measurements against
    rules. Keeping the rules out of Spark makes them trivial to unit-test.

Results are rows of a ``dq_results`` Delta table, so the history of every
run is queryable, and ``raise_on_failure`` stops the job when a critical
check fails instead of letting bad data flow downstream.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


@dataclass(frozen=True)
class TableRules:
    table: str
    key: list[str] = field(default_factory=list)          # must be unique and not null
    not_null: list[str] = field(default_factory=list)     # must have zero nulls
    min_rows: int = 1
    # child column -> (parent DataFrame name, parent column); every value must exist in the parent
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    exact_duplicates: bool = False   # count fully identical rows (warning: silver removes them)


@dataclass(frozen=True)
class CheckResult:
    layer: str
    table: str
    check: str
    column: str
    value: int
    threshold: int
    passed: bool
    severity: str  # "error" stops the pipeline, "warn" is recorded only


def profile(df: DataFrame, rules: TableRules, parents: dict[str, DataFrame] | None = None) -> dict:
    """Measure ``df`` in as few Spark passes as possible."""
    cols = sorted(set(rules.key) | set(rules.not_null))
    agg = [F.count(F.lit(1)).alias("__rows")]
    agg += [F.sum(F.col(c).isNull().cast("int")).alias(f"null:{c}") for c in cols]
    if rules.key:
        agg.append(F.countDistinct(*[F.col(c) for c in rules.key]).alias("__distinct_keys"))
    row = df.agg(*agg).first().asDict()

    metrics = {"rows": row["__rows"], "nulls": {c: row[f"null:{c}"] or 0 for c in cols}}
    if rules.key:
        # countDistinct ignores rows with a null key, so nulls are not counted as duplicates here
        non_null_keys = df.dropna(subset=rules.key).count() if any(metrics["nulls"][c] for c in rules.key) else row["__rows"]
        metrics["duplicate_keys"] = non_null_keys - row["__distinct_keys"]
    if rules.exact_duplicates:
        data_cols = [c for c in df.columns if not c.startswith("_")]  # ignore ingest metadata
        metrics["exact_duplicates"] = row["__rows"] - df.dropDuplicates(data_cols).count()
    metrics["orphans"] = {}
    for child_col, (parent_name, parent_col) in rules.foreign_keys.items():
        parent = (parents or {})[parent_name].select(F.col(parent_col).alias("__k")).distinct()
        metrics["orphans"][child_col] = (
            df.select(F.col(child_col).alias("__k")).dropna().join(parent, "__k", "left_anti").count()
        )
    return metrics


def evaluate(layer: str, rules: TableRules, metrics: dict, null_severity: str = "error") -> list[CheckResult]:
    """Turn measurements into pass/fail results. Pure Python."""
    r = [CheckResult(layer, rules.table, "min_rows", "*", metrics["rows"], rules.min_rows,
                     metrics["rows"] >= rules.min_rows, "error")]
    for col, n in metrics["nulls"].items():
        severity = "error" if col in rules.key else null_severity
        r.append(CheckResult(layer, rules.table, "null_count", col, n, 0, n == 0, severity))
    if "duplicate_keys" in metrics:
        n = metrics["duplicate_keys"]
        r.append(CheckResult(layer, rules.table, "duplicate_keys", ",".join(rules.key), n, 0, n == 0, "error"))
    if "exact_duplicates" in metrics:
        n = metrics["exact_duplicates"]
        r.append(CheckResult(layer, rules.table, "exact_duplicate_rows", "*", n, 0, n == 0, "warn"))
    for col, n in metrics["orphans"].items():
        r.append(CheckResult(layer, rules.table, "orphan_foreign_key", col, n, 0, n == 0, "error"))
    return r


def to_rows(results: list[CheckResult]) -> list[dict]:
    return [asdict(x) for x in results]


def raise_on_failure(results: list[CheckResult]) -> None:
    failed = [x for x in results if not x.passed and x.severity == "error"]
    if failed:
        lines = "\n".join(f"  {x.table}.{x.column}: {x.check}={x.value} (allowed {x.threshold})" for x in failed)
        raise ValueError(f"{len(failed)} data-quality check(s) failed:\n{lines}")
