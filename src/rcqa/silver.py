"""Bronze -> silver cleaning for the Synthea tables.

Bronze keeps every value exactly as it arrived (all strings). Silver makes the
data usable without changing its meaning:
  * snake_case column names
  * blank strings -> NULL, whitespace trimmed (Synthea pads some text with double spaces)
  * real types: dates, timestamps, numbers
  * exact duplicate rows removed
Silver still contains direct identifiers (names, SSNs, addresses). It is a
restricted layer; de-identification happens on the way to gold (stage 4).
"""
from __future__ import annotations

import re

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from rcqa.dq import TableRules

# Per-table typing. Anything not listed stays a string.
SPECS: dict[str, dict[str, list[str]]] = {
    "patients":      {"date": ["birthdate", "deathdate"],
                      "double": ["lat", "lon", "healthcare_expenses", "healthcare_coverage", "income"]},
    "encounters":    {"timestamp": ["start", "stop"],
                      "double": ["base_encounter_cost", "total_claim_cost", "payer_coverage"]},
    "conditions":    {"date": ["start", "stop"]},
    "medications":   {"timestamp": ["start", "stop"], "int": ["dispenses"],
                      "double": ["base_cost", "payer_coverage", "totalcost"]},
    "observations":  {"timestamp": ["date"]},
    "procedures":    {"timestamp": ["start", "stop"], "double": ["base_cost"]},
    "allergies":     {"date": ["start", "stop"]},
    "careplans":     {"date": ["start", "stop"]},
    "immunizations": {"timestamp": ["date"], "double": ["base_cost"]},
}

# Data-quality rules applied to each silver table.
_P = ("patients", "id")
_E = ("encounters", "id")
RULES: dict[str, TableRules] = {
    "patients":      TableRules("patients", key=["id"], not_null=["birthdate", "gender"]),
    "encounters":    TableRules("encounters", key=["id"], not_null=["patient", "start", "encounterclass"],
                                foreign_keys={"patient": _P}),
    "conditions":    TableRules("conditions", not_null=["patient", "start", "code", "description"],
                                foreign_keys={"patient": _P, "encounter": _E}),
    "medications":   TableRules("medications", not_null=["patient", "start", "code", "description"],
                                foreign_keys={"patient": _P, "encounter": _E}),
    "observations":  TableRules("observations", not_null=["patient", "date", "code", "description"],
                                foreign_keys={"patient": _P, "encounter": _E}),
    "procedures":    TableRules("procedures", not_null=["patient", "start", "code", "description"],
                                foreign_keys={"patient": _P, "encounter": _E}),
    "allergies":     TableRules("allergies", not_null=["patient", "start", "code", "description"],
                                foreign_keys={"patient": _P}),
    "careplans":     TableRules("careplans", key=["id"], not_null=["patient", "start", "code"],
                                foreign_keys={"patient": _P}),
    "immunizations": TableRules("immunizations", not_null=["patient", "date", "code"],
                                foreign_keys={"patient": _P}),
}

_CASTS = {"date": "date", "timestamp": "timestamp", "double": "double", "int": "int"}


def snake(name: str) -> str:
    """'BASE_ENCOUNTER_COST' -> 'base_encounter_cost', 'ReasonCode' -> 'reason_code'."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


def clean(df: DataFrame, table: str) -> DataFrame:
    """Apply the silver rules to one bronze table. Bronze metadata columns (``_*``) are kept."""
    data_cols = [c for c in df.columns if not c.startswith("_")]
    meta_cols = [c for c in df.columns if c.startswith("_")]

    # 1. names + blanks: trim, collapse repeated spaces, empty -> NULL
    exprs = []
    for c in data_cols:
        v = F.regexp_replace(F.trim(F.col(f"`{c}`")), r"\s{2,}", " ")
        exprs.append(F.when(v == "", None).otherwise(v).alias(snake(c)))
    out = df.select(*exprs, *meta_cols)

    # 2. types. try_cast returns NULL instead of failing, and the DQ null checks then catch bad values.
    for kind, cols in SPECS.get(table, {}).items():
        for c in (c for c in cols if c in out.columns):  # tolerate optional columns
            out = out.withColumn(c, F.expr(f"try_cast({c} AS {_CASTS[kind]})"))

    # observations.value mixes numbers and text ("Low Risk (MFS Score 0 - 24)"); keep both forms
    if table == "observations":
        out = out.withColumn("value_num", F.expr("try_cast(value AS double)"))

    # 3. exact duplicates (ignoring ingest metadata, which differs per load)
    return out.dropDuplicates([snake(c) for c in data_cols])
