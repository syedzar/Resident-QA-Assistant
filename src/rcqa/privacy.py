"""Privacy scanner: find direct identifiers in text that should be de-identified.

Two independent nets, so one missing rule does not let a leak through:
  1. Format patterns: anything shaped like an SSN, UUID, Synthea driver's
     licence/passport number, phone number, or a Synthea-style name ("Tennille443").
  2. Known values: every name, street address and record ID from the restricted
     silver layer, plus each resident's OWN exact birth date (matching every
     resident's birth date against every row would flag ordinary encounter dates).
Used by the gold notebook (blocks the pipeline), by the stage 5 chunk check, and by pytest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

PATTERNS: dict[str, re.Pattern] = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "uuid": re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    "drivers_licence": re.compile(r"\bS\d{8}\b"),
    "passport": re.compile(r"\bX\d{8}X\b"),
    "phone": re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "synthea_name": re.compile(r"\b[A-Z][a-z]+\d{2,3}\b"),   # Synthea appends digits to every name
}
_TOKEN = re.compile(r"[A-Za-z0-9'-]+")


@dataclass(frozen=True)
class Leak:
    kind: str
    value: str


@dataclass
class KnownIdentifiers:
    """Identifier values taken from silver_patients (restricted)."""
    tokens: set[str] = field(default_factory=set)        # single-word: names, IDs
    phrases: set[str] = field(default_factory=set)       # multi-word: street addresses
    birthdates: dict[int, date] = field(default_factory=dict)  # resident_id -> own birth date

    @classmethod
    def from_patient_rows(cls, rows, keymap: dict[str, int]) -> "KnownIdentifiers":
        """rows: dicts with silver_patients columns; keymap: patient UUID -> resident_id."""
        known = cls()
        for r in rows:
            for col in ("first", "middle", "last", "maiden", "ssn", "drivers", "passport", "id"):
                if r.get(col):
                    known.tokens.update(t for t in str(r[col]).split() if len(t) >= 3)
            if r.get("address"):
                known.phrases.add(str(r["address"]).strip())
            if r.get("id") in keymap and r.get("birthdate"):
                known.birthdates[keymap[r["id"]]] = r["birthdate"]
        return known


def _date_forms(d: date) -> list[str]:
    return [d.isoformat(), d.strftime("%m/%d/%Y"), f"{d:%B} {d.day}, {d.year}", f"{d.day} {d:%B} {d.year}"]


def find_leaks(text: str, known: KnownIdentifiers | None = None, resident_id: int | None = None) -> list[Leak]:
    leaks = [Leak(kind, m.group()) for kind, rx in PATTERNS.items() for m in rx.finditer(text)]
    if known:
        leaks += [Leak("known_identifier", t) for t in set(_TOKEN.findall(text)) & known.tokens]
        leaks += [Leak("address", p) for p in known.phrases if p in text]
        bd = known.birthdates.get(resident_id) if resident_id is not None else None
        if bd:
            leaks += [Leak("birthdate", f) for f in _date_forms(bd) if f in text]
    return leaks
