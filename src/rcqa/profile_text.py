"""Render one gold resident profile as plain text with stable '## ' section headings.

Stage 5 chunks this text by section, so every chunk is citable as
"Resident 204 > Active medications". Pure Python: unit-tested without Spark.
Input is ``Row.asDict(recursive=True)`` from gold_resident_profiles.
"""
from __future__ import annotations

from datetime import date
from typing import Any

NO_FALLS_NOTE = ("No fall events are recorded for this resident. The source record contains "
                 "fall-risk assessments (Morse Fall Scale) and injury diagnoses, not fall incidents.")


def _d(value: Any) -> str:
    if value is None:
        return "unknown date"
    return value.isoformat() if isinstance(value, date) else str(value)[:10]


def _lines(items: list[dict] | None, fmt, empty: str) -> list[str]:
    return [f"- {fmt(i)}" for i in items] if items else [empty]


def render(p: dict, as_of: str) -> str:
    rid = p["resident_id"]
    out = [f"# Resident {rid}",
           f"Sex: {p.get('sex') or 'unknown'} | Age: {p.get('age_band') if p.get('age_band') == '90+' else p.get('age')}"
           f" (age band {p.get('age_band')}) | Profile as of {as_of}", ""]

    out.append("## Fall risk and falls")
    latest = p.get("latest_fall_risk")
    if latest:
        score = latest.get("score")
        out.append(f"Latest Morse Fall Scale assessment: {latest.get('risk_level') or 'level not recorded'}"
                   + (f", total score {score:g}" if score is not None else "")
                   + f", assessed {_d(latest.get('assessed_date'))}.")
        hist = p.get("fall_risk_history") or []
        if len(hist) > 1:
            out.append("Previous assessments: " + "; ".join(
                f"{_d(h.get('assessed_date'))} {h.get('risk_level') or '?'}" for h in hist[1:]) + ".")
    else:
        out.append("No Morse Fall Scale assessment is recorded.")
    out += [NO_FALLS_NOTE, ""]

    out.append("## Injuries")
    out += _lines(p.get("injuries"),
                  lambda i: f"{_d(i.get('onset_date'))}: {i.get('description')}"
                            + (f" (resolved {_d(i.get('resolved_date'))})" if i.get("resolved_date") else " (active)"),
                  "No injury diagnoses recorded.")
    out.append("")

    out.append("## Active conditions")
    out += _lines(p.get("active_conditions"),
                  lambda c: f"{c.get('description')} (since {_d(c.get('onset_date'))})",
                  "No active disorders recorded.")
    out.append("")

    out.append("## Active medications")
    out += _lines(p.get("active_medications"),
                  lambda m: f"{m.get('description')} (since {_d(m.get('start_date'))}"
                            + (f"; for {m.get('reason')})" if m.get("reason") else ")"),
                  "No active medications recorded.")
    out.append("")

    out.append("## Allergies")
    out += _lines(p.get("allergies"),
                  lambda a: f"{a.get('description')}"
                            + (f"; reaction: {a.get('reaction')}" if a.get("reaction") else "")
                            + (f" ({a.get('severity').lower()})" if a.get("severity") else ""),
                  "No active allergies recorded.")
    out.append("")

    n = p.get("n_encounters") or 0
    out.append(f"## Recent encounters (latest {len(p.get('recent_encounters') or [])} of {n})")
    out += _lines(p.get("recent_encounters"),
                  lambda e: f"{_d(e.get('encounter_date'))} {e.get('encounter_class')}: {e.get('description')}"
                            + (f"; reason: {e.get('reason')}" if e.get("reason") else ""),
                  "No encounters recorded.")
    return "\n".join(out).strip() + "\n"
