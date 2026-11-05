"""Scoring for the 60-question eval set (pure functions, unit-tested).

Gold labels (see scripts/build_eval_set.py) are independent of chunking:
  resident question -> {"source_id": "resident_204", "section": "Active medications"}
  document question -> {"source_id": "oreg_246_22", "evidence": "post-fall assessment"}
A retrieved or cited chunk is *relevant* if it comes from that source and has that section /
contains that evidence. So hit@k means the same thing for 100-word and 400-word chunks.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

from rcqa import rag


def load_questions(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def is_relevant(chunk: dict, expected: dict | None) -> bool:
    """chunk: dict with source_id, section, text (a hit payload or a Source's fields)."""
    if not expected or chunk.get("source_id") != expected["source_id"]:
        return False
    if "section" in expected:
        return (chunk.get("section") or "") == expected["section"]
    return re.search(expected["evidence"], chunk.get("text") or "", re.I) is not None


def ranked_hits(retriever, item: dict, k: int, use_filter: bool = True) -> list[dict]:
    """What the answer step would see, ranked by score. Resident questions search the resident's
    record; document questions search public chunks, limited to the question's document type when
    ``use_filter`` (the app's category selector) and across all public types otherwise."""
    if item["resident_id"] is not None:
        hits = retriever(item["question"], k, item["resident_id"], None)
    else:
        hits = retriever(item["question"], k, None, item["document_types"] if use_filter else None)
    return sorted(hits, key=lambda h: -h["score"])


def first_relevant_rank(hits: list[dict], expected: dict | None) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if is_relevant(h, expected)), None)


def retrieval_metrics(ranks: list[int | None], ks=(1, 3, 5, 10)) -> dict[str, float]:
    """ranks: 1-based rank of the first relevant chunk per question (None = not retrieved)."""
    n = len(ranks)
    out = {f"hit_at_{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    return out


def score_answer(item: dict, a: rag.Answer) -> dict:
    """Per-question answer record. 'grounded' = answered AND at least one cited source is a gold source."""
    cited_ok = any(is_relevant({"source_id": s.source_id, "section": s.section, "text": s.text}, item["expected"])
                   for s in a.cited)
    kw = item.get("answer_regex")
    return {"id": item["id"], "kind": item["kind"], "answered": a.supported, "reason": a.reason,
            "grounded": a.supported and cited_ok,
            "keyword_ok": bool(a.supported and kw and re.search(kw, a.text, re.I)),
            "uncited_sentences": a.uncited_sentences, "recited": a.recited,
            "invalid_citations": a.invalid_citations, "top_score": round(a.top_score, 4),
            "answer": a.text, "cited": [s.citation for s in a.cited]}


def answer_metrics(rows: list[dict]) -> dict[str, float]:
    pos = [r for r in rows if r["kind"] != "negative"]
    neg = [r for r in rows if r["kind"] == "negative"]
    answered = [r for r in pos if r["answered"]]
    out = {
        "answer_rate": len(answered) / len(pos),
        "grounded_rate": sum(r["grounded"] for r in pos) / len(pos),
        # the README's headline: of the answers we GAVE, how many cite no correct source
        "unsupported_answer_pct": 100 * sum(not r["grounded"] for r in answered) / len(answered) if answered else 0.0,
        "keyword_accuracy": sum(r["keyword_ok"] for r in pos) / len(pos),
        "uncited_sentences_per_answer": mean([r["uncited_sentences"] for r in answered]) if answered else 0.0,
        "citations_corrected": float(sum(r["recited"] for r in rows)),
    }
    if neg:
        out["refusal_rate_negatives"] = sum(not r["answered"] for r in neg) / len(neg)
    for kind in ("resident", "document"):
        sub = [r for r in pos if r["kind"] == kind]
        if sub:
            out[f"grounded_rate_{kind}"] = sum(r["grounded"] for r in sub) / len(sub)
    return out
