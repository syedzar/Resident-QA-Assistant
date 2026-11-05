import json
from pathlib import Path

from rcqa import evaluation as ev, rag

ROOT = Path(__file__).resolve().parents[1]
RES = {"source_id": "resident_204", "section": "Active medications"}
DOC = {"source_id": "oreg_246_22", "evidence": "post-fall assessment"}


def test_relevance_by_section_and_by_evidence():
    assert ev.is_relevant({"source_id": "resident_204", "section": "Active medications", "text": "x"}, RES)
    assert not ev.is_relevant({"source_id": "resident_305", "section": "Active medications", "text": "x"}, RES)
    assert ev.is_relevant({"source_id": "oreg_246_22", "section": "s. 54", "text": "a Post-fall assessment is"}, DOC)
    assert not ev.is_relevant({"source_id": "oreg_246_22", "section": "s. 55", "text": "skin and wound"}, DOC)
    assert not ev.is_relevant({"source_id": "x"}, None)


def test_retrieval_metrics():
    m = ev.retrieval_metrics([1, 3, None, 6], ks=(1, 5))
    assert m == {"hit_at_1": 0.25, "hit_at_5": 0.5, "mrr": (1 + 1 / 3 + 1 / 6) / 4}


def _answer(supported, cited, text="Takes metformin [1]."):
    return rag.Answer("q", text, supported, "answered" if supported else "llm_idk", cited=cited)


def _src(source_id, section, text=""):
    return rag.Source(1, "c", None, "resident_profile", 204, 0.8, text, source_id, section)


def test_answer_metrics():
    item = {"id": "a", "kind": "resident", "expected": RES, "answer_regex": "metformin"}
    good = ev.score_answer(item, _answer(True, [_src("resident_204", "Active medications")]))
    wrong_source = ev.score_answer({**item, "id": "b"}, _answer(True, [_src("resident_204", "Allergies")]))
    refused = ev.score_answer({**item, "id": "c"}, _answer(False, []))
    neg_ok = ev.score_answer({"id": "n", "kind": "negative", "expected": None, "answer_regex": None},
                             _answer(False, []))
    assert good["grounded"] and good["keyword_ok"] and not wrong_source["grounded"]
    m = ev.answer_metrics([good, wrong_source, refused, neg_ok])
    assert m["answer_rate"] == 2 / 3 and m["grounded_rate"] == 1 / 3
    assert m["unsupported_answer_pct"] == 50.0 and m["refusal_rate_negatives"] == 1.0


def test_eval_set_shape():
    items = ev.load_questions(ROOT / "data" / "eval" / "questions.jsonl")
    kinds = [q["kind"] for q in items]
    assert kinds.count("resident") == 25 and kinds.count("document") == 25 and kinds.count("negative") == 10
    assert len({q["id"] for q in items}) == len(items)
    for q in items:
        if q["kind"] == "resident":
            assert q["resident_id"] and q["expected"]["source_id"] == f"resident_{q['resident_id']}"
        if q["kind"] == "document":
            assert q["document_types"] and q["expected"]["evidence"]
    json.dumps(items)
