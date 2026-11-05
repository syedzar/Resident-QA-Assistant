"""RAG guards, with a fake retriever and a fake LLM (no Ollama, no Qdrant needed)."""
from rcqa import rag


def hit(cid, score, dtype="regulation", resident_id=None, citation=None):
    return {"chunk_id": cid, "score": score, "document_type": dtype, "resident_id": resident_id,
            "citation": citation or f"cite {cid}", "citation_url": "https://x", "text": f"text of {cid}"}


class FakeRetriever:
    def __init__(self, resident_hits=(), public_hits=()):
        self.resident_hits, self.public_hits, self.calls = list(resident_hits), list(public_hits), []

    def __call__(self, question, k, resident_id=None, document_types=None):
        self.calls.append((k, resident_id, document_types))
        return (self.resident_hits if resident_id is not None else self.public_hits)[:k]


class FakeLLM:
    name = "fake"

    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


PUBLIC = [hit("s54", 0.80, citation="O. Reg. 246/22, s. 54 (Falls prevention and management)"), hit("s55", 0.70)]


def test_answer_with_valid_citations():
    llm = FakeLLM("A post-fall assessment must be conducted [1]. Use a clinically appropriate tool [1, 2].")
    a = rag.answer("post-fall?", FakeRetriever(public_hits=PUBLIC), llm)
    assert a.supported and a.reason == "answered"
    assert [s.citation for s in a.cited] == ["O. Reg. 246/22, s. 54 (Falls prevention and management)", "cite s55"]
    assert a.uncited_sentences == 0
    assert "[S1] O. Reg. 246/22, s. 54" in llm.prompts[0] and "text of s54" in llm.prompts[0]


def test_low_retrieval_score_refuses_without_calling_llm():
    llm = FakeLLM("should not be used [1]")
    a = rag.answer("capital of France?", FakeRetriever(public_hits=[hit("s1", 0.46)]), llm)
    assert not a.supported and a.reason == "low_retrieval_score" and a.text == rag.IDK
    assert llm.prompts == []


def test_llm_says_i_dont_know():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("I don't know."))
    assert not a.supported and a.reason == "llm_idk" and a.text == rag.IDK


def test_uncited_answer_is_replaced_by_fallback():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("Residents should be checked hourly."))
    assert not a.supported and a.reason == "no_citations" and a.text == rag.IDK
    assert a.raw == "Residents should be checked hourly."      # kept for evaluation


def test_invalid_citation_numbers_are_dropped_and_counted():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("Assess after a fall [1, 7]. Also call family [9]."))
    assert a.supported and a.invalid_citations == 2
    assert a.text == "Assess after a fall [1]. Also call family."
    assert a.uncited_sentences == 1                              # "Also call family." has no valid source


def test_only_invalid_citations_means_unsupported():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("Something [8]."))
    assert not a.supported and a.reason == "no_citations"


def test_llm_error_refuses():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM(TimeoutError("ollama down")))
    assert not a.supported and a.reason == "llm_error" and a.text == rag.IDK


def test_resident_question_uses_two_pools_and_names_the_resident():
    r = FakeRetriever(resident_hits=[hit("r204_meds", 0.70, "resident_profile", 204)],
                      public_hits=[hit("warfarin_box", 0.82, "drug_label")])
    llm = FakeLLM("Resident 204 takes warfarin [1]. Warfarin can cause major bleeding [2].")
    a = rag.answer("is she on a blood thinner?", r, llm, resident_id=204, document_types=["drug_label"])
    assert r.calls == [(8, 204, None), (5, None, ["drug_label"])]   # whole record, then public pool
    assert [s.document_type for s in a.sources] == ["resident_profile", "drug_label"]
    assert "about Resident 204" in llm.prompts[0]
    assert a.supported and len(a.cited) == 2


def test_parse_citations():
    assert rag.parse_citations("a [2] b [1, 3] c [2]") == [2, 1, 3]
    assert rag.parse_citations("no sources") == []


def test_summarize_resident_requires_citations():
    chunks = [hit("r204_fall", 1, "resident_profile", 204, "Resident 204 > Fall risk and falls"),
              hit("r204_meds", 1, "resident_profile", 204, "Resident 204 > Active medications")]
    ok = rag.summarize_resident(204, chunks, FakeLLM("- Low fall risk [1]\n- Takes metformin [2]"))
    assert ok.supported and [s.citation for s in ok.cited] == [c["citation"] for c in chunks]
    bad = rag.summarize_resident(204, chunks, FakeLLM("- Doing well overall"))
    assert not bad.supported and bad.reason == "no_citations"
    assert rag.summarize_resident(999, [], FakeLLM("x [1]")).reason == "low_retrieval_score"


def test_citation_only_reply_is_not_an_answer():
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("[2]"))
    assert not a.supported and a.reason == "no_content"


def test_plain_resident_question_uses_only_the_record():
    r = FakeRetriever(resident_hits=[hit("r204_meds", 0.70, "resident_profile", 204)],
                      public_hits=[hit("rnao_p40", 0.80, "guideline")])
    a = rag.answer("what medications?", r, FakeLLM("Takes metformin daily [1]."), resident_id=204)
    assert r.calls == [(8, 204, None)] and [s.document_type for s in a.sources] == ["resident_profile"]


def test_wrong_citation_number_is_corrected_from_the_text():
    hits = [hit("s54", 0.77), hit("s298", 0.71)]
    hits[0]["text"] = "when a resident has fallen, a post-fall assessment is conducted using a clinically appropriate instrument"
    hits[1]["text"] = "term means the period that starts on the latest of the 90th day"
    llm = FakeLLM("When a resident has fallen, a post-fall assessment is conducted using an appropriate instrument [2].")
    a = rag.answer("q", FakeRetriever(public_hits=hits), llm)
    assert a.supported and a.recited == 1
    assert a.text.endswith("instrument [1].") and [s.citation for s in a.cited] == ["cite s54"]


def test_correct_citation_is_left_alone():
    hits = [hit("s54", 0.77), hit("s298", 0.71)]
    hits[0]["text"] = "post-fall assessment is conducted"
    a = rag.answer("q", FakeRetriever(public_hits=hits), FakeLLM("A post-fall assessment is conducted [1]."))
    assert a.recited == 0 and a.text == "A post-fall assessment is conducted [1]."


def test_source_labels_and_tidy_display():
    llm = FakeLLM("[S1] A post-fall assessment is required [S1]. Use a falls tool [S1, S2].")
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), llm)
    assert "[S1] O. Reg. 246/22, s. 54" in llm.prompts[0]
    assert a.text == "A post-fall assessment is required [1]. Use a falls tool [1, 2]."


def test_paragraph_numbers_inside_sources_are_not_citations():
    # "[7]" here is the Bill of Rights paragraph, not a source: it's invalid, the [S1] claim stands
    a = rag.answer("q", FakeRetriever(public_hits=PUBLIC), FakeLLM("Residents have a right to privacy [S1]. See [7]."))
    assert a.supported and a.invalid_citations == 1 and [s.n for s in a.cited] == [1]


def test_boilerplate_overlap_does_not_hide_a_wrong_citation():
    hits = [hit("s54", 0.77), hit("s298", 0.71)]
    hits[0]["text"] = ("every licensee of a long-term care home shall ensure that when a resident has fallen "
                       "the resident is assessed and a post-fall assessment is conducted")
    hits[1]["text"] = "every licensee of a long-term care home shall ensure the term means the period"
    llm = FakeLLM("Every licensee shall ensure that when a resident has fallen, a post-fall assessment is conducted [S2].")
    a = rag.answer("q", FakeRetriever(public_hits=hits), llm)
    assert a.recited == 1 and a.text.endswith("conducted [1].")


def test_uncited_count_keeps_trailing_citation_with_its_sentence():
    assert rag.uncited_sentences("Takes metformin since 2026-10-18. [6]\nAlso takes B12 daily. [6]") == 0
