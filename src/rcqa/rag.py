"""Retrieval-augmented answers with citations and an "I don't know" fallback.

Three independent guards, so a weak answer is refused instead of guessed:
  1. Retrieval gate: if even the best chunk scores below ``min_score``, we don't call the LLM.
  2. Prompt: the LLM must answer only from the numbered sources, cite [n] after each fact,
     and reply exactly "I don't know" otherwise.
  3. Output check: an answer that cites no valid source is replaced with the fallback.
     Citation numbers that don't exist are dropped and counted.

Resident questions retrieve from two pools: that resident's own chunks (filtered by
``resident_id``) and public references (laws, guidelines, labels), so a question like
"is she on a blood thinner, and what should we watch for?" can cite both.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

IDK = "I don't know. The available sources don't cover this."
MIN_SCORE = 0.65          # measured: on-topic top hits >= 0.72, off-topic <= 0.61 (README); tuned in stage 8

# Sources first, rules and question last: small models follow the instructions nearest the answer.
PROMPT = """You are a reference assistant for staff in an Ontario long-term care home.

Sources:
{sources}
{resident_note}
Rules:
- Answer the question using ONLY the sources above. Do not use outside knowledge or add advice.
- Write full sentences in your own words from the sources. END each sentence with the label of the
  source that supports it, e.g. "... [S2]." Labels look like [S1], [S2]; numbers inside a source's
  text (section or paragraph numbers) are NOT labels.
- Never number your own sentences or bullets; if you use a list, start items with "-".
- If the sources do not answer the question, reply with exactly: I don't know
- At most 5 sentences.

Question: {question}
Answer:"""


@dataclass
class Source:
    n: int
    citation: str
    citation_url: str | None
    document_type: str
    resident_id: int | None
    score: float
    text: str


@dataclass
class Answer:
    question: str
    text: str
    supported: bool
    reason: str                     # answered | low_retrieval_score | llm_idk | no_citations | no_content | llm_error
    sources: list[Source] = field(default_factory=list)   # everything retrieved, in prompt order
    cited: list[Source] = field(default_factory=list)     # the sources the answer actually cites
    raw: str = ""                   # unedited LLM output, for debugging and evaluation
    invalid_citations: int = 0
    uncited_sentences: int = 0
    top_score: float = 0.0
    recited: int = 0                # citations corrected by verify_citations (model cited the wrong number)


class LLM(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class OllamaLLM:
    """Local model through Ollama's REST API. Temperature 0 so eval runs are repeatable."""

    def __init__(self, model: str | None = None, url: str | None = None, timeout: int = 120):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.url = (url or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.name = f"ollama/{self.model}"

    def generate(self, prompt: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0, "num_ctx": 8192}}).encode()
        req = urllib.request.Request(f"{self.url}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["response"]


# Retriever: (question, k, resident_id, document_types) -> list of hit dicts (payload + score)
Retriever = Callable[[str, int, "int | None", "list[str] | None"], list[dict]]


def qdrant_retriever(client, collection: str, embedder) -> Retriever:
    from rcqa import index

    def retrieve(question, k, resident_id=None, document_types=None):
        return index.search(client, collection, embedder.embed_query(question), k, resident_id, document_types)
    return retrieve


def retrieve(retriever: Retriever, question: str, resident_id: int | None = None,
             document_types: list[str] | None = None, k: int = 5, k_resident: int = 8) -> list[dict]:
    """Public question: top-k public chunks.
    Resident question: that resident's WHOLE record (about 6 chunks, so ranking it can only lose a
    section), plus top-k public chunks only if ``document_types`` asks for references."""
    hits = []
    if resident_id is not None:
        hits += retriever(question, k_resident, resident_id, None)          # resident's own chunks only
    if resident_id is None or document_types:
        hits += retriever(question, k, None, document_types)                 # public references only
    seen, out = set(), []
    for h in hits:
        if h["chunk_id"] not in seen:
            seen.add(h["chunk_id"])
            out.append(h)
    return out


def to_sources(hits: list[dict]) -> list[Source]:
    return [Source(i, h["citation"], h.get("citation_url"), h["document_type"], h.get("resident_id"),
                   float(h["score"]), h["text"]) for i, h in enumerate(hits, 1)]


def build_prompt(question: str, sources: list[Source], resident_id: int | None = None) -> str:
    note = (f"\nThe question is about Resident {resident_id}. Use only that resident's record for facts about them.\n"
            if resident_id is not None else "")
    blocks = "\n\n".join(f"[S{s.n}] {s.citation}\n{s.text}" for s in sources)
    return PROMPT.format(resident_note=note, sources=blocks, question=question.strip())


# Sources are labelled [S1], [S2]... so they can't be confused with numbers inside the text
# (laws and labels are full of "7. Every resident has the right..."). Answers are shown as [1], [2].
_CITE = re.compile(r"\[S?(\d+(?:\s*,\s*S?\d+)*)\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?!\[)|\n+")   # keep "... done. [3]" with its citation


def parse_citations(text: str) -> list[int]:
    """'... [1] ... [2, 3]' -> [1, 2, 3] (in order of first appearance)."""
    out = []
    for m in _CITE.finditer(text):
        for n in re.findall(r"\d+", m.group(1)):
            if int(n) not in out:
                out.append(int(n))
    return out


def strip_invalid_citations(text: str, valid: set[int]) -> str:
    """'[2, 9]' with valid {1, 2} -> '[2]'; '[9]' -> '' (a citation to a source that wasn't given)."""
    def keep(m):
        nums = [n for n in re.findall(r"\d+", m.group(1)) if int(n) in valid]
        return f"[{', '.join(nums)}]" if nums else ""
    return re.sub(r" +([.,;])", r"\1", _CITE.sub(keep, text))   # no space left before punctuation


_WORD = re.compile(r"[a-z0-9]{4,}")
_CLAIM = re.compile(r"(?P<body>[^\[\]]*?)(?P<cite>\[S?\d+(?:\s*,\s*S?\d+)*\])")


def _overlap(claim: str, source: str) -> float:
    """Share of the claim's content words (4+ letters) that appear in the source text."""
    words = set(_WORD.findall(claim.lower()))
    return len(words & set(_WORD.findall(source.lower()))) / len(words) if words else 0.0


def verify_citations(text: str, sources: list[Source], margin: float = 0.25, strong: float = 0.6) -> tuple[str, int]:
    """Check each claim against the source it cites, without trusting the model's numbering.
    Re-cite a claim when another source contains clearly more of its words (best >= strong and
    best - cited >= margin). Legal text shares boilerplate ("licensee", "shall ensure"), so an
    absolute threshold alone isn't enough. Returns (text, number of citations corrected)."""
    by_n = {s.n: s.text for s in sources}
    fixed = 0

    def check(m):
        nonlocal fixed
        body, nums = m.group("body"), [int(n) for n in re.findall(r"\d+", m.group("cite"))]
        if len(_WORD.findall(body.lower())) < 3:
            return m.group(0)
        cited = max((_overlap(body, by_n.get(n, "")) for n in nums), default=0)
        best_n, best = max(((n, _overlap(body, t)) for n, t in by_n.items()), key=lambda x: x[1])
        if best_n not in nums and best >= strong and best - cited >= margin:
            fixed += 1
            return f"{body}[{best_n}]"
        return m.group(0)
    return _CLAIM.sub(check, text), fixed


def tidy(text: str) -> str:
    """Clean display text (input already has plain [n] from strip_invalid_citations): drop citations
    used as line labels ("[2] Every licensee..."), literal "[n]" placeholders, and a citation that
    repeats the one right before it ("... [3]. [3]")."""
    text = re.sub(r"(?m)^(\s*[-*]?\s*)\[[\d, ]+\]\s+(?=\w)", r"\1", text)
    text = text.replace(" [n]", "").replace("[n]", "")
    text = re.sub(r"(\[[\d, ]+\])(\.?)\s*\1", r"\1\2", text)
    return text.strip()


def uncited_sentences(text: str) -> int:
    """Sentences/bullets that state something but carry no [n]. Used as a support metric in stage 8."""
    parts = [p.strip(" -*•\t") for p in _SENTENCE.split(text)]
    return sum(1 for p in parts if len(p.split()) >= 3 and not _CITE.search(p))


def _is_idk(text: str) -> bool:
    return text.strip().lower().strip(" .\"'").startswith("i don't know") or text.strip().lower().startswith("i do not know")


def answer(question: str, retriever: Retriever, llm: LLM, resident_id: int | None = None,
           document_types: list[str] | None = None, k: int = 5, k_resident: int = 8,
           min_score: float = MIN_SCORE) -> Answer:
    sources = to_sources(retrieve(retriever, question, resident_id, document_types, k, k_resident))
    top = max((s.score for s in sources), default=0.0)
    if top < min_score:
        return Answer(question, IDK, False, "low_retrieval_score", sources, top_score=top)

    try:
        raw = llm.generate(build_prompt(question, sources, resident_id)).strip()
    except Exception as e:   # model down / timeout: refuse, never invent
        return Answer(question, IDK, False, "llm_error", sources, raw=repr(e), top_score=top)
    if _is_idk(raw):
        return Answer(question, IDK, False, "llm_idk", sources, raw=raw, top_score=top)

    nums = parse_citations(raw)
    by_n = {s.n: s for s in sources}
    cited = [by_n[n] for n in nums if n in by_n]
    invalid = sum(1 for n in nums if n not in by_n)
    if not cited:
        return Answer(question, IDK, False, "no_citations", sources, raw=raw, invalid_citations=invalid, top_score=top)
    text = tidy(strip_invalid_citations(raw, set(by_n)))
    if len(_CITE.sub("", text).split()) < 3:        # e.g. the model replied just "[2]"
        return Answer(question, IDK, False, "no_content", sources, raw=raw, top_score=top)
    text, recited = verify_citations(text, sources)
    cited = [by_n[n] for n in parse_citations(text)]
    return Answer(question, text.strip(), True, "answered", sources, cited, raw, invalid,
                  uncited_sentences(text), top, recited)


SUMMARY_PROMPT = """You are preparing a short shift-handover summary for staff in an Ontario long-term care home.

Resident {rid}'s record, in numbered sections:
{sources}

Rules:
- Use ONLY the record above. No advice beyond it.
- Write at most 8 bullet points. Start each bullet with "-" (never with a number).
- END each bullet with the label of the section it came from, like this: "- Low fall risk, score 15 [S1]"
- Cover fall risk, injuries, active conditions, active medications, allergies and recent encounters.
  If a section says nothing is recorded, say so briefly.

Summary:"""


def summarize_resident(resident_id: int, record_chunks: list[dict], llm: LLM) -> Answer:
    """record_chunks: every chunk of this resident (see index.resident_chunks)."""
    q = f"Summarize resident {resident_id}"
    sources = to_sources([{**h, "score": 1.0} for h in record_chunks])
    if not sources:
        return Answer(q, IDK, False, "low_retrieval_score")
    blocks = "\n\n".join(f"[S{s.n}] {s.citation}\n{s.text}" for s in sources)
    try:
        raw = llm.generate(SUMMARY_PROMPT.format(rid=resident_id, sources=blocks)).strip()
    except Exception as e:
        return Answer(q, IDK, False, "llm_error", sources, raw=repr(e))
    by_n = {s.n: s for s in sources}
    cited = [by_n[n] for n in parse_citations(raw) if n in by_n]
    if not cited:
        return Answer(q, IDK, False, "no_citations", sources, raw=raw)
    text = tidy(strip_invalid_citations(raw, set(by_n)))
    return Answer(q, text, True, "answered", sources, cited, raw, uncited_sentences=uncited_sentences(text), top_score=1.0)
