"""Resident Care Q&A Assistant: Streamlit front end.

    streamlit run app/streamlit_app.py

Two modes:
  * Policies & guidelines: searches only public documents (laws, RNAO, PHO, drug labels).
  * A resident: searches only that resident's de-identified record, optionally plus references.
Every answer shows clickable sources; unsupported questions get "I don't know".
Synthetic data only. Not clinical advice.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient

from rcqa import index, rag
from rcqa.embed import Embedder

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Resident Care Q&A", page_icon="🩺", layout="wide")

TYPE_LABELS = {"regulation": "O. Reg. 246/22", "legislation": "Fixing Long-Term Care Act", "guideline": "RNAO guidelines",
               "ipac_guidance": "PHO infection control", "drug_label": "Drug labels"}
REASONS = {"low_retrieval_score": "no source was close enough to the question",
           "llm_idk": "the sources found don't contain the answer",
           "no_citations": "the draft answer cited no source, so it was withheld",
           "no_content": "the draft answer had no content", "llm_error": "the language model is not reachable"}


@st.cache_resource
def services():
    client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                          api_key=os.getenv("QDRANT_API_KEY") or None)
    return client, Embedder(), rag.OllamaLLM()


def show_answer(a: rag.Answer, seconds: float) -> None:
    if a.supported:
        st.markdown(a.text)
    else:
        st.warning(f"**I don't know.** The available sources don't cover this ({REASONS.get(a.reason, a.reason)}).")
    st.caption(f"{seconds:.1f}s · best source score {a.top_score:.2f}"
               + (f" · {a.recited} citation(s) corrected" if a.recited else ""))
    if a.cited:
        st.subheader("Sources")
        for s in a.cited:
            with st.expander(f"[{s.n}] {s.citation}"):
                if s.citation_url:
                    st.markdown(f"[Open the original source ↗]({s.citation_url})")
                st.text(s.text[:1500])
    others = [s for s in a.sources if s not in a.cited]
    if others:
        with st.expander(f"Also retrieved but not cited ({len(others)})"):
            for s in others:
                link = f" · [source ↗]({s.citation_url})" if s.citation_url else ""
                st.markdown(f"- [{s.n}] {s.citation} (score {s.score:.2f}){link}")


def main() -> None:
    client, embedder, llm = services()
    st.title("Resident Care Q&A Assistant")
    st.caption("Answers come only from cited sources: Ontario long-term care law, RNAO and PHO guidance, "
               "drug labels, and **synthetic, de-identified** resident records. Not clinical advice.")

    with st.sidebar:
        st.header("Ask about")
        mode = st.radio("Mode", ["Policies & guidelines", "A resident"], label_visibility="collapsed")
        resident_id = None
        if mode == "A resident":
            resident_id = st.number_input("Resident ID (101–1100)", min_value=101, max_value=1100, value=204, step=1)
            types = st.multiselect("Also search references", list(TYPE_LABELS), default=[],
                                   format_func=TYPE_LABELS.get, help="e.g. add Drug labels for medication safety")
        else:
            types = st.multiselect("Sources", list(TYPE_LABELS), default=list(TYPE_LABELS), format_func=TYPE_LABELS.get)
        st.divider()
        st.caption(f"Model: {llm.name} · Index: {index.ALIAS}")

    examples = (["What is this resident's fall risk?", "What medications is this resident taking?",
                 "Does this resident have any allergies?"] if resident_id else
                ["What does the regulation require after a resident falls?",
                 "How often should people at risk of pressure injuries be repositioned?",
                 "What are the signs of bleeding to watch for with warfarin?"])
    st.write("Try: " + " · ".join(f"*{e}*" for e in examples))

    with st.form("ask"):
        q = st.text_input("Question", placeholder=examples[0])
        cols = st.columns([1, 2, 3])
        ask = cols[0].form_submit_button("Ask", type="primary")
        summarize = cols[1].form_submit_button("Summarize this resident", disabled=resident_id is None)

    if ask and q.strip():
        with st.spinner("Searching sources and drafting an answer…"):
            t = time.time()
            retriever = rag.qdrant_retriever(client, index.ALIAS, embedder)
            a = rag.answer(q, retriever, llm, int(resident_id) if resident_id else None, types or None)
        show_answer(a, time.time() - t)
    elif summarize and resident_id:
        with st.spinner(f"Summarizing resident {resident_id}…"):
            t = time.time()
            a = rag.summarize_resident(int(resident_id), index.resident_chunks(client, index.ALIAS, int(resident_id)), llm)
        st.subheader(f"Resident {resident_id}: handover summary")
        show_answer(a, time.time() - t)


main()
