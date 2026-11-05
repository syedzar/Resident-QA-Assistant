"""LIVE search tests against the loaded Qdrant collection (skipped if Qdrant or the collection is missing).

These are sanity checks that retrieval + filters work end to end on the real chunks.
The scored evaluation (50 questions, hit@k) is stage 8.
"""
import os

import pytest
from qdrant_client import QdrantClient

from rcqa import index

URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _client():
    try:
        c = QdrantClient(url=URL, api_key=os.getenv("QDRANT_API_KEY") or None, timeout=3)
        return c if c.collection_exists(index.ALIAS) else None
    except Exception:
        return None


CLIENT = _client()
pytestmark = pytest.mark.skipif(CLIENT is None, reason=f"no '{index.ALIAS}' collection at {URL}")


@pytest.fixture(scope="module")
def embed():
    from rcqa.embed import Embedder
    return Embedder().embed_query


def test_collection_has_payload_indexes():
    schema = CLIENT.get_collection(index.ALIAS).payload_schema
    assert {"resident_id", "document_type", "source_id"} <= set(schema)


def test_regulation_question_finds_falls_section(embed):
    hits = index.search(CLIENT, index.ALIAS, embed("what must happen after a resident falls? post-fall assessment"),
                        k=5, document_types=["regulation"])
    assert any("s. 54" in h["citation"] for h in hits), [h["citation"] for h in hits]


def test_resident_filter_only_returns_that_resident(embed):
    hits = index.search(CLIENT, index.ALIAS, embed("what medications is the resident taking?"), k=10, resident_id=204)
    assert hits and {h["resident_id"] for h in hits} == {204}


def test_public_search_never_returns_resident_chunks(embed):
    hits = index.search(CLIENT, index.ALIAS, embed("resident medications and allergies"), k=20)
    assert hits and all(h["document_type"] != "resident_profile" for h in hits)
