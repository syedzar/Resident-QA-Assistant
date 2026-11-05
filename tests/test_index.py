"""Qdrant layout + filtered search, using in-memory Qdrant and hand-made vectors (no model, no Docker)."""
import pytest
from qdrant_client import QdrantClient

from rcqa import index

DIM = 4


def row(cid, dtype, vec, resident_id=None, **kw):
    return {"chunk_id": cid, "source_id": kw.get("source_id", f"src_{cid}"), "document_type": dtype,
            "title": "t", "section": "s", "page": None, "resident_id": resident_id, "source_date": "2026-01-01",
            "citation": f"cite {cid}", "citation_url": "https://x", "text": f"text {cid}", "config_id": "w200_o40",
            "vector": vec}


ROWS = [
    row("r204_meds", "resident_profile", [1, 0, 0, 0], resident_id=204),
    row("r305_meds", "resident_profile", [1, 0, 0, 0.01], resident_id=305),   # nearly identical vector!
    row("reg_s54", "regulation", [0.9, 0.1, 0, 0]),
    row("rnao_p37", "guideline", [0, 1, 0, 0]),
    row("warfarin", "drug_label", [0, 0, 1, 0]),
]


@pytest.fixture()
def client():
    c = QdrantClient(":memory:")
    index.ensure_collection(c, "chunks_w200_o40", DIM)
    assert index.upsert(c, "chunks_w200_o40", ROWS, batch=2) == 5
    return c


def ids(hits):
    return [h["chunk_id"] for h in hits]


def test_resident_filter_never_returns_another_resident(client):
    hits = index.search(client, "chunks_w200_o40", [1, 0, 0, 0], k=5, resident_id=204)
    assert ids(hits) == ["r204_meds"]          # 305's chunk is closer than any doc, but filtered out


def test_resident_plus_public_types(client):
    hits = index.search(client, "chunks_w200_o40", [1, 0, 0, 0], k=5, resident_id=204, document_types=["regulation"])
    assert ids(hits) == ["r204_meds", "reg_s54"]


def test_no_resident_means_public_documents_only(client):
    hits = index.search(client, "chunks_w200_o40", [1, 0, 0, 0], k=5)
    assert "r204_meds" not in ids(hits) and "r305_meds" not in ids(hits)
    assert ids(hits)[0] == "reg_s54"
    # asking for resident_profile without a resident must not open up everyone's records
    hits = index.search(client, "chunks_w200_o40", [1, 0, 0, 0], k=5, document_types=["resident_profile", "guideline"])
    assert ids(hits) == ["rnao_p37"]


def test_document_type_filter(client):
    assert ids(index.search(client, "chunks_w200_o40", [1, 0, 0, 0], k=5, document_types=["drug_label"])) == ["warfarin"]


def test_reload_is_idempotent_and_payload_is_citable(client):
    index.upsert(client, "chunks_w200_o40", ROWS)
    assert client.count("chunks_w200_o40").count == 5
    hit = index.search(client, "chunks_w200_o40", [0, 0, 1, 0], k=1)[0]
    assert hit["citation"] == "cite warfarin" and hit["citation_url"] == "https://x" and hit["resident_id"] is None


def test_alias_swap(client):
    index.point_alias(client, "chunks_w200_o40")
    index.ensure_collection(client, "chunks_w100_o20", DIM)
    index.point_alias(client, "chunks_w100_o20")
    aliases = {a.alias_name: a.collection_name for a in client.get_aliases().aliases}
    assert aliases == {index.ALIAS: "chunks_w100_o20"}
