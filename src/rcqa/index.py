"""Qdrant collection layout, loading and filtered search.

* One collection per chunking config (``chunks_w200_o40``), so stage 8 can compare
  configs side by side; the alias ``rcqa_chunks`` points at the one the app uses.
* Payload indexes on ``resident_id`` (integer) and ``document_type`` / ``source_id``
  (keyword) make filtered search fast and exact: a question about resident 204 can
  only ever see resident 204's chunks plus whichever public document types we allow.
* Point ids are UUIDv5 of ``chunk_id``, so re-loading is idempotent (upsert, no dupes).
"""
from __future__ import annotations

import uuid
from typing import Iterable

from qdrant_client import QdrantClient, models

ALIAS = "rcqa_chunks"
PAYLOAD_FIELDS = ["chunk_id", "source_id", "document_type", "title", "section", "page", "resident_id",
                  "source_date", "citation", "citation_url", "text", "config_id"]
PROFILE_SECTIONS = ["Fall risk and falls", "Injuries", "Active conditions", "Active medications", "Allergies",
                    "Recent encounters"]   # order rendered by rcqa.profile_text
PUBLIC_TYPES = ["guideline", "ipac_guidance", "legislation", "regulation", "drug_label"]
_NS = uuid.UUID("5b1f6c1e-3c0a-4c55-9c39-2d7b0e0b9a11")


def collection_name(config_id: str) -> str:
    return f"chunks_{config_id}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


def ensure_collection(client: QdrantClient, name: str, dim: int, recreate: bool = False) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(name, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
    for field, schema in [("resident_id", models.PayloadSchemaType.INTEGER),
                          ("document_type", models.PayloadSchemaType.KEYWORD),
                          ("source_id", models.PayloadSchemaType.KEYWORD)]:
        client.create_payload_index(name, field, field_schema=schema)


def _payload(row: dict) -> dict:
    p = {k: row.get(k) for k in PAYLOAD_FIELDS}
    for k in ("resident_id", "page"):
        if p[k] is not None and p[k] == p[k]:   # NaN from pandas -> None
            p[k] = int(p[k])
        else:
            p[k] = None
    return p


def upsert(client: QdrantClient, name: str, rows: Iterable[dict], batch: int = 256) -> int:
    """rows: dicts with the payload fields plus ``vector``. Returns the number of points written."""
    n, buf = 0, []
    for row in rows:
        buf.append(models.PointStruct(id=point_id(row["chunk_id"]), vector=list(map(float, row["vector"])),
                                      payload=_payload(row)))
        if len(buf) == batch:
            client.upsert(name, buf, wait=True)
            n, buf = n + len(buf), []
    if buf:
        client.upsert(name, buf, wait=True)
        n += len(buf)
    return n


def point_alias(client: QdrantClient, name: str, alias: str = ALIAS) -> None:
    """Atomically switch the app's alias to ``name`` (blue/green swap)."""
    ops = [models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=name, alias_name=alias))]
    if alias in {a.alias_name for a in client.get_aliases().aliases}:
        ops.insert(0, models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=ops)


def build_filter(resident_id: int | None = None, document_types: list[str] | None = None) -> models.Filter | None:
    """resident_id set   -> that resident's chunks (+ any public ``document_types`` given).
    resident_id None  -> public documents only; resident chunks are never returned."""
    types = list(document_types or PUBLIC_TYPES)
    public = models.FieldCondition(key="document_type", match=models.MatchAny(any=[t for t in types if t != "resident_profile"]))
    if resident_id is None:
        return models.Filter(must=[public])
    own = models.Filter(must=[models.FieldCondition(key="resident_id", match=models.MatchValue(value=int(resident_id)))])
    should = [own] + ([public] if document_types else [])
    return models.Filter(should=should)


def resident_chunks(client: QdrantClient, collection: str, resident_id: int) -> list[dict]:
    """Every chunk of one resident's record, in profile order (for "Summarize this resident")."""
    f = models.Filter(must=[models.FieldCondition(key="resident_id", match=models.MatchValue(value=int(resident_id)))])
    points, _ = client.scroll(collection, scroll_filter=f, limit=100, with_payload=True)
    order = {s: i for i, s in enumerate(PROFILE_SECTIONS)}
    return sorted((p.payload for p in points), key=lambda p: (order.get(p["section"], 99), p["chunk_id"]))


def search(client: QdrantClient, collection: str, vector: list[float], k: int = 5,
           resident_id: int | None = None, document_types: list[str] | None = None) -> list[dict]:
    res = client.query_points(collection, query=vector, limit=k, with_payload=True,
                              query_filter=build_filter(resident_id, document_types))
    return [{**p.payload, "score": p.score} for p in res.points]
