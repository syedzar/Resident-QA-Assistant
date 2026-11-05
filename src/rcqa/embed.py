"""Embedding model shared by Databricks (indexing) and the app (queries).

BGE-small via fastembed (ONNX on CPU): free, 384 dimensions, no PyTorch install, and
the SAME weights on a laptop and on Databricks serverless. Documents and queries must
use the same model; BGE also expects an instruction prefix on queries, which
fastembed's ``query_embed`` adds for us.
"""
from __future__ import annotations

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384


class Embedder:
    def __init__(self, model: str = MODEL, cache_dir: str | None = None, local_files_only: bool = False):
        """cache_dir + local_files_only: load a pre-downloaded copy (Databricks workers have no internet)."""
        from fastembed import TextEmbedding   # imported lazily so tests without the model stay fast
        self.model_name = model
        self._m = TextEmbedding(model, cache_dir=cache_dir, local_files_only=local_files_only)

    def embed_docs(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return [v.tolist() for v in self._m.embed(texts, batch_size=batch_size)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._m.query_embed([text]))).tolist()
