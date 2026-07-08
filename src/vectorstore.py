"""A lightweight, persistent vector store backed by NumPy.

Vectors are L2-normalized, so cosine similarity reduces to a dot product. The
store is intentionally dependency-light (NumPy only) for reliability and speed
at student-corpus scale, and is a drop-in concept for FAISS / Chroma / Pinecone.

Persistence layout (under ``storage_dir``):
    embeddings.npy   - float32 matrix of shape (N, dim)
    meta.jsonl       - one JSON object per row (text + citation metadata)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from .embeddings import get_embedder
from .ingestion import Chunk


class VectorStore:
    def __init__(self, storage_dir, embedder=None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.emb_path = self.storage_dir / "embeddings.npy"
        self.meta_path = self.storage_dir / "meta.jsonl"
        self.embedder = embedder or get_embedder()
        self._vectors: Optional[np.ndarray] = None
        self._meta: List[dict] = []
        self._load()

    # --- persistence ---------------------------------------------------
    def _load(self) -> None:
        if self.emb_path.exists() and self.meta_path.exists():
            self._vectors = np.load(self.emb_path)
            lines = self.meta_path.read_text(encoding="utf-8").splitlines()
            self._meta = [json.loads(ln) for ln in lines if ln.strip()]
        else:
            self._vectors = None
            self._meta = []

    def _save(self) -> None:
        if self._vectors is not None:
            np.save(self.emb_path, self._vectors)
        with self.meta_path.open("w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # --- public API ----------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._meta)

    @property
    def meta(self) -> List[dict]:
        return self._meta

    @property
    def id_to_idx(self) -> dict:
        return {m["chunk_id"]: i for i, m in enumerate(self._meta)}

    def embed_query(self, text: str) -> np.ndarray:
        return self.embedder.encode([text])[0]

    def cosine_all(self, qvec: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query vector against every stored vector."""
        if self._vectors is None:
            return np.zeros(0, dtype=np.float32)
        return self._vectors @ qvec

    def add_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0
        vecs = self.embedder.encode([c.text for c in chunks])
        self._vectors = vecs if self._vectors is None else np.vstack([self._vectors, vecs])
        self._meta.extend(c.to_dict() for c in chunks)
        self._save()
        return len(chunks)

    def query(self, text: str, k: int = 4) -> List[dict]:
        if self._vectors is None or not self._meta:
            return []
        q = self.embedder.encode([text])[0]
        sims = self._vectors @ q  # cosine similarity (vectors are normalized)
        k = min(k, len(self._meta))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        results = []
        for i in top:
            m = dict(self._meta[int(i)])
            m["score"] = float(sims[int(i)])
            results.append(m)
        return results

    def reset(self) -> None:
        self._vectors = None
        self._meta = []
        for p in (self.emb_path, self.meta_path):
            if p.exists():
                p.unlink()
