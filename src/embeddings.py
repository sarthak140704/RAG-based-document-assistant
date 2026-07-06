"""Text embedding backends.

Two interchangeable backends, both returning L2-normalized, fixed-dimension
vectors so the vector store can use a plain dot product for cosine similarity:

* ``SentenceTransformerEmbedder`` - best semantic quality (needs torch).
* ``HashingEmbedder`` - pure NumPy, deterministic, fast, offline. Ideal for
  tests, CI, and quick demos with no heavy downloads.
"""
from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

from .config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Signed feature-hashing embedder over unigrams + bigrams (pure NumPy)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    @property
    def dimension(self) -> int:
        return self.dim

    def _add(self, vec: np.ndarray, feature: str) -> None:
        h = int(hashlib.md5(feature.encode("utf-8")).hexdigest(), 16)
        idx = h % self.dim
        sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
        vec[idx] += sign

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _tokenize(text)
        for i, tok in enumerate(tokens):
            self._add(vec, tok)
            if i + 1 < len(tokens):
                self._add(vec, tok + "_" + tokens[i + 1])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (downloaded on first use)."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # Method was renamed across sentence-transformers versions.
        get_dim = getattr(
            self.model,
            "get_embedding_dimension",
            getattr(self.model, "get_sentence_embedding_dimension", None),
        )
        self._dim = int(get_dim()) if get_dim else int(
            self.model.encode(["_"], convert_to_numpy=True).shape[1]
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        emb = self.model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return emb.astype(np.float32)


def get_embedder():
    """Return the configured embedder, falling back to hashing if ST is missing."""
    if settings.embedding_backend == "hashing":
        return HashingEmbedder(settings.hashing_dim)
    try:
        return SentenceTransformerEmbedder(settings.embedding_model)
    except Exception as exc:  # pragma: no cover - depends on optional dep
        print(
            f"[embeddings] sentence-transformers unavailable ({exc}); "
            "falling back to the hashing embedder."
        )
        return HashingEmbedder(settings.hashing_dim)
