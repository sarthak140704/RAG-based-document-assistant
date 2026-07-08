"""Hybrid retrieval: dense (vector) + sparse (BM25), fused with Reciprocal Rank
Fusion, with an optional cross-encoder re-ranking stage.

Why this matters
----------------
* Dense retrieval finds semantically similar text (paraphrases, synonyms).
* BM25 finds exact keyword / rare-term matches dense models sometimes miss.
* Reciprocal Rank Fusion (RRF) merges the two rankings without needing to
  normalize their very different score scales.
* An optional cross-encoder re-ranker reads each (query, passage) pair jointly
  for a final, higher-precision ordering.

Every returned candidate keeps its dense cosine ``score`` (used for the
confidence signal) plus a ``fused_score`` (RRF) and, if re-ranked, a
``rerank_score``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .keyword import BM25Index

RRF_K = 60  # standard RRF damping constant


class CrossEncoderReranker:
    """Re-orders candidates with a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return candidates
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)


def get_reranker(settings) -> Optional[CrossEncoderReranker]:
    if getattr(settings, "rerank", "none") == "cross-encoder":
        try:
            return CrossEncoderReranker(settings.rerank_model)
        except Exception as exc:  # pragma: no cover - optional heavy dep
            print(f"[retrieval] cross-encoder unavailable ({exc}); skipping re-rank.")
    return None


class HybridRetriever:
    def __init__(self, store, reranker: Optional[CrossEncoderReranker] = None, rrf_k: int = RRF_K):
        self.store = store
        self.reranker = reranker
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25Index] = None
        self._bm25_size = -1

    def _ensure_bm25(self) -> None:
        """(Re)build the BM25 index when the store's contents change."""
        if self._bm25 is None or self._bm25_size != self.store.size:
            self._bm25 = BM25Index([m.get("text", "") for m in self.store.meta])
            self._bm25_size = self.store.size

    def search(
        self, query: str, top_k: int = 4, candidate_k: int = 10, rerank: bool = False
    ) -> List[Dict]:
        if self.store.size == 0:
            return []
        self._ensure_bm25()

        dense = self.store.query(query, candidate_k)
        sparse = self._bm25.query(query, candidate_k)
        sparse_meta = [self.store.meta[i] for i, _ in sparse]

        dense_ranks = {d["chunk_id"]: r for r, d in enumerate(dense)}
        sparse_ranks = {m["chunk_id"]: r for r, m in enumerate(sparse_meta)}

        # Union of candidates keyed by chunk_id.
        candidates: Dict[str, Dict] = {}
        for d in dense:
            candidates[d["chunk_id"]] = dict(d)
        for m in sparse_meta:
            candidates.setdefault(m["chunk_id"], dict(m))

        # Attach a consistent dense cosine score to every candidate.
        sims = self.store.cosine_all(self.store.embed_query(query))
        id_to_idx = self.store.id_to_idx

        fused: List[Dict] = []
        for cid, meta in candidates.items():
            rrf = 0.0
            if cid in dense_ranks:
                rrf += 1.0 / (self.rrf_k + dense_ranks[cid])
            if cid in sparse_ranks:
                rrf += 1.0 / (self.rrf_k + sparse_ranks[cid])
            idx = id_to_idx.get(cid)
            if idx is not None and idx < len(sims):
                meta["score"] = float(sims[idx])
            meta["fused_score"] = rrf
            fused.append(meta)

        fused.sort(key=lambda m: m["fused_score"], reverse=True)

        if rerank and self.reranker:
            pool = self.reranker.rerank(query, fused[:candidate_k])
            return pool[:top_k]
        return fused[:top_k]
