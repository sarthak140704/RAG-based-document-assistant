"""A small, dependency-free BM25 keyword index.

BM25 is the classic lexical ranking function used by search engines. Combining
it with dense (vector) retrieval — "hybrid search" — catches both exact keyword
matches (BM25) and semantic/paraphrase matches (embeddings), which is more
robust than either alone.
"""
from __future__ import annotations

import math
import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, texts: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [_tokenize(t) for t in texts]
        self.n = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

        df: dict = {}
        self.tf: List[dict] = []
        for doc in self.docs:
            freqs: dict = {}
            for term in doc:
                freqs[term] = freqs.get(term, 0) + 1
            self.tf.append(freqs)
            for term in freqs:
                df[term] = df.get(term, 0) + 1

        self.idf = {
            term: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    def query(self, text: str, k: int) -> List[Tuple[int, float]]:
        """Return the top-k ``(doc_index, score)`` for non-zero BM25 scores."""
        if self.n == 0 or self.avgdl == 0:
            return []
        terms = _tokenize(text)
        scored: List[Tuple[int, float]] = []
        for i in range(self.n):
            freqs = self.tf[i]
            dl = self.doc_len[i]
            score = 0.0
            for term in terms:
                f = freqs.get(term)
                if not f:
                    continue
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
