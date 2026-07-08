"""The end-to-end RAG pipeline: ingest -> retrieve -> generate (with citations)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import settings as default_settings
from .ingestion import ingest_paths
from .llm import get_llm
from .retrieval import HybridRetriever, get_reranker
from .vectorstore import VectorStore


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Dict]

    @property
    def confidence(self) -> float:
        """Top retrieval score (0 if nothing retrieved)."""
        return float(self.sources[0]["score"]) if self.sources else 0.0

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
        }


class RAGPipeline:
    def __init__(self, settings=None, store=None, llm=None):
        self.settings = settings or default_settings
        self.store = store or VectorStore(self.settings.storage_dir)
        self.llm = llm or get_llm(self.settings)
        self.retriever = HybridRetriever(self.store, reranker=get_reranker(self.settings))

    def ingest(self, paths) -> int:
        chunks = ingest_paths(paths, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.store.add_chunks(chunks)

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Dict]:
        top_k = k or self.settings.top_k
        if getattr(self.settings, "retrieval_mode", "hybrid") == "hybrid":
            results = self.retriever.search(
                question,
                top_k=top_k,
                candidate_k=getattr(self.settings, "candidate_k", 10),
                rerank=getattr(self.settings, "rerank", "none") != "none",
            )
        else:
            results = self.store.query(question, top_k)
        min_score = getattr(self.settings, "min_score", 0.0)
        if min_score > 0:
            results = [r for r in results if r.get("score", 0.0) >= min_score]
        return results

    def answer(self, question: str, k: Optional[int] = None, history=None) -> Answer:
        search_q = self.llm.condense_query(question, history) if history else question
        sources = self.retrieve(search_q, k)
        text = self.llm.generate(question, sources, history)
        return Answer(question=question, answer=text, sources=sources)

    def stream_answer(self, question: str, k: Optional[int] = None, history=None):
        """Return ``(sources, token_iterator)`` for streaming UIs."""
        search_q = self.llm.condense_query(question, history) if history else question
        sources = self.retrieve(search_q, k)
        return sources, self.llm.stream(question, sources, history)

    def reset(self) -> None:
        self.store.reset()
