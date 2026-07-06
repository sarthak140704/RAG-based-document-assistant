"""The end-to-end RAG pipeline: ingest -> retrieve -> generate (with citations)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import settings as default_settings
from .ingestion import ingest_paths
from .llm import get_llm
from .vectorstore import VectorStore


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Dict]

    def to_dict(self) -> dict:
        return {"question": self.question, "answer": self.answer, "sources": self.sources}


class RAGPipeline:
    def __init__(self, settings=None, store=None, llm=None):
        self.settings = settings or default_settings
        self.store = store or VectorStore(self.settings.storage_dir)
        self.llm = llm or get_llm(self.settings)

    def ingest(self, paths) -> int:
        chunks = ingest_paths(paths, self.settings.chunk_size, self.settings.chunk_overlap)
        return self.store.add_chunks(chunks)

    def retrieve(self, question: str, k: Optional[int] = None) -> List[Dict]:
        return self.store.query(question, k or self.settings.top_k)

    def answer(self, question: str, k: Optional[int] = None) -> Answer:
        sources = self.retrieve(question, k)
        text = self.llm.generate(question, sources)
        return Answer(question=question, answer=text, sources=sources)

    def reset(self) -> None:
        self.store.reset()
