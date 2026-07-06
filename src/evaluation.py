"""A small evaluation harness for the RAG pipeline.

Given a QA set (list of dicts), it measures three placement-friendly metrics:

* retrieval_hit_rate - fraction of questions whose ``expected_source`` appears
  in the retrieved passages (did we fetch the right document?).
* citation_rate      - fraction of answers that contain a bracketed citation.
* grounded_rate      - fraction of answers containing ``expected_answer_contains``
  (a cheap proxy for factual grounding).

QA item schema::

    {
      "question": "...",
      "expected_source": "sample.txt",              # optional
      "expected_answer_contains": "photovoltaic"    # optional
    }
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List

_CITATION_RE = re.compile(r"\[\d+\]")


@dataclass
class EvalResult:
    n: int
    retrieval_hit_rate: float
    citation_rate: float
    grounded_rate: float
    details: List[Dict]

    def summary(self) -> str:
        return (
            f"n={self.n}  "
            f"retrieval_hit_rate={self.retrieval_hit_rate:.2f}  "
            f"citation_rate={self.citation_rate:.2f}  "
            f"grounded_rate={self.grounded_rate:.2f}"
        )


def _has_citation(answer: str) -> bool:
    return bool(_CITATION_RE.search(answer))


def evaluate(pipeline, qa_set: List[Dict]) -> EvalResult:
    details: List[Dict] = []
    hits = cited = grounded = 0

    for item in qa_set:
        question = item["question"]
        expected_source = item.get("expected_source")
        expected_substr = (item.get("expected_answer_contains") or "").lower()

        res = pipeline.answer(question)
        retrieved = {s.get("source") for s in res.sources}

        hit = (expected_source in retrieved) if expected_source else None
        has_cite = _has_citation(res.answer)
        is_grounded = (expected_substr in res.answer.lower()) if expected_substr else None

        hits += 1 if hit else 0
        cited += 1 if has_cite else 0
        grounded += 1 if is_grounded else 0

        details.append(
            {
                "question": question,
                "answer": res.answer,
                "retrieved_sources": sorted(s for s in retrieved if s),
                "expected_source": expected_source,
                "retrieval_hit": hit,
                "has_citation": has_cite,
                "grounded": is_grounded,
            }
        )

    denom = len(qa_set) or 1
    return EvalResult(
        n=len(qa_set),
        retrieval_hit_rate=round(hits / denom, 3),
        citation_rate=round(cited / denom, 3),
        grounded_rate=round(grounded / denom, 3),
        details=details,
    )


def load_qa_set(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
