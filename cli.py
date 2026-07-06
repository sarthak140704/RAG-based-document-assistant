"""Command-line interface for the AI Research Assistant.

Examples::

    # Index documents
    python cli.py ingest data/sample.txt

    # Ask a question
    python cli.py ask "What is retrieval-augmented generation?"

    # Run the evaluation harness
    python cli.py eval data/eval_qa.json

    # Clear the index
    python cli.py reset
"""
from __future__ import annotations

import argparse
import json
import sys

from src.config import settings
from src.evaluation import evaluate, load_qa_set
from src.rag import RAGPipeline


def _cmd_ingest(pipeline: RAGPipeline, args) -> None:
    added = pipeline.ingest(args.paths)
    print(f"Indexed {added} chunks. Store now holds {pipeline.store.size} chunks.")


def _cmd_ask(pipeline: RAGPipeline, args) -> None:
    res = pipeline.answer(args.question)
    print("\nAnswer:\n" + res.answer + "\n")
    print("Sources:")
    for i, s in enumerate(res.sources, start=1):
        print(f"  [{i}] {s.get('source')} (page {s.get('page')}) score={s.get('score', 0):.3f}")


def _cmd_eval(pipeline: RAGPipeline, args) -> None:
    qa = load_qa_set(args.qa_path)
    result = evaluate(pipeline, qa)
    print(result.summary())
    if args.verbose:
        print(json.dumps(result.details, indent=2, ensure_ascii=False))


def _cmd_reset(pipeline: RAGPipeline, args) -> None:
    pipeline.reset()
    print("Index cleared.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="AI Research Assistant (RAG) CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Index one or more documents")
    p_ingest.add_argument("paths", nargs="+")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=_cmd_ask)

    p_eval = sub.add_parser("eval", help="Run the evaluation harness")
    p_eval.add_argument("qa_path")
    p_eval.add_argument("-v", "--verbose", action="store_true")
    p_eval.set_defaults(func=_cmd_eval)

    p_reset = sub.add_parser("reset", help="Clear the index")
    p_reset.set_defaults(func=_cmd_reset)

    args = parser.parse_args(argv)
    pipeline = RAGPipeline(settings)
    args.func(pipeline, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
