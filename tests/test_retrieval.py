from src.config import Settings
from src.embeddings import HashingEmbedder
from src.ingestion import build_chunks
from src.keyword import BM25Index
from src.llm import ExtractiveLLM
from src.rag import RAGPipeline
from src.retrieval import HybridRetriever
from src.vectorstore import VectorStore


def test_bm25_ranks_keyword_match():
    texts = [
        "the photovoltaic effect converts sunlight into electricity",
        "wind turbines convert wind into power",
        "hydro dams use falling water",
    ]
    bm25 = BM25Index(texts)
    top = bm25.query("photovoltaic sunlight", k=3)
    assert top, "expected non-empty BM25 results"
    assert top[0][0] == 0  # first document is the best keyword match


def test_bm25_empty_and_no_match():
    assert BM25Index([]).query("anything", k=3) == []
    bm25 = BM25Index(["alpha beta", "gamma delta"])
    assert bm25.query("zzz", k=3) == []


def _store(tmp_path):
    doc = tmp_path / "energy.txt"
    doc.write_text(
        "The photovoltaic effect converts sunlight into electricity. "
        "Solar panels are made of silicon cells. "
        "Wind turbines convert wind into power. "
        "Hydro dams use falling water to spin turbines. ",
        encoding="utf-8",
    )
    store = VectorStore(tmp_path / "storage", embedder=HashingEmbedder(256))
    store.add_chunks(build_chunks(doc, chunk_size=12, overlap=3))
    return store


def test_hybrid_retriever_returns_scored_results(tmp_path):
    store = _store(tmp_path)
    retriever = HybridRetriever(store)
    results = retriever.search("how do solar panels use sunlight", top_k=3, candidate_k=5)
    assert results
    assert all("score" in r and "fused_score" in r for r in results)
    # Results are ordered by fused score.
    fused = [r["fused_score"] for r in results]
    assert fused == sorted(fused, reverse=True)


def test_hybrid_rebuilds_when_store_grows(tmp_path):
    store = _store(tmp_path)
    retriever = HybridRetriever(store)
    retriever.search("sunlight", top_k=2, candidate_k=3)
    first_size = retriever._bm25_size
    # Add more content; the BM25 index must rebuild on next search.
    doc2 = tmp_path / "extra.txt"
    doc2.write_text("Geothermal energy taps heat from the earth. " * 5, encoding="utf-8")
    store.add_chunks(build_chunks(doc2, chunk_size=10, overlap=2))
    res = retriever.search("geothermal earth heat", top_k=3, candidate_k=5)
    assert retriever._bm25_size != first_size
    assert any(r["source"] == "extra.txt" for r in res)


def test_pipeline_hybrid_mode(tmp_path):
    s = Settings()
    s.storage_dir = tmp_path / "storage"
    s.embedding_backend = "hashing"
    s.retrieval_mode = "hybrid"
    s.top_k = 3
    store = VectorStore(s.storage_dir, embedder=HashingEmbedder(256))
    p = RAGPipeline(settings=s, store=store, llm=ExtractiveLLM())
    p.ingest([_store_path(tmp_path)])
    res = p.answer("photovoltaic sunlight electricity")
    assert res.sources
    assert res.sources[0]["source"] == "energy.txt"
    assert "[1]" in res.answer


def _store_path(tmp_path):
    doc = tmp_path / "energy.txt"
    doc.write_text(
        "The photovoltaic effect converts sunlight into electricity. "
        "Solar panels are made of silicon cells. " * 5,
        encoding="utf-8",
    )
    return doc
