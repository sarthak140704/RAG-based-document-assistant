import numpy as np

from src.embeddings import HashingEmbedder
from src.evaluation import evaluate
from src.ingestion import build_chunks
from src.llm import ExtractiveLLM
from src.rag import RAGPipeline
from src.vectorstore import VectorStore


def test_hashing_embedder_normalized_and_fixed_dim():
    emb = HashingEmbedder(128)
    vecs = emb.encode(["hello world", "a completely different sentence"])
    assert vecs.shape == (2, 128)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_vectorstore_add_query_and_persist(tmp_path):
    store = VectorStore(tmp_path / "storage", embedder=HashingEmbedder(256))
    doc = tmp_path / "solar.txt"
    doc.write_text(
        "The photovoltaic effect converts sunlight into electricity. "
        "Solar panels are made of silicon cells. "
        "Wind turbines convert wind into power. " * 10,
        encoding="utf-8",
    )
    store.add_chunks(build_chunks(doc, chunk_size=20, overlap=5))
    assert store.size > 0

    results = store.query("How do solar panels convert sunlight?", k=3)
    assert results
    assert results[0]["source"] == "solar.txt"
    assert "score" in results[0]

    # Persistence: a fresh store over the same dir must see the data.
    reloaded = VectorStore(tmp_path / "storage", embedder=HashingEmbedder(256))
    assert reloaded.size == store.size


def _pipeline(tmp_path):
    from src.config import Settings

    s = Settings()
    s.storage_dir = tmp_path / "storage"
    s.embedding_backend = "hashing"
    s.top_k = 3
    store = VectorStore(s.storage_dir, embedder=HashingEmbedder(256))
    return RAGPipeline(settings=s, store=store, llm=ExtractiveLLM())


def test_rag_answer_has_citation(tmp_path):
    doc = tmp_path / "solar.txt"
    doc.write_text(
        "The photovoltaic effect converts sunlight into electricity. "
        "Solar panels are made of silicon cells. " * 10,
        encoding="utf-8",
    )
    p = _pipeline(tmp_path)
    p.ingest([doc])

    res = p.answer("How do solar panels convert sunlight?")
    assert res.sources
    assert res.sources[0]["source"] == "solar.txt"
    assert "[1]" in res.answer


def test_extractive_llm_empty_sources():
    assert "couldn't find" in ExtractiveLLM().generate("q", []).lower()


def test_evaluation_metrics(tmp_path):
    doc = tmp_path / "solar.txt"
    doc.write_text("Solar panels use the photovoltaic effect. " * 20, encoding="utf-8")
    p = _pipeline(tmp_path)
    p.ingest([doc])

    qa = [
        {
            "question": "What effect do solar panels use?",
            "expected_source": "solar.txt",
            "expected_answer_contains": "photovoltaic",
        }
    ]
    result = evaluate(p, qa)
    assert result.n == 1
    assert result.retrieval_hit_rate == 1.0
    assert result.citation_rate == 1.0


def _doc(tmp_path):
    doc = tmp_path / "solar.txt"
    doc.write_text("Solar panels use the photovoltaic effect. " * 20, encoding="utf-8")
    return doc


def test_answer_confidence(tmp_path):
    p = _pipeline(tmp_path)
    p.ingest([_doc(tmp_path)])
    res = p.answer("photovoltaic effect")
    assert res.confidence == res.sources[0]["score"]
    assert res.confidence > 0


def test_extractive_streaming_matches_generate(tmp_path):
    p = _pipeline(tmp_path)
    p.ingest([_doc(tmp_path)])
    sources, stream = p.stream_answer("photovoltaic")
    streamed = "".join(stream).strip()
    full = p.llm.generate("photovoltaic", sources).strip()
    assert streamed == full
    assert "[1]" in streamed


def test_condense_query_extractive():
    llm = ExtractiveLLM()
    assert llm.condense_query("why?", None) == "why?"
    out = llm.condense_query("why is that?", [("what is RAG?", "an answer")])
    assert "what is RAG?" in out and "why is that?" in out


def test_min_score_gate(tmp_path):
    p = _pipeline(tmp_path)
    p.ingest([_doc(tmp_path)])
    p.settings.min_score = 999.0  # impossible threshold -> nothing passes
    res = p.answer("photovoltaic")
    assert res.sources == []
    assert "couldn't find" in res.answer.lower()
