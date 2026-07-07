from src.ingestion import build_chunks, chunk_text, smart_chunks, split_paragraphs, split_sentences


def test_chunk_overlap_and_coverage():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) >= 4
    # Overlap: the last word of chunk 0 should reappear near the start of chunk 1.
    c0, c1 = chunks[0].split(), chunks[1].split()
    assert c0[-1] in c1[:15]
    # Coverage: the very last word must appear in the final chunk.
    assert words[-1] in chunks[-1].split()


def test_chunk_empty_text():
    assert chunk_text("", 30, 10) == []
    assert chunk_text("   \n\t ", 30, 10) == []


def test_overlap_larger_than_size_is_clamped():
    words = [f"w{i}" for i in range(50)]
    # overlap >= chunk_size would loop forever if not clamped
    chunks = chunk_text(" ".join(words), chunk_size=10, overlap=20)
    assert len(chunks) > 1


def test_build_chunks_metadata(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("alpha beta gamma delta " * 50, encoding="utf-8")

    chunks = build_chunks(f, chunk_size=20, overlap=5)

    assert chunks
    assert all(c.source == "doc.txt" for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert chunks[0].chunk_id == "doc.txt::p1::c0"


def test_split_paragraphs_and_sentences():
    text = "First para line one.\n\nSecond para here. It has two sentences."
    paras = split_paragraphs(text)
    assert len(paras) == 2
    assert split_sentences(paras[1]) == ["Second para here.", "It has two sentences."]


def test_smart_chunks_packs_whole_sentences():
    text = (
        "Alpha is the first item. Beta is the second item. Gamma is the third item. "
        "Delta is the fourth item. Epsilon is the fifth item."
    )
    chunks = smart_chunks(text, chunk_size=6, overlap=2)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    for word in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]:
        assert word in joined


def test_smart_chunks_long_sentence_falls_back_to_words():
    long_sentence = " ".join(f"w{i}" for i in range(50))  # no punctuation
    chunks = smart_chunks(long_sentence, chunk_size=10, overlap=2)
    assert len(chunks) > 1
    assert "w49" in " ".join(chunks)
