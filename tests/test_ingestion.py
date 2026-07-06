from src.ingestion import build_chunks, chunk_text


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
