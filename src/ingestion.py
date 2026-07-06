"""Document loading and chunking.

Supports PDF (via pypdf) and plain-text/markdown files. Text is normalized and
split into overlapping, word-based windows. Each chunk carries its source file
name and page number so answers can cite exactly where information came from.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_pdf(path: Path) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def _read_text(path: Path) -> List[Tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="ignore"))]


_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}


def load_pages(path) -> List[Tuple[int, str]]:
    """Return a list of ``(page_number, raw_text)`` tuples for a document."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in _TEXT_SUFFIXES:
        return _read_text(path)
    raise ValueError(f"Unsupported file type: {suffix!r}")


_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping windows of ``chunk_size`` words."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = chunk_size // 4
    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(path, chunk_size: int, overlap: int) -> List[Chunk]:
    source = Path(path).name
    chunks: List[Chunk] = []
    for page_num, raw in load_pages(path):
        norm = _normalize(raw)
        if not norm:
            continue
        for j, piece in enumerate(chunk_text(norm, chunk_size, overlap)):
            chunk_id = f"{source}::p{page_num}::c{j}"
            chunks.append(Chunk(text=piece, source=source, page=page_num, chunk_id=chunk_id))
    return chunks


def ingest_paths(paths: Iterable, chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in paths:
        all_chunks.extend(build_chunks(p, chunk_size, overlap))
    return all_chunks
