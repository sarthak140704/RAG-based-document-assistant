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


_PARA_RE = re.compile(r"\n\s*\n")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines; collapse inner whitespace per paragraph."""
    return [_WS_RE.sub(" ", p).strip() for p in _PARA_RE.split(text) if p.strip()]


def split_sentences(text: str) -> List[str]:
    """Naive sentence splitter on ., !, ? boundaries."""
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


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


def smart_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Structure-aware chunking.

    Packs whole sentences (respecting paragraph boundaries) into chunks of up to
    ``chunk_size`` words, carrying trailing sentences up to ``overlap`` words into
    the next chunk. Sentences longer than ``chunk_size`` fall back to word windows.
    This keeps ideas intact far better than a blind word window.
    """
    segments: List[str] = []
    for para in split_paragraphs(text):
        sentences = split_sentences(para)
        segments.extend(sentences if sentences else [para])
    if not segments:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg.split())
        if seg_len > chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(chunk_text(seg, chunk_size, overlap))
            continue
        if current and current_len + seg_len > chunk_size:
            chunks.append(" ".join(current))
            # Carry trailing sentences (up to `overlap` words) into the next chunk.
            keep: List[str] = []
            keep_len = 0
            for s in reversed(current):
                s_len = len(s.split())
                if keep_len + s_len > overlap:
                    break
                keep.insert(0, s)
                keep_len += s_len
            current, current_len = keep, keep_len
        current.append(seg)
        current_len += seg_len

    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def build_chunks(path, chunk_size: int, overlap: int) -> List[Chunk]:
    source = Path(path).name
    chunks: List[Chunk] = []
    for page_num, raw in load_pages(path):
        if not raw or not raw.strip():
            continue
        for j, piece in enumerate(smart_chunks(raw, chunk_size, overlap)):
            chunk_id = f"{source}::p{page_num}::c{j}"
            chunks.append(Chunk(text=piece, source=source, page=page_num, chunk_id=chunk_id))
    return chunks


def ingest_paths(paths: Iterable, chunk_size: int, overlap: int) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for p in paths:
        all_chunks.extend(build_chunks(p, chunk_size, overlap))
    return all_chunks
