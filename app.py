"""Streamlit UI for the AI Research Assistant.

Run with:  streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

# Bridge Streamlit Cloud secrets -> environment variables BEFORE importing
# src.config, so settings (which read os.getenv) pick them up on deploy.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:
    pass

from src.config import settings
from src.rag import Answer, RAGPipeline

st.set_page_config(page_title="Research Assistant", page_icon="\U0001F4DA", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(settings)


pipeline = get_pipeline()

st.title("\U0001F4DA Research Assistant")
st.caption("Ask questions across your own documents and get answers with citations (RAG).")

with st.sidebar:
    st.header("Add documents")
    uploads = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", disabled=not uploads, type="primary"):
        tmpdir = Path(tempfile.mkdtemp())
        paths = []
        for uf in uploads:
            p = tmpdir / uf.name
            p.write_bytes(uf.getbuffer())
            paths.append(p)
        with st.spinner("Chunking + embedding\u2026"):
            added = pipeline.ingest(paths)
        st.success(f"Indexed {added} chunks from {len(paths)} file(s).")

    st.divider()
    st.metric("Indexed chunks", pipeline.store.size)
    st.write(f"**LLM provider:** `{settings.llm_provider}`")
    st.write(f"**Embeddings:** `{settings.embedding_backend}`")
    st.write(f"**Top-k:** {settings.top_k}")
    if st.button("Reset index"):
        pipeline.reset()
        st.session_state.pop("history", None)
        st.rerun()

st.header("Ask a question")

if "history" not in st.session_state:
    st.session_state.history = []  # list[Answer]


def _confidence_label(conf: float) -> str:
    hi, mid = (0.5, 0.25) if settings.embedding_backend == "st" else (0.15, 0.06)
    if conf >= hi:
        return "High"
    if conf >= mid:
        return "Medium"
    return "Low"


def _render_sources(res: Answer) -> None:
    conf = res.confidence
    label = _confidence_label(conf)
    with st.expander(f"Sources ({len(res.sources)}) \u00b7 confidence: {label} ({conf:.2f})"):
        if not res.sources:
            st.caption("No matching passages found.")
        for i, s in enumerate(res.sources, start=1):
            st.markdown(
                f"**[{i}] {s.get('source')}** \u2014 page {s.get('page')} "
                f"\u00b7 score {s.get('score', 0):.3f}"
            )
            text = s.get("text", "")
            st.caption(text[:500] + ("\u2026" if len(text) > 500 else ""))


# Render the prior conversation (chronological).
for res in st.session_state.history:
    with st.chat_message("user"):
        st.write(res.question)
    with st.chat_message("assistant"):
        st.write(res.answer)
        _render_sources(res)

question = st.chat_input("Ask something about your documents\u2026")
if question:
    if pipeline.store.size == 0:
        st.warning("Index some documents first (use the sidebar).")
    else:
        with st.chat_message("user"):
            st.write(question)
        history_pairs = [(r.question, r.answer) for r in st.session_state.history]
        sources, token_iter = pipeline.stream_answer(question, history=history_pairs)
        with st.chat_message("assistant"):
            answer_text = st.write_stream(token_iter)
            res = Answer(question=question, answer=answer_text, sources=sources)
            if res.confidence and _confidence_label(res.confidence) == "Low":
                st.caption("\u26a0\ufe0f Low retrieval confidence \u2014 answer may be incomplete.")
            _render_sources(res)
        st.session_state.history.append(res)
