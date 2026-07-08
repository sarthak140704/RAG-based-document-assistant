# Lean image: dependency-light runtime (no torch) using the hashing embedder.
# For semantic embeddings, set EMBEDDING_BACKEND=st and add sentence-transformers.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_BACKEND=hashing \
    LLM_PROVIDER=extractive

# Runtime dependencies (add `sentence-transformers` here to enable EMBEDDING_BACKEND=st).
RUN pip install --no-cache-dir \
    "numpy>=1.26" "pypdf>=4.2" "python-dotenv>=1.0" "streamlit>=1.36" "openai>=1.30" "requests>=2.31"

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
