# 📚 AI Research Assistant (RAG)

Ask questions across your own documents (PDF / TXT / Markdown) and get concise
answers **with inline citations** back to the exact source and page. Built as a
clean, from-scratch **Retrieval-Augmented Generation (RAG)** pipeline.

> Final-year / portfolio project. Designed to be *impressive but practical*:
> it runs fully offline with zero API keys, yet upgrades to real LLMs and
> semantic embeddings with a single environment variable.

---

## ✨ Features

- **End-to-end RAG**: ingestion → chunking → embedding → retrieval → grounded generation.
- **Citations everywhere** — every answer references the passages it used (`[1]`, `[2]`), with source file + page.
- **Runs offline, no keys needed** — a built-in *extractive* answerer and a pure-NumPy *hashing* embedder mean `pip install` → run.
- **Pluggable LLMs** — OpenAI, Azure OpenAI, or local Ollama via one env var.
- **Pluggable embeddings** — `sentence-transformers` for quality, or hashing for speed/offline.
- **Persistent vector store** — lightweight NumPy cosine store (drop-in concept for FAISS / Chroma / Pinecone).
- **Evaluation harness** — measures retrieval hit-rate, citation rate, and answer grounding.
- **Two front-ends** — a Streamlit chat UI and a CLI.
- **Tested** — unit tests for chunking, retrieval, persistence, and evaluation.

## 🏗️ Architecture

```
                ┌────────────┐   chunks   ┌──────────────┐  vectors  ┌───────────────┐
  PDF/TXT/MD ─▶ │ ingestion  │ ─────────▶ │  embeddings  │ ────────▶ │  vector store │
                └────────────┘            └──────────────┘           └───────┬───────┘
                                                                             │ top-k
   question ──────────────────────────────────────────────────────▶ retrieve │
                                                                             ▼
                                              ┌──────────────┐   answer + [citations]
                                              │     LLM      │ ─────────────────────────▶
                                              └──────────────┘
```

| Module | Responsibility |
| --- | --- |
| `src/ingestion.py` | Load PDFs/text, normalize, sliding-window chunk with page metadata |
| `src/embeddings.py` | `sentence-transformers` or pure-NumPy hashing embedder |
| `src/vectorstore.py` | Persistent, normalized cosine-similarity store |
| `src/llm.py` | Provider-agnostic LLM + extractive offline fallback |
| `src/rag.py` | Orchestrates ingest / retrieve / answer |
| `src/evaluation.py` | Retrieval + grounding metrics |
| `app.py` / `cli.py` | Streamlit UI / command line |

## 🚀 Quickstart

```powershell
# 1. (optional) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install
pip install -r requirements.txt

# 3. copy the example env (defaults work offline, no keys required)
copy .env.example .env

# 4a. run the web app
streamlit run app.py

# 4b. …or use the CLI
python cli.py ingest data/sample.txt
python cli.py ask "What is retrieval-augmented generation?"
```

### ⚡ Fastest offline start (no torch download)

Set the hashing embedder to skip the large `sentence-transformers`/torch download:

```powershell
$env:EMBEDDING_BACKEND = "hashing"
python cli.py ingest data/sample.txt
python cli.py ask "Why does chunk overlap matter?"
```

## 🔌 Using a real LLM

Edit `.env`:

```ini
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# or local Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3
```

For best retrieval quality, keep `EMBEDDING_BACKEND=st` (the default).

## 📊 Evaluation

```powershell
python cli.py ingest data/sample.txt
python cli.py eval data/eval_qa.json -v
```

Reports, for the QA set in `data/eval_qa.json`:

- **retrieval_hit_rate** — did the right document get retrieved?
- **citation_rate** — did the answer include citations?
- **grounded_rate** — did the answer contain the expected fact?

## 🧪 Tests

```powershell
pip install pytest
python -m pytest -q
```

## ⚙️ Configuration (env vars)

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `extractive` | `extractive` \| `openai` \| `azure` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `EMBEDDING_BACKEND` | `st` | `st` (sentence-transformers) \| `hashing` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `200` | Chunk size in words |
| `CHUNK_OVERLAP` | `40` | Overlap in words |
| `TOP_K` | `4` | Passages retrieved per query |

## 🗺️ Roadmap / stretch goals

- Hybrid search (keyword + vector) and re-ranking
- Streaming answers and conversation memory
- Swap the NumPy store for FAISS/Chroma at larger scale
- LLM-as-a-judge faithfulness scoring in the eval harness

## 🎤 Resume talking points

- Built a production-shaped **RAG** system with **source-grounded citations**.
- Designed **provider- and embedding-agnostic** abstractions (offline fallback → real LLMs).
- Added a **quantitative evaluation harness** separating retrieval vs. generation quality.
