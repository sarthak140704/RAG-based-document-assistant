"""Central configuration, loaded from environment variables with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- LLM ---
    llm_provider: str = os.getenv("LLM_PROVIDER", "extractive").lower()
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # --- Embeddings ---
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "st").lower()  # st | hashing
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    hashing_dim: int = _env_int("HASHING_DIM", 512)

    # --- Chunking (in words) ---
    chunk_size: int = _env_int("CHUNK_SIZE", 200)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 40)

    # --- Retrieval ---
    top_k: int = _env_int("TOP_K", 4)

    # --- Paths ---
    storage_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "storage")
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")

    def __post_init__(self) -> None:
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
