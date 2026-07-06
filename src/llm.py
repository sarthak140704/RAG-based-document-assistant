"""LLM abstraction layer.

A single ``get_llm(settings)`` factory returns a provider-agnostic object with a
``generate(question, sources)`` method. Supported providers:

* ``extractive`` - no API key required; stitches the most relevant passages into
  a cited answer so the whole app runs offline. This is the default.
* ``openai``     - OpenAI (or any OpenAI-compatible base URL).
* ``azure``      - Azure OpenAI.
* ``ollama``     - a local Ollama server.

All LLM providers share a strict, citation-enforcing prompt.
"""
from __future__ import annotations

from typing import Dict, List

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the numbered context passages provided. Cite every claim with bracketed "
    "numbers like [1] or [2] that refer to the passages you used. If the answer "
    'is not contained in the context, reply exactly: "I couldn\'t find this in '
    'the provided documents." Be concise.'
)


def build_context(sources: List[Dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        loc = f"{s.get('source', '?')} (page {s.get('page', '?')})"
        blocks.append(f"[{i}] {loc}\n{s.get('text', '')}")
    return "\n\n".join(blocks)


def build_user_prompt(question: str, sources: List[Dict]) -> str:
    return (
        f"Context passages:\n\n{build_context(sources)}\n\n"
        f"Question: {question}\n\nAnswer with citations:"
    )


class BaseLLM:
    def generate(self, question: str, sources: List[Dict]) -> str:  # pragma: no cover
        raise NotImplementedError


class ExtractiveLLM(BaseLLM):
    """No-API fallback: returns a cited answer built from the top passages."""

    def generate(self, question: str, sources: List[Dict]) -> str:
        if not sources:
            return "I couldn't find this in the provided documents."
        parts = []
        for i, s in enumerate(sources[:3], start=1):
            snippet = (s.get("text") or "").strip()
            if len(snippet) > 400:
                snippet = snippet[:400].rsplit(" ", 1)[0] + "\u2026"
            parts.append(f"{snippet} [{i}]")
        return " ".join(parts)


class _ChatLLM(BaseLLM):
    """Shared chat-completions call for OpenAI-style clients."""

    client = None
    model = ""

    def generate(self, question: str, sources: List[Dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, sources)},
            ],
            temperature=0.0,
        )
        return (resp.choices[0].message.content or "").strip()


class OpenAILLM(_ChatLLM):
    def __init__(self, model: str, api_key: str, base_url: str = ""):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model


class AzureOpenAILLM(_ChatLLM):
    def __init__(self, model: str, api_key: str, endpoint: str, api_version: str):
        from openai import AzureOpenAI

        self.client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )
        self.model = model


class OllamaLLM(BaseLLM):
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, question: str, sources: List[Dict]) -> str:
        import requests

        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(question, sources)},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


def get_llm(settings):
    """Build an LLM from settings, falling back to extractive on any failure."""
    provider = settings.llm_provider
    try:
        if provider == "openai" and settings.openai_api_key:
            return OpenAILLM(
                settings.llm_model, settings.openai_api_key, settings.openai_base_url
            )
        if provider == "azure" and settings.openai_api_key and settings.azure_endpoint:
            return AzureOpenAILLM(
                settings.llm_model,
                settings.openai_api_key,
                settings.azure_endpoint,
                settings.azure_api_version,
            )
        if provider == "ollama":
            return OllamaLLM(settings.llm_model, settings.ollama_base_url)
    except Exception as exc:  # pragma: no cover - network/dep dependent
        print(f"[llm] provider {provider!r} init failed ({exc}); using extractive fallback.")
    return ExtractiveLLM()
