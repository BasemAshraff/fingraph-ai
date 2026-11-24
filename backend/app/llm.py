"""Model layer. OpenAI (or any OpenAI-compatible server) when configured, otherwise offline fallbacks.

The offline embedder is a deterministic hashed bag-of-words. It has no semantic understanding,
so it is only for tests and for running the app without an API key. Real retrieval quality
comes from the OpenAI embeddings.
"""
import hashlib
import logging
import re

import numpy as np

from . import config

log = logging.getLogger("fingraph.llm")


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    dim = 512
    name = "offline-hashing-512"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
                sign = -1.0 if (digest >> 1) & 1 else 1.0
                out[row, digest % self.dim] += sign
        return _normalize(out)


class OpenAIProvider:
    def __init__(self):
        from openai import OpenAI

        kwargs = {"api_key": config.OPENAI_API_KEY or "not-needed"}
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        self.client = OpenAI(**kwargs)
        self.name = f"openai:{config.EMBED_MODEL}"
        self.chat_model = config.CHAT_MODEL

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), 96):  # batch to stay well under API limits
            batch = texts[start : start + 96]
            response = self.client.embeddings.create(model=config.EMBED_MODEL, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return _normalize(np.array(vectors, dtype=np.float32))

    def chat(self, messages: list[dict], json_mode: bool = False, temperature: float = 0.1) -> str:
        kwargs = {"model": self.chat_model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


class OfflineProvider:
    name = "offline"
    chat_model = None

    def __init__(self):
        self.embedder = HashingEmbedder()

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed(texts)

    def chat(self, messages, json_mode=False, temperature=0.1) -> str:
        raise RuntimeError("No LLM configured")


_provider = None


def get_provider():
    """Return the active provider. Chat is unavailable when running offline."""
    global _provider
    if _provider is None:
        if config.OPENAI_API_KEY or config.OPENAI_BASE_URL:
            _provider = OpenAIProvider()
            log.info("Using LLM provider %s", _provider.name)
        else:
            _provider = OfflineProvider()
            log.warning("OPENAI_API_KEY not set: running in offline mode")
    return _provider


def llm_available() -> bool:
    return not isinstance(get_provider(), OfflineProvider)


def embed_texts(texts: list[str]) -> np.ndarray:
    provider = get_provider()
    if not texts:
        return np.zeros((0, HashingEmbedder.dim), dtype=np.float32)
    return provider.embed(texts)
