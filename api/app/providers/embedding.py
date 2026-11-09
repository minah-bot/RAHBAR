"""Question embedding for the API (section 3.2, step 3).

The indexer embedded every chunk with EMBED_MODEL; the question must be embedded with
the very same model or the similarity scores mean nothing (section 4.4 enforces this).
Ollama serves the embeddings (OLLAMA_URL), including when generation is hosted.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import httpx

from ..config import Settings


class EmbeddingError(RuntimeError):
    """The embedding backend was unreachable or answered nonsense."""


def prepare(model: str, text: str, kind: str) -> str:
    """nomic-embed-text was trained with task prefixes; without them recall drops.
    The indexer applies the same function with kind="document". Other models: no-op."""
    if model.startswith("nomic-embed-text"):
        return f"search_{kind}: {text}"
    return text


class OllamaEmbedder:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.embed_model
        self.client = httpx.Client(
            base_url=settings.ollama_url, timeout=httpx.Timeout(30.0, connect=5.0)
        )
        # The cache lookup and the retriever embed the same question back to back;
        # a tiny memo makes the second call free.
        self._memo: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    def embed_query(self, question: str) -> list[float]:
        with self._lock:
            if question in self._memo:
                self._memo.move_to_end(question)
                return self._memo[question]
        try:
            r = self.client.post(
                "/api/embed",
                json={"model": self.model, "input": [prepare(self.model, question, "query")]},
            )
            r.raise_for_status()
            vector = r.json()["embeddings"][0]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise EmbeddingError(f"embedding call failed: {exc!r}") from exc
        with self._lock:
            self._memo[question] = vector
            while len(self._memo) > 256:
                self._memo.popitem(last=False)
        return vector
