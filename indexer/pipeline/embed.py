"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import httpx
import math

from .config import IndexerSettings


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else.

    With Ollama: POST {ollama_url}/api/embed with {"model": ..., "input": [texts]}
    returns {"embeddings": [[...], ...]}. Batch the calls; one request per chunk is slow.
    """
    if not texts:
        return []
    vectors: list[list[float]] = []
    with httpx.Client(timeout=120) as client:
        for offset in range(0, len(texts), 64):
            response = client.post(
                f"{settings.ollama_url.rstrip('/')}/api/embed",
                json={"model": settings.embed_model, "input": texts[offset:offset + 64]},
            )
            response.raise_for_status()
            batch = response.json().get("embeddings")
            if not isinstance(batch, list) or len(batch) != len(texts[offset:offset + 64]):
                raise ValueError("embedding server returned an unexpected batch")
            vectors.extend([[float(value) for value in vector] for vector in batch])
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or any(not vector or not all(math.isfinite(value) for value in vector)
                                   for vector in vectors):
        raise ValueError("embedding server returned an invalid vector")
    return vectors
