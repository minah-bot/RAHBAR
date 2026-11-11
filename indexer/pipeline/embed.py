"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import time

import httpx

from .config import IndexerSettings

BATCH = 16


def prepare(model: str, text: str, kind: str) -> str:
    """Same function as api/app/providers/embedding.py. nomic-embed-text expects a task prefix."""
    if model.startswith("nomic-embed-text"):
        return f"search_{kind}: {text}"
    return text


def embed(texts: list[str], settings: IndexerSettings) -> list[list[float]]:
    """Embed with settings.embed_model and nothing else, in batches, with retries."""
    vectors: list[list[float]] = []
    with httpx.Client(base_url=settings.ollama_url,
                      timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        for start in range(0, len(texts), BATCH):
            batch = [prepare(settings.embed_model, t, "document")
                     for t in texts[start:start + BATCH]]
            for attempt in range(1, 4):
                try:
                    r = client.post("/api/embed", json={"model": settings.embed_model,
                                                        "input": batch})
                    r.raise_for_status()
                    got = r.json()["embeddings"]
                    if len(got) != len(batch):
                        raise ValueError(f"asked for {len(batch)} vectors, got {len(got)}")
                    vectors.extend(got)
                    break
                except (httpx.HTTPError, ValueError, KeyError):
                    if attempt == 3:
                        raise
                    time.sleep(2 * attempt)
    return vectors
