"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import hashlib
from typing import Any

from .config import IndexerSettings
from .models import Chunk


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if len(chunks) != len(vectors):
        raise ValueError("each chunk must have exactly one embedding")
    if not chunks:
        raise ValueError("refusing to replace the index with an empty corpus")
    client = get_client(settings)
    # Build a temporary collection first. Swap the collection name only after every
    # record has been written, so failed jobs leave the currently served index intact.
    target = f"{settings.collection}__build"
    try:
        client.delete_collection(target)
    except Exception:
        pass
    fingerprint = hashlib.sha256("\n".join(c.id for c in chunks).encode()).hexdigest()
    metadata = {"hnsw:space": "cosine", "embed_model": settings.embed_model,
                "dimension": len(vectors[0]), "index_fingerprint": fingerprint}
    collection = client.create_collection(target, metadata=metadata)
    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(ids=[c.id for c in batch], embeddings=vectors[start:start + batch_size],
                       documents=[c.text for c in batch], metadatas=[c.metadata() for c in batch])
    try:
        client.delete_collection(settings.collection)
    except Exception:
        pass
    active = client.get_or_create_collection(settings.collection, metadata=metadata)
    staged = client.get_collection(target)
    records = staged.get(include=["documents", "metadatas", "embeddings"])
    for start in range(0, len(records["ids"]), batch_size):
        end = start + batch_size
        embeddings = records["embeddings"][start:end]
        active.add(ids=records["ids"][start:end],
                   embeddings=[list(vector) for vector in embeddings],
                   documents=records["documents"][start:end],
                   metadatas=records["metadatas"][start:end])
    client.delete_collection(target)
    return len(chunks)
