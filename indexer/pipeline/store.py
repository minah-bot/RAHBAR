"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .config import IndexerSettings
from .models import Chunk


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def reuse_embeddings(chunks: list[Chunk], settings: IndexerSettings) -> dict[str, list[float]]:
    """Reuse vectors only when the active collection uses the pinned model.

    Chunk IDs include document identity, content, and version metadata. That makes an ID
    match sufficient to reuse its vector safely, while changed chunks are embedded again.
    """
    if not chunks:
        return {}
    client = get_client(settings)
    collections = client.list_collections()
    names = {item if isinstance(item, str) else item.name for item in collections}
    if settings.collection not in names:
        return {}
    collection = client.get_collection(settings.collection)
    metadata = collection.metadata or {}
    if metadata.get("embed_model") != settings.embed_model:
        return {}
    expected_dimension = metadata.get("dimension")
    chunk_ids = [chunk.id for chunk in chunks]
    result: dict[str, list[float]] = {}
    for offset in range(0, len(chunk_ids), 500):
        rows = collection.get(ids=chunk_ids[offset:offset + 500], include=["embeddings"])
        ids = rows.get("ids", [])
        embeddings = rows.get("embeddings")
        if embeddings is None:
            continue
        for chunk_id, raw_vector in zip(ids, embeddings):
            vector = [float(value) for value in raw_vector]
            if (not vector or (expected_dimension and len(vector) != expected_dimension)
                    or not all(math.isfinite(value) for value in vector)):
                continue
            result[str(chunk_id)] = vector
    return result


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if len(chunks) != len(vectors):
        raise ValueError("each chunk must have exactly one embedding")
    if not chunks:
        raise ValueError("refusing to replace the index with an empty corpus")
    unique: dict[str, Chunk] = {}
    unique_vectors: dict[str, list[float]] = {}
    for chunk, vector in zip(chunks, vectors):
        previous = unique.get(chunk.id)
        if previous is not None and (previous != chunk or unique_vectors[chunk.id] != vector):
            raise ValueError(f"conflicting chunks share id {chunk.id!r}")
        unique[chunk.id] = chunk
        unique_vectors[chunk.id] = vector
    chunks = list(unique.values())
    vectors = [unique_vectors[chunk.id] for chunk in chunks]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) == 0:
        raise ValueError("all chunks must have non-empty vectors with one shared dimension")
    if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
        raise ValueError("chunk embeddings must contain only finite values")
    client = get_client(settings)
    # Build a temporary collection first. Swap the collection name only after every
    # record has been written, so failed jobs leave the currently served index intact.
    target = f"{settings.collection}__build"
    try:
        client.delete_collection(target)
    except Exception:
        pass
    fingerprint_rows = [
        f"{c.id}\0{c.source}\0{c.section}\0{c.page}\0{c.doc_version}\0{c.superseded}"
        for c in chunks
    ]
    fingerprint = hashlib.sha256("\n".join(sorted(fingerprint_rows)).encode()).hexdigest()
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
