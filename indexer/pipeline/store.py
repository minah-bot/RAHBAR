"""Stage 5: Store. The only code in the whole system that writes to Chroma.

What happens to the old index while the API is serving from it (ADR 0001): nothing, until the
new one is complete. Each run builds a brand-new collection `<base>-<index_id>`, writes
every chunk, and only then moves the pointer collection `<base>-active` to it. The API follows
the pointer, so pods keep answering from the old index until the swap and never see a
half-built one. Old collections are deleted afterwards, keeping the previous one for rollback.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

from .config import IndexerSettings
from .models import Chunk

POINTER_SUFFIX = "-active"
WRITE_BATCH = 128


def get_client(settings: IndexerSettings) -> Any:
    import chromadb  # the thin client, from chromadb-client in requirements.txt
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                               settings=ChromaSettings(anonymized_telemetry=False))


def _active_collection_name(client: Any, settings: IndexerSettings) -> str | None:
    """The collection the API is serving: follow the pointer, else the plain collection."""
    names = {c if isinstance(c, str) else c.name for c in client.list_collections()}
    pointer = settings.collection + POINTER_SUFFIX
    if pointer in names:
        got = client.get_collection(pointer).get(ids=["active"])
        if got["documents"] and got["documents"][0] in names:
            return str(got["documents"][0])
    return settings.collection if settings.collection in names else None


def reuse_embeddings(chunks: list[Chunk], settings: IndexerSettings) -> dict[str, list[float]]:
    """Vectors of chunks already in the served index, but only if it was built with the same
    embedding model. Chunk ids cover document identity, version and content (chunk.py), so an id
    match means the same text: it is safe to reuse its vector, and changed chunks are re-embedded."""
    if not chunks:
        return {}
    client = get_client(settings)
    name = _active_collection_name(client, settings)
    if name is None:
        return {}
    collection = client.get_collection(name)
    metadata = collection.metadata or {}
    if metadata.get("embed_model") != settings.embed_model:
        return {}
    expected = metadata.get("dimension")
    ids = [c.id for c in chunks]
    reused: dict[str, list[float]] = {}
    for offset in range(0, len(ids), 500):
        rows = collection.get(ids=ids[offset:offset + 500], include=["embeddings"])
        embeddings = rows.get("embeddings")
        if embeddings is None:
            continue
        for chunk_id, raw in zip(rows.get("ids", []), embeddings, strict=False):
            vector = [float(v) for v in raw]
            if (not vector or (expected and len(vector) != expected)
                    or not all(math.isfinite(v) for v in vector)):
                continue
            reused[str(chunk_id)] = vector
    return reused


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        raise ValueError("refusing to publish an empty index")
    seen: dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        if chunk.id in seen:
            raise ValueError(f"conflicting chunks share id {chunk.id!r}")
        seen[chunk.id] = position
    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("all chunks must have non-empty vectors with one shared dimension")
    if any(not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("chunk embeddings must contain only finite values")
    fingerprint = hashlib.sha256("\n".join(sorted(
        f"{c.id}\0{c.source}\0{c.section}\0{c.page}\0{c.doc_version}\0{c.superseded}"
        for c in chunks)).encode()).hexdigest()
    client = get_client(settings)
    index_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    name = f"{settings.collection}-{index_id}"
    collection = client.create_collection(name=name, metadata={
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,          # the pin the API checks (4.4)
        "dimension": len(vectors[0]),
        "index_id": index_id,
        "index_fingerprint": fingerprint,     # same corpus and chunking give the same value
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunks": len(chunks),
    })
    for i in range(0, len(chunks), WRITE_BATCH):
        part = chunks[i:i + WRITE_BATCH]
        collection.add(
            ids=[c.id for c in part],
            documents=[c.text for c in part],
            embeddings=vectors[i:i + WRITE_BATCH],
            metadatas=[c.metadata() for c in part],
        )
    if collection.count() != len(chunks):
        raise RuntimeError("collection count differs from chunks written; not publishing")
    _swap_pointer(client, settings, name)
    _delete_old(client, settings, keep={name})
    return int(collection.count())


def _swap_pointer(client: Any, settings: IndexerSettings, name: str) -> None:
    pointer = client.get_or_create_collection(name=settings.collection + POINTER_SUFFIX)
    previous = pointer.get(ids=["active"])
    old = previous["documents"][0] if previous["documents"] else None
    pointer.upsert(ids=["active"], documents=[name], embeddings=[[0.0, 1.0]],
                   metadatas=[{"previous": old or ""}])


def _delete_old(client: Any, settings: IndexerSettings, keep: set[str]) -> None:
    pointer = client.get_collection(settings.collection + POINTER_SUFFIX)
    previous = (pointer.get(ids=["active"])["metadatas"] or [{}])[0].get("previous", "")
    if previous:
        keep = keep | {previous}                       # one generation kept for rollback
    prefix = settings.collection + "-"
    for col in client.list_collections():
        col_name = col if isinstance(col, str) else col.name
        if col_name.startswith(prefix) and col_name != settings.collection + POINTER_SUFFIX \
                and col_name not in keep:
            client.delete_collection(col_name)
