"""Stage 5: Store. The only code in the whole system that writes to Chroma.

What happens to the old index while the API is serving from it (ADR 0001): nothing, until the
new one is complete. Each run builds a brand-new collection `<base>-<index_id>`, writes
every chunk, and only then moves the pointer collection `<base>-active` to it. The API follows
the pointer, so pods keep answering from the old index until the swap and never see a
half-built one. Old collections are deleted afterwards, keeping the previous one for rollback.
"""

from __future__ import annotations

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


def store(chunks: list[Chunk], vectors: list[list[float]], settings: IndexerSettings) -> int:
    """Write every chunk with its vector and metadata, and record the embedding model
    and vector dimension in the collection metadata (section 4.4). Return the count."""
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        raise ValueError("refusing to publish an empty index")
    client = get_client(settings)
    index_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    name = f"{settings.collection}-{index_id}"
    collection = client.create_collection(name=name, metadata={
        "hnsw:space": "cosine",
        "embed_model": settings.embed_model,          # the pin the API checks (4.4)
        "dimension": len(vectors[0]),
        "index_id": index_id,
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
