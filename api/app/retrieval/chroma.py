"""The real retriever (sections 3.2 and 4.4). Reads the index; never writes it."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from ..config import Settings
from ..providers.embedding import EmbeddingError, OllamaEmbedder
from ..schemas import Passage
from .index import POINTER_SUFFIX, IndexInfo, IndexNotReady, pin_conflicts

log = logging.getLogger("rahbar.retrieval")


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, embedder: OllamaEmbedder | None = None) -> None:
        import chromadb  # the thin client, from chromadb-client in requirements.txt
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.embedder = embedder or OllamaEmbedder(settings)
        self.client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                                          settings=ChromaSettings(anonymized_telemetry=False))
        self._info: IndexInfo | None = None
        self._info_at = 0.0
        self._lock = threading.Lock()

    # -- what is the index? -------------------------------------------------------
    def _resolve(self) -> IndexInfo:
        base = self.settings.collection
        try:
            name = base
            try:
                pointer = self.client.get_collection(base + POINTER_SUFFIX)
                got = pointer.get(ids=["active"])
                if got["documents"]:
                    name = got["documents"][0]
            except Exception:  # no pointer yet (dummy index): read the base collection
                pass
            col = self.client.get_collection(name)
            meta = col.metadata or {}
            dim = meta.get("dimension")
            return IndexInfo(
                collection=name,
                count=col.count(),
                embed_model=meta.get("embed_model"),
                dimension=int(dim) if dim is not None else None,
                index_id=str(meta["index_id"]) if "index_id" in meta else name,
            )
        except IndexNotReady:
            raise
        except Exception as exc:
            text = repr(exc)
            if "does not exist" in text or "NotFound" in text or "not found" in text.lower():
                raise IndexNotReady(
                    "collection", f"collection {base!r} does not exist: has the indexer run?"
                ) from exc
            raise IndexNotReady("chroma", f"chroma unreachable: {text}") from exc

    def index_info(self, force: bool = False) -> IndexInfo:
        with self._lock:
            fresh = time.monotonic() - self._info_at < self.settings.index_ttl_seconds
            if self._info is not None and fresh and not force:
                return self._info
        info = self._resolve()
        with self._lock:
            self._info, self._info_at = info, time.monotonic()
        return info

    def index_id(self) -> str | None:
        return self.index_info().index_id

    def check_ready(self) -> dict[str, object]:
        """Readiness: Chroma reachable, collection non-empty, embedding pin holds."""
        info = self.index_info(force=True)
        if info.count == 0:
            raise IndexNotReady("collection", f"collection {info.collection!r} is empty")
        problems = pin_conflicts(info, self.settings.embed_model)
        if problems:
            for problem in problems:
                log.error("embedding pin violated", extra={
                    "conflict": problem, "index_embed_model": info.embed_model,
                    "api_embed_model": self.settings.embed_model, "collection": info.collection})
            raise IndexNotReady("embedding-pin", "; ".join(problems))
        return {"collection": info.collection, "chunks": info.count,
                "embed_model": info.embed_model, "index_id": info.index_id}

    # -- the query -----------------------------------------------------------------
    def retrieve(
        self, question: str, k: int, *, include_superseded: bool = False
    ) -> list[Passage]:
        info = self.index_info()
        problems = pin_conflicts(info, self.settings.embed_model)
        if problems:  # refuse to serve vectors from a different model, even between probes
            raise IndexNotReady("embedding-pin", "; ".join(problems))
        try:
            vector = self.embedder.embed_query(question)
        except EmbeddingError as exc:
            raise IndexNotReady("embedder", str(exc)) from exc
        problems = pin_conflicts(info, self.settings.embed_model, len(vector))
        if problems:
            raise IndexNotReady("embedding-pin", "; ".join(problems))

        where: dict[str, Any] | None = None if include_superseded else {"superseded": False}
        try:
            col = self.client.get_collection(info.collection)
            query_embeddings: Any = [vector]
            res = col.query(query_embeddings=query_embeddings, n_results=k, where=where,
                            include=["documents", "metadatas", "distances"])
        except Exception as exc:
            raise IndexNotReady("chroma", f"chroma query failed: {exc!r}") from exc

        passages: list[Passage] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            meta = meta or {}
            page = meta.get("page")
            passages.append(Passage(
                chunk_id=chunk_id, text=text or "", source=str(meta.get("source", "")),
                section=str(meta.get("section", "")),
                page=int(page) if isinstance(page, int | float) else None,
                score=round(1.0 - float(dist), 4),   # Chroma returns cosine distance
                doc_version=str(meta.get("doc_version", "")),
                superseded=bool(meta.get("superseded", False)),
            ))
        return passages
