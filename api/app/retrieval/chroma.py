"""The real retriever (sections 3.2 and 4.4). Reads the index; never writes it."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

from ..config import Settings
from ..providers.embedding import EmbeddingError, OllamaEmbedder
from ..schemas import Passage
from .index import POINTER_SUFFIX, IndexInfo, IndexNotReady, pin_conflicts

log = logging.getLogger("rahbar.retrieval")

_WORDS = re.compile(r"\w+(?:[.-]\w+)*")
_YEARS = re.compile(r"\b(?:19|20)\d{2}\b")


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings, embedder: OllamaEmbedder | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or OllamaEmbedder(settings)
        # No connection here. If Chroma is down when the pod starts, the process must still boot:
        # /healthz stays 200 (liveness never depends on Chroma) and /readyz answers 503 naming it.
        self.client: Any = None
        self._versions: tuple[str, tuple[str, ...]] | None = None   # (index_id, doc_versions)
        self._info: IndexInfo | None = None
        self._info_at = 0.0
        self._lock = threading.Lock()

    def _get_client(self) -> Any:
        """Connect on first use; the chromadb client connects (and raises) when constructed."""
        if self.client is None:
            import chromadb  # the thin client, from chromadb-client in requirements.txt
            from chromadb.config import Settings as ChromaSettings

            self.client = chromadb.HttpClient(
                host=self.settings.chroma_host, port=self.settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False))
        return self.client

    # -- what is the index? -------------------------------------------------------
    def _resolve(self) -> IndexInfo:
        base = self.settings.collection
        try:
            name = base
            try:
                pointer = self._get_client().get_collection(base + POINTER_SUFFIX)
                got = pointer.get(ids=["active"])
                if got["documents"]:
                    name = got["documents"][0]
            except Exception:  # no pointer yet (dummy index): read the base collection
                pass
            col = self._get_client().get_collection(name)
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

        where = self._where(question, info, include_superseded)
        if where == {}:                 # an explicit year with no document of that year
            return []
        try:
            col = self._get_client().get_collection(info.collection)
            query_embeddings: Any = [vector]
            res = col.query(query_embeddings=query_embeddings,
                            n_results=max(1, min(max(k * 3, k), info.count)), where=where,
                            include=["documents", "metadatas", "distances"])
        except Exception as exc:
            raise IndexNotReady("chroma", f"chroma query failed: {exc!r}") from exc

        # Exact identifiers and numbers ("5.4", "PKR 3,000") blur in embeddings, so blend the
        # semantic score with the share of the question's terms found verbatim in the chunk.
        weight = min(max(self.settings.lexical_weight, 0.0), 1.0)
        terms = set(_WORDS.findall(question.casefold()))
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ranked: list[Passage] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            meta = meta or {}
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))      # Chroma returns cosine distance
            lexical = len(terms & set(_WORDS.findall((text or "").casefold()))) / max(1, len(terms))
            page = meta.get("page")
            ranked.append(Passage(
                chunk_id=chunk_id, text=text or "", source=str(meta.get("source", "")),
                section=str(meta.get("section", "")),
                page=int(page) if isinstance(page, int | float) else None,
                score=round(min(1.0, similarity * (1.0 - weight) + lexical * weight), 4),
                doc_version=str(meta.get("doc_version", "")),
                superseded=bool(meta.get("superseded", False)),
            ))
        ranked.sort(key=lambda p: p.score, reverse=True)
        passages = ranked[:k]
        return passages

    def _doc_versions(self, info: IndexInfo) -> tuple[str, ...]:
        """Distinct doc_version labels in the served index, cached per index."""
        if self._versions is not None and self._versions[0] == info.index_id:
            return self._versions[1]
        col = self._get_client().get_collection(info.collection)
        found: set[str] = set()
        for offset in range(0, info.count, 1000):
            rows = col.get(limit=min(1000, info.count - offset), offset=offset,
                           include=["metadatas"])
            found.update(str(m["doc_version"]) for m in rows.get("metadatas") or []
                         if m and m.get("doc_version") is not None)
        self._versions = (str(info.index_id), tuple(sorted(found)))
        return self._versions[1]

    def _where(self, question: str, info: IndexInfo,
               include_superseded: bool) -> dict[str, Any] | None:
        """The metadata filter, applied inside the Chroma query. Naming a year asks for the
        document of that year (current or superseded); otherwise superseded documents are
        excluded unless the caller says the user asked about an older version. `{}` means the
        question names a year that no indexed document has."""
        years = set(_YEARS.findall(question))
        if years:
            matched = [v for v in self._doc_versions(info) if any(y in v for y in years)]
            if not matched:
                return {}
            return {"doc_version": matched[0]} if len(matched) == 1 else {
                "doc_version": {"$in": matched}}
        return None if include_superseded else {"superseded": False}
