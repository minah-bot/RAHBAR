"""The real retriever. Yours to build (sections 3.2 and 4.4).

Connection details are already here; the query is not.
"""

from __future__ import annotations

import httpx
import re
from typing import Any

from ..config import Settings
from ..schemas import Passage


class ChromaRetriever:
    name = "chroma"

    def __init__(self, settings: Settings) -> None:
        import chromadb  # the thin client, from chromadb-client in requirements.txt
        from chromadb.config import Settings as ChromaSettings

        self.settings = settings
        self.client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port,
                                          settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = None
        self._versions: tuple[str, ...] | None = None

    @property
    def collection(self):
        # Refresh the handle so a long-lived API process notices a collection rebuilt
        # by the separate indexer Job, without needing an API restart.
        if self._collection is None or hasattr(self, "client"):
            previous = self._collection
            current = self.client.get_collection(self.settings.collection)
            if previous is not None and getattr(previous, "id", None) != getattr(current, "id", None):
                self._versions = None
            self._collection = current
        return self._collection

    def embed(self, text: str) -> list[float]:
        response = httpx.post(f"{self.settings.ollama_url.rstrip('/')}/api/embed",
                              json={"model": self.settings.embed_model, "input": text}, timeout=120)
        response.raise_for_status()
        return [float(value) for value in response.json()["embeddings"][0]]

    def retrieve(self, question: str, k: int) -> list[Passage]:
        if k <= 0:
            return []
        query_vector = self.embed(question)
        collection = self.collection
        indexed_dimension = (collection.metadata or {}).get("dimension")
        if indexed_dimension is not None and len(query_vector) != indexed_dimension:
            raise ValueError(
                f"query embedding dimension {len(query_vector)} does not match index "
                f"dimension {indexed_dimension}"
            )
        count = collection.count()
        if not count:
            return []
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        requested_years = set(years)
        filters: dict[str, object]
        historical_query = bool(requested_years)
        if historical_query:
            versions = [version for version in self._document_versions(collection)
                        if any(year in version for year in requested_years)]
            if not versions:
                return []
            if len(versions) == 1:
                filters = {"doc_version": versions[0]}
            else:
                filters = {"$or": [{"doc_version": version} for version in versions]}
        else:
            filters = {"superseded": False}
        response = collection.query(
            query_embeddings=[query_vector], n_results=min(max(k * 3, k), count),
            where=filters, include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        # Combine semantic rank with exact term overlap. Exact identifiers and numbers
        # matter in university rules; normalize scores to the same [0, 1] range.
        terms = {term for term in re.findall(r"[\w]+(?:[.-][\w]+)*", question.casefold())
                 if len(term) > 1}
        hits: list[tuple[float, Passage]] = []
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            # An explicit year is an intentional request for that historical version.
            # Otherwise the where clause excludes superseded policies.
            if not text or not meta or (not historical_query and meta.get("superseded", False)):
                continue
            version = str(meta.get("doc_version", ""))
            if requested_years and not any(year in version for year in requested_years):
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            body_terms = set(re.findall(r"[\w]+(?:[.-][\w]+)*", text.casefold()))
            lexical = len(terms & body_terms) / max(1, len(terms))
            score = min(1.0, similarity * 0.85 + lexical * 0.15)
            hits.append((score, Passage(chunk_id=str(chunk_id), text=text,
                source=str(meta.get("source", "unknown")), section=str(meta.get("section", "")),
                page=int(meta["page"]) if meta.get("page") is not None else None, score=score)))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [passage for score, passage in hits[:k]
                if score >= self.settings.relevance_floor]

    def _document_versions(self, collection: Any) -> tuple[str, ...]:
        """Read distinct indexed version labels for exact Chroma metadata filters."""
        cached = getattr(self, "_versions", None)
        if cached is not None:
            return cached
        count = collection.count()
        versions: set[str] = set()
        page_size = 1000
        for offset in range(0, count, page_size):
            response = collection.get(
                limit=min(page_size, count - offset), offset=offset, include=["metadatas"]
            )
            for metadata in response.get("metadatas") or []:
                if metadata and metadata.get("doc_version") is not None:
                    versions.add(str(metadata["doc_version"]))
        self._versions = tuple(sorted(versions))
        return self._versions
