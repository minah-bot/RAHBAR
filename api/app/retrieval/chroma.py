"""The real retriever. Yours to build (sections 3.2 and 4.4).

Connection details are already here; the query is not.
"""

from __future__ import annotations

import httpx
import re

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

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_collection(self.settings.collection)
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
        count = self.collection.count()
        if not count:
            return []
        years = re.findall(r"\b(?:19|20)\d{2}\b", question)
        filters: dict[str, object]
        if years:
            filters = {"$and": [{"doc_version": {"$contains": years[0]}}]}
        else:
            filters = {"superseded": False}
        response = self.collection.query(
            query_embeddings=[query_vector], n_results=min(max(k * 3, k), count),
            where=filters, include=["documents", "metadatas", "distances"],
        )
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        # Combine semantic rank with exact term overlap. Exact identifiers and numbers
        # matter in university rules; normalize scores to the same [0, 1] range.
        terms = {term.casefold() for term in question.split() if len(term) > 1}
        hits: list[tuple[float, Passage]] = []
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            if not text or not meta or meta.get("superseded", False):
                continue
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            body_terms = set(text.casefold().split())
            lexical = len(terms & body_terms) / max(1, len(terms))
            score = min(1.0, similarity * 0.85 + lexical * 0.15)
            hits.append((score, Passage(chunk_id=str(chunk_id), text=text,
                source=str(meta.get("source", "unknown")), section=str(meta.get("section", "")),
                page=int(meta["page"]) if meta.get("page") is not None else None, score=score)))
        hits.sort(key=lambda item: item[0], reverse=True)
        return [passage for score, passage in hits[:k]
                if score >= self.settings.relevance_floor]
