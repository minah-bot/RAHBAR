"""The retriever interface. The rest of the API only ever sees this."""

from __future__ import annotations

from typing import Protocol

from ..schemas import Passage


class Retriever(Protocol):
    name: str

    def retrieve(
        self, question: str, k: int, *, include_superseded: bool = False
    ) -> list[Passage]:
        """Return up to k passages, best first. Superseded documents are excluded
        unless the caller says the user asked about an older version."""
        ...

    def index_id(self) -> str | None:
        """Identifier of the index being served (None for the stub). Used to scope the cache."""
        ...

    def check_ready(self) -> dict[str, object]:
        """Raise IndexNotReady, naming the dependency, unless the index can serve:
        Chroma reachable, the collection exists with count() > 0, the embedding pin holds."""
        ...
