"""What the API knows about the index it reads, and the embedding-pin check (section 4.4).

The indexer builds each index in a fresh collection `<base>-<index_id>` and then moves a
pointer (collection `<base>-active`) to it. The API follows the pointer, so a re-index
swaps the index with no downtime (ADR 0001). If no pointer exists it reads `<base>` itself,
which is what the week-1 dummy index writes.
"""

from __future__ import annotations

from dataclasses import dataclass

POINTER_SUFFIX = "-active"


class IndexNotReady(RuntimeError):
    """Raised when the API must not serve: names the dependency that failed."""

    def __init__(self, dependency: str, detail: str) -> None:
        super().__init__(detail)
        self.dependency = dependency
        self.detail = detail


@dataclass(frozen=True)
class IndexInfo:
    collection: str
    count: int
    embed_model: str | None
    dimension: int | None
    index_id: str | None


def pin_conflicts(info: IndexInfo, embed_model: str, query_dimension: int | None = None) -> list[str]:
    """Return one sentence per conflict between the index and this process's EMBED_MODEL.
    An empty list means the pin holds."""
    problems: list[str] = []
    if info.embed_model is None:
        problems.append(
            f"collection {info.collection!r} records no embed_model: "
            "it was not built by the RAHBAR indexer"
        )
    elif info.embed_model != embed_model:
        problems.append(
            f"embedding model mismatch: the index was built with {info.embed_model!r} "
            f"but this API is configured with EMBED_MODEL={embed_model!r}"
        )
    if (
        query_dimension is not None
        and info.dimension is not None
        and query_dimension != info.dimension
    ):
        problems.append(
            f"embedding dimension mismatch: index vectors have {info.dimension} dimensions, "
            f"the API's embedding model returns {query_dimension}"
        )
    return problems
