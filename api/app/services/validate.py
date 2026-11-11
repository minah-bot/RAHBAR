"""Step 7: validate. Every citation must match a chunk retrieved for this request (3.4)."""

from __future__ import annotations

from ..schemas import Citation, Passage, PassageCitation


def sanitize_citations(citations: list[PassageCitation], passages: list[Passage]) -> tuple[list[Citation], int]:
    """Resolve model passage IDs to citation metadata from this request's retrieval only."""
    by_id = {p.chunk_id: p for p in passages}
    kept: list[Citation] = []
    seen: set[str] = set()
    for citation in citations:
        passage = by_id.get(citation.passage_id)
        if passage is not None and citation.passage_id not in seen:
            seen.add(citation.passage_id)
            kept.append(Citation(source=passage.source, section=passage.section, page=passage.page))
    dropped = len(citations) - len(kept)
    return kept, max(dropped, 0)
