"""Step 7: validate. Every citation must match a chunk retrieved for this request (3.4)."""

from __future__ import annotations

from ..schemas import Answer, Citation, Passage


def sanitize_citations(answer: Answer, passages: list[Passage]) -> tuple[Answer, int]:
    """Drop citations the retriever never returned and return how many were dropped.

    A citation naming a (source, section) that was not retrieved is fabricated. We remove
    it and count it (rahbar_citations_dropped_total) so the hallucination rate stays visible
    even though the user never sees the fake."""
    by_key = {(p.source, p.section): p for p in passages}
    kept: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for c in answer.citations:
        key = (c.source, c.section)
        if key in by_key and key not in seen:
            seen.add(key)
            kept.append(Citation(source=c.source, section=c.section, page=by_key[key].page))
    dropped = len(answer.citations) - len(kept)
    return answer.model_copy(update={"citations": kept}), max(dropped, 0)
