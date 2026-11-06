"""Turns a question into an AskResponse. This is where the query pipeline lives.

The version below is the bare minimum that works with the stubs: retrieve,
generate, return. Section 3.2 lists eight steps; this file does two of them.
"""

from __future__ import annotations

import logging
from typing import Literal

from ..config import Settings
from ..logjson import request_id_var
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..schemas import AskResponse, Citation, RetrievedRef
from .cache import SemanticCache
from ..routes.metrics import ABSTENTIONS, CACHE_HITS, GENERATION_FAILURES

log = logging.getLogger("rahbar.answer")


def answer_question(
    question: str, retriever: Retriever, provider: LLMProvider, settings: Settings
) -> AskResponse:
    cache = SemanticCache(settings, retriever, provider.name)
    cached = cache.lookup(question)
    if cached:
        CACHE_HITS.inc()
        return cached.model_copy(update={"request_id": request_id_var.get(), "cached": True})
    passages = retriever.retrieve(question, k=settings.top_k)
    passages = [p for p in passages if p.score >= settings.relevance_floor]
    if not passages:
        answer = None
        answer_text = "I could not find enough relevant information in the available documents to answer this question."
        citations: list[Citation] = []
        answered = False
        confidence: Literal["high", "medium", "low"] = "low"
        ABSTENTIONS.inc()
    else:
        answer = None
        for retry in range(2):
            try:
                answer = provider.generate(question, passages)
                valid = {(p.source, p.section, p.page) for p in passages}
                if any((c.source, c.section, c.page) not in valid for c in answer.citations):
                    raise ValueError("provider cited a source outside retrieved evidence")
                if answer.answered_from_context and not answer.citations:
                    raise ValueError("context-supported answer must cite evidence")
                break
            except Exception:
                answer = None
                if retry == 0:
                    log.warning("generation validation failed; retrying")
        if answer is None:
            GENERATION_FAILURES.inc()
            answer_text = "I could not produce a validated answer from the retrieved documents."
            citations, answered, confidence = [], False, "low"
            ABSTENTIONS.inc()
        else:
            answer_text = answer.answer
            citations = answer.citations
            answered = answer.answered_from_context
            confidence = answer.confidence
            if not answered:
                citations = []
                ABSTENTIONS.inc()

    log.info(
        "answered",
        extra={
            "retriever": retriever.name,
            "provider": provider.name,
            "top_k": settings.top_k,
            "chunk_ids": [p.chunk_id for p in passages],
            "scores": [round(p.score, 4) for p in passages],
            "answered_from_context": answered,
        },
    )
    response = AskResponse(
        request_id=request_id_var.get(),
        answer=answer_text,
        citations=citations,
        answered_from_context=answered,
        confidence=confidence,
        retrieved=[
            RetrievedRef(
                chunk_id=p.chunk_id, source=p.source, section=p.section, page=p.page, score=p.score
            )
            for p in passages
        ],
        provider=provider.name,
        cached=False,
    )
    cache.save(question, response)
    return response
