"""Turns a question into an AskResponse: the query pipeline (section 3.2).

Receive and validate happen in the route (pydantic). This module does the other steps:
cache check, retrieve, abstain below the relevance floor, generate, validate with one retry
and a safe fallback, check citations, store. Each step is a small function so it can be
unit-tested with a stub retriever and a stub provider.
"""

from __future__ import annotations

import inspect
import logging
import time


from ..config import Settings
from ..logjson import request_id_var
from ..providers.embedding import EmbeddingError
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..retrieval.index import IndexNotReady
from ..retrieval.version import wants_older_version
from ..routes import metrics
from ..schemas import Answer, AskResponse, Passage, RetrievedRef
from .cache import SemanticCache
from .validate import sanitize_citations

log = logging.getLogger("rahbar.answer")

ABSTAIN_TEXT = "The documents I have do not cover this question."
FALLBACK_TEXT = (
    "I could not produce a validated answer from the retrieved documents. Please try again "
    "or rephrase your question."
)


def _refs(passages: list[Passage]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                         page=p.page, score=p.score) for p in passages]


def _response(answer: Answer, passages: list[Passage], provider: LLMProvider) -> AskResponse:
    return AskResponse(
        request_id=request_id_var.get(), answer=answer.answer, citations=answer.citations,
        answered_from_context=answer.answered_from_context, confidence=answer.confidence,
        retrieved=_refs(passages), provider=provider.name,
    )


def generate_validated(
    question: str, passages: list[Passage], provider: LLMProvider
) -> Answer | None:
    """Step 6 and 7: generate, parse, validate; retry once; None means use the fallback.

    The model's structure is enforced by the provider (format= / JSON mode) and then
    validated here with pydantic; structure can be guaranteed, truth cannot."""
    for attempt in (1, 2):
        try:
            answer = provider.generate(question, passages)
        except Exception as exc:   # any provider failure: retry once, then the safe fallback, never a 500
            metrics.VALIDATION_FAILURES.inc()
            log.warning("model output rejected", extra={"attempt": attempt, "error": repr(exc)[:300]})
            continue
        answer, dropped = sanitize_citations(answer, passages)
        if dropped:
            metrics.CITATIONS_DROPPED.inc(dropped)
            log.warning("fabricated citations dropped", extra={"dropped": dropped})
        if answer.answered_from_context and not answer.answer.strip():
            metrics.VALIDATION_FAILURES.inc()   # an "answer" with no text is a failed generation
            log.warning("answer was empty", extra={"attempt": attempt})
            continue
        if answer.answered_from_context and not answer.citations:
            metrics.VALIDATION_FAILURES.inc()   # a claim with no valid source is not an answer
            log.warning("answer had no valid citation", extra={"attempt": attempt})
            continue
        return answer
    return None


def _retrieve(retriever: Retriever, question: str, k: int, older: bool) -> list[Passage]:
    """Call retrieve(), passing include_superseded only if the retriever accepts it, so a plain
    `retrieve(question, k)` implementation (the starter's original interface) still works."""
    if "include_superseded" in inspect.signature(retriever.retrieve).parameters:
        return retriever.retrieve(question, k=k, include_superseded=older)
    return retriever.retrieve(question, k=k)


def answer_question(
    question: str,
    retriever: Retriever,
    provider: LLMProvider,
    settings: Settings,
    cache: SemanticCache | None = None,
    follow_up: bool = False,
) -> AskResponse:
    older = wants_older_version(question)
    # Never cache: follow-ups, older-version questions (their answer depends on the filter
    # the question text barely changes), abstentions, fallbacks, anything that failed validation.
    never_cache = follow_up or older
    use_cache = cache is not None and not never_cache
    index_id: str | None = None
    vector: list[float] | None = None

    if use_cache and cache is not None:
        try:
            index_id = getattr(retriever, "index_id", lambda: None)()
            embedder = getattr(retriever, "embedder", None)
            if index_id and embedder is not None:
                query_vec: list[float] = embedder.embed_query(question)
                vector = query_vec
                hit, similarity = cache.lookup(query_vec, index_id, question)
                if hit is not None:
                    metrics.CACHE_LOOKUPS.labels(result="hit").inc()
                    log.info("answered", extra={"cached": True, "similarity": round(similarity, 4),
                                                "abstained": False, "top_k": settings.top_k})
                    return hit.model_copy(update={"request_id": request_id_var.get(),
                                                  "cached": True})
                metrics.CACHE_LOOKUPS.labels(result="miss").inc()
            else:
                use_cache = False
        except (IndexNotReady, EmbeddingError):
            use_cache = False   # retrieval below raises the proper 503
        except Exception as exc:  # Redis down must not take answering down with it
            metrics.CACHE_LOOKUPS.labels(result="error").inc()
            log.warning("cache unavailable", extra={"error": repr(exc)[:200]})
            use_cache = False

    started = time.perf_counter()
    passages = _retrieve(retriever, question, settings.top_k, older)
    metrics.RETRIEVAL_SECONDS.observe(time.perf_counter() - started)

    # Only passages above the floor reach the model: weak neighbours are noise it could cite.
    grounded = [p for p in passages if p.score >= settings.relevance_floor]
    if not grounded:
        metrics.ABSTENTIONS.labels(reason="below_relevance_floor").inc()
        response = AskResponse(
            request_id=request_id_var.get(), answer=ABSTAIN_TEXT, citations=[],
            answered_from_context=False, confidence="low", retrieved=_refs(passages),
            provider=provider.name)
        _log(settings, retriever, provider, passages, response, "below_relevance_floor")
        return response

    started = time.perf_counter()
    answer = generate_validated(question, grounded, provider)
    metrics.GENERATION_SECONDS.observe(time.perf_counter() - started)

    if answer is None:
        metrics.FALLBACKS.inc()
        response = _response(Answer(reasoning="fallback", answer=FALLBACK_TEXT, citations=[],
                                    answered_from_context=False, confidence="low"),
                             passages, provider)
        _log(settings, retriever, provider, passages, response, "fallback")
        return response

    if not answer.answered_from_context:
        metrics.ABSTENTIONS.labels(reason="model_declined").inc()
        answer = answer.model_copy(update={"citations": [], "confidence": "low"})
        response = _response(answer, passages, provider)
        _log(settings, retriever, provider, passages, response, "model_declined")
        return response

    response = _response(answer, passages, provider)
    if use_cache and cache is not None and index_id and vector is not None:
        try:
            cache.store(vector, index_id, response, question)   # only validated, grounded answers get here
        except Exception as exc:
            log.warning("cache store failed", extra={"error": repr(exc)[:200]})
    _log(settings, retriever, provider, passages, response, None)
    return response


def _log(settings: Settings, retriever: Retriever, provider: LLMProvider,
         passages: list[Passage], response: AskResponse, reason: str | None) -> None:
    log.info("answered", extra={
        "retriever": retriever.name, "provider": provider.name, "top_k": settings.top_k,
        "chunk_ids": [p.chunk_id for p in passages],
        "scores": [round(p.score, 4) for p in passages],
        "answered_from_context": response.answered_from_context,
        "abstained": not response.answered_from_context, "reason": reason, "cached": False,
    })
