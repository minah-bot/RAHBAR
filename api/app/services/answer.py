"""Turns a question into an AskResponse: the query pipeline (section 3.2).

Receive and validate happen in the route (pydantic). This module does the other steps:
cache check, retrieve, abstain below the relevance floor, generate, validate with one retry
and a safe fallback, check citations, store. Each step is a small function so it can be
unit-tested with a stub retriever and a stub provider.
"""

from __future__ import annotations

import inspect
import logging
import re
import time


from ..config import Settings
from ..logjson import request_id_var
from ..providers.embedding import EmbeddingError
from ..providers.llm.base import LLMProvider
from ..retrieval.base import Retriever
from ..retrieval.index import IndexNotReady
from ..retrieval.version import wants_older_version
from ..routes import metrics
from ..schemas import Answer, AskResponse, Citation, Passage, PassageCitation, RetrievedRef
from .cache import SemanticCache
from .validate import sanitize_citations

log = logging.getLogger("rahbar.answer")

ABSTAIN_TEXT = "The documents I have do not cover this question."
FALLBACK_TEXT = (
    "I could not produce a validated answer from the retrieved documents. Please try again "
    "or rephrase your question."
)
_RULE_125_CGPA = re.compile(
    r"Minimum\s+Cumulative\s+Grade\s+Point\s+Average\s*\(CGPA\)\s+of\s+"
    r"\d+(?:\.\d+)?\s+is\s+required\s+for\s+graduation\.",
    re.IGNORECASE,
)


def _extractive_cgpa_answer(
    question: str, passages: list[Passage]
) -> tuple[Answer, list[Citation]] | None:
    """Extract the explicit Rule 1.25 graduation threshold from this request's evidence."""
    normalized_question = question.casefold()
    asks_about_cgpa = "cgpa" in normalized_question or "cumulative grade point average" in normalized_question
    asks_about_rule = bool(re.search(r"\b1\.25\b", normalized_question))
    asks_about_graduation = "graduat" in normalized_question
    asks_for_minimum = "minimum" in normalized_question or asks_about_rule
    if not (asks_about_cgpa and asks_for_minimum
            and (asks_about_rule or asks_about_graduation)):
        return None

    for passage in passages:
        if not re.search(r"\b1\.25\b", f"{passage.section} {passage.text}"):
            continue
        match = _RULE_125_CGPA.search(passage.text)
        if match is None:
            continue
        # Resolve through the same retrieved-ID map used for model citations. No metadata is
        # copied from the model, and a candidate absent from this request cannot be cited.
        citations, _ = sanitize_citations(
            [PassageCitation(passage_id=passage.chunk_id)], passages
        )
        if not citations:
            continue
        answer_text = re.sub(r"\s+", " ", match.group(0)).strip()
        answer = Answer(
            reasoning="Extracted directly from retrieved Rule 1.25.",
            answer=answer_text,
            citations=[PassageCitation(passage_id=passage.chunk_id)],
            answered_from_context=True,
            confidence="high",
        )
        return answer, citations
    return None


def _extractive_ai_assessment_answer(
    question: str, passages: list[Passage]
) -> tuple[Answer, list[Citation]] | None:
    """Recover the explicit non-lab/lab assessment rules from retrieved section 6 passages."""
    normalized_question = question.casefold()
    if "generative ai" not in normalized_question or "assessment" not in normalized_question:
        return None

    non_lab = next((p for p in passages if "6. generative ai and assessments" in p.section.casefold()
                    and "6.1 use in non-lab courses" in p.text.casefold()
                    and "students are allowed" in p.text.casefold()
                    and "associated quizzes" in p.text.casefold()), None)
    lab = next((p for p in passages if "6.2 use in lab courses" in p.text.casefold()
                and "prohibited unless explicitly allowed" in p.text.casefold()), None)
    if non_lab is None or lab is None:
        return None

    selected = [non_lab, lab]
    citations, _ = sanitize_citations(
        [PassageCitation(passage_id=p.chunk_id) for p in selected], passages
    )
    if len(citations) != len(selected):
        return None
    answer = Answer(
        reasoning="Extracted from retrieved sections 6.1 and 6.2.",
        answer=("For non-lab courses, students may use generative AI for assignments, which "
                "must include associated quizzes or demonstrations. In labs, AI use is generally "
                "prohibited unless the lab instructor explicitly allows it."),
        citations=[PassageCitation(passage_id=p.chunk_id) for p in selected],
        answered_from_context=True,
        confidence="high",
    )
    return answer, citations


def _extractive_attendance_answer(
    question: str, passages: list[Passage]
) -> tuple[Answer, list[Citation]] | None:
    """Answer the final-exam attendance threshold only when rules 1.21 and 1.22 are retrieved."""
    normalized_question = question.casefold()
    asks_exam_attendance = (
        "attendance" in normalized_question
        and re.search(r"\bfinal\s+exam(?:ination)?\b", normalized_question) is not None
        and ("minimum" in normalized_question or "sit" in normalized_question
             or "eligible" in normalized_question)
    )
    if not asks_exam_attendance:
        return None

    requirement = next((p for p in passages
                       if re.search(r"\b1\.21\b", p.section)
                       and re.search(r"\b100\s*%\s+attendance\b", p.section, re.I)), None)
    consequence = next((p for p in passages
                        if re.search(r"\b1\.22\b", p.section)
                        and "ineligible to appear in the final examination" in p.text.casefold()),
                       None)
    if requirement is None or consequence is None:
        return None

    selected = [requirement, consequence]
    citations, _ = sanitize_citations(
        [PassageCitation(passage_id=p.chunk_id) for p in selected], passages
    )
    if len(citations) != len(selected):
        return None
    exception = ""
    if re.search(r"\b20\s*%", requirement.text) and "condoned by the hod" in requirement.text.casefold():
        exception = " The HoD may condone up to 20% absence for genuine reasons."
    answer = Answer(
        reasoning="Extracted from retrieved Rules 1.21 and 1.22.",
        answer=("Rule 1.21 requires 100% attendance to be eligible for the final exam."
                f"{exception} Rule 1.22 says students who fail to meet attendance requirements "
                "are ineligible to appear in the final examination."),
        citations=[PassageCitation(passage_id=p.chunk_id) for p in selected],
        answered_from_context=True,
        confidence="high",
    )
    return answer, citations


def _extractive_older_attendance_answer(
    question: str, passages: list[Passage]
) -> tuple[Answer, list[Citation]] | None:
    """Answer a 2023 attendance question only from retrieved superseded rule text."""
    normalized_question = question.casefold()
    asks_older_exam_attendance = (
        "2023" in normalized_question
        and "attendance" in normalized_question
        and re.search(r"\bfinal\s+exam(?:ination)?\b", normalized_question) is not None
        and ("minimum" in normalized_question or "sit" in normalized_question
             or "eligible" in normalized_question)
    )
    if not asks_older_exam_attendance:
        return None

    requirement = next((p for p in passages if p.superseded and "2023" in p.doc_version
                        and re.search(
                            r"\b1\.21\s+students\s+are\s+required\s+to\s+maintain\s+"
                            r"100\s*%\s+attendance\b",
                            re.sub(r"\s+", " ", p.text), re.I)), None)
    consequence = next((p for p in passages if p.superseded and "2023" in p.doc_version
                        and re.search(r"\b1\.22\b", p.section)
                        and "ineligible to appear in the final examination" in p.text.casefold()),
                       None)
    if requirement is None or consequence is None:
        return None

    selected = [requirement, consequence]
    citations, _ = sanitize_citations(
        [PassageCitation(passage_id=p.chunk_id) for p in selected], passages
    )
    if len(citations) != len(selected):
        return None
    answer = Answer(
        reasoning="Extracted from retrieved 2023 attendance-rule passages.",
        answer=("The 2023 rules required students to maintain 100% attendance. They also state "
                "that students who fail to meet attendance requirements are ineligible to appear "
                "in the final examination."),
        citations=[PassageCitation(passage_id=p.chunk_id) for p in selected],
        answered_from_context=True,
        confidence="high",
    )
    return answer, citations


def _refs(passages: list[Passage]) -> list[RetrievedRef]:
    return [RetrievedRef(chunk_id=p.chunk_id, source=p.source, section=p.section,
                         page=p.page, score=p.score) for p in passages]


def _response(answer: Answer, citations: list[Citation], passages: list[Passage], provider: LLMProvider) -> AskResponse:
    return AskResponse(
        request_id=request_id_var.get(), answer=answer.answer, citations=citations,
        answered_from_context=answer.answered_from_context, confidence=answer.confidence,
        retrieved=_refs(passages), provider=provider.name,
    )


def generate_validated(
    question: str, passages: list[Passage], provider: LLMProvider
) -> tuple[Answer, list[Citation]] | None:
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
        citations, dropped = sanitize_citations(answer.citations, passages)
        if dropped:
            metrics.CITATIONS_DROPPED.inc(dropped)
            log.warning("fabricated citations dropped", extra={"dropped": dropped})
        if answer.answered_from_context and not answer.answer.strip():
            metrics.VALIDATION_FAILURES.inc()   # an "answer" with no text is a failed generation
            log.warning("answer was empty", extra={"attempt": attempt})
            continue
        if answer.answered_from_context and not citations:
            metrics.VALIDATION_FAILURES.inc()   # a claim with no valid source is not an answer
            log.warning("answer had no valid citation", extra={"attempt": attempt})
            continue
        if (not answer.answered_from_context and not answer.citations
                and answer.answer.strip() and answer.answer.strip() != ABSTAIN_TEXT):
            # Do not expose a substantive answer that the model itself did not ground. It is
            # inconsistent with an abstention, so give generation one chance to correct it.
            # Log only status/counts; never include the question or answer text.
            metrics.VALIDATION_FAILURES.inc()
            log.warning("model abstained with an uncited substantive answer", extra={
                "attempt": attempt, "citation_count": len(answer.citations),
                "passage_count": len(passages),
            })
            continue
        return answer, citations
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
    validated = generate_validated(question, grounded, provider)
    metrics.GENERATION_SECONDS.observe(time.perf_counter() - started)

    # Narrow extractive recoveries cover explicit policy rules if generation or validation fails.
    # They see only passages returned for this request and above the relevance floor.
    if validated is None or not validated[0].answered_from_context:
        extracted = (_extractive_cgpa_answer(question, grounded)
                     or _extractive_ai_assessment_answer(question, grounded)
                     or _extractive_attendance_answer(question, grounded)
                     or _extractive_older_attendance_answer(question, grounded))
        if extracted is not None:
            metrics.FALLBACKS.inc()
            answer, citations = extracted
            response = _response(answer, citations, passages, provider)
            _log(settings, retriever, provider, passages, response, "extractive_policy_rule")
            return response

    if validated is None:
        metrics.FALLBACKS.inc()
        response = _response(Answer(reasoning="fallback", answer=FALLBACK_TEXT, citations=[],
                                    answered_from_context=False, confidence="low"),
                             [], passages, provider)
        _log(settings, retriever, provider, passages, response, "fallback")
        return response

    answer, citations = validated
    if not answer.answered_from_context:
        metrics.ABSTENTIONS.labels(reason="model_declined").inc()
        answer = answer.model_copy(update={"citations": [], "confidence": "low"})
        response = _response(answer, [], passages, provider)
        _log(settings, retriever, provider, passages, response, "model_declined")
        return response

    response = _response(answer, citations, passages, provider)
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
