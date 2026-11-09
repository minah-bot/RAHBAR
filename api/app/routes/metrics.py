"""Prometheus metrics at /metrics (section 4.8).

Request count and latency are wired up already. Add your own the same way:
retrieval latency, generation latency, cache hits, abstentions, fallbacks, tokens.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

router = APIRouter()

REQUESTS = Counter("rahbar_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram(
    "rahbar_request_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)


RETRIEVAL_SECONDS = Histogram(
    "rahbar_retrieval_seconds", "Embed the question and query Chroma",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
GENERATION_SECONDS = Histogram(
    "rahbar_generation_seconds", "LLM call including the one retry",
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
CACHE_LOOKUPS = Counter("rahbar_cache_lookups_total", "Semantic cache lookups", ["result"])
ABSTENTIONS = Counter("rahbar_abstentions_total", "Answers declined", ["reason"])
FALLBACKS = Counter("rahbar_fallbacks_total", "Safe canned responses after a failed retry")
CITATIONS_DROPPED = Counter(
    "rahbar_citations_dropped_total", "Model citations that matched no retrieved chunk"
)
VALIDATION_FAILURES = Counter("rahbar_validation_failures_total", "Model output rejected")
TOKENS_IN = Histogram("rahbar_tokens_in", "Prompt tokens per request",
                      buckets=(100, 250, 500, 1000, 2000, 4000, 8000))
TOKENS_OUT = Histogram("rahbar_tokens_out", "Completion tokens per request",
                       buckets=(25, 50, 100, 200, 400, 800, 1600))


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
