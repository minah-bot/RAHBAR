"""Semantic cache on Redis (section 3.3, ADR 0002).

A question hits when its embedding is within CACHE_THRESHOLD cosine similarity of a cached one.
Entries live under `rahbar:cache:<index_id>:<uuid>`, so they are tied to the index that
produced them: after a re-index the API reads a new index_id, the old entries stop matching,
and they are purged. The indexer never touches Redis (NetworkPolicy: redis from api only).
"""

from __future__ import annotations

import json
import logging
import math
import re
import uuid
from typing import Any

from ..routes import metrics
from ..schemas import AskResponse

log = logging.getLogger("rahbar.cache")

PREFIX = "rahbar:cache:"
MAX_ENTRIES = 2000   # a linear scan stays cheap; ADR 0002 says when to move to a vector index


_YEARS = re.compile(r"\b(?:19|20)\d{2}\b")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class SemanticCache:
    def __init__(self, redis_client: Any, threshold: float, ttl_seconds: int) -> None:
        self.redis = redis_client
        self.threshold = threshold
        self.ttl = ttl_seconds
        self._purged_for: str | None = None

    def ping(self) -> bool:
        return bool(self.redis.ping())

    def invalidate_other_indexes(self, index_id: str) -> int:
        """Delete every entry that belongs to a different index (invalidate on re-index)."""
        if self._purged_for == index_id:
            return 0
        removed = 0
        mine = f"{PREFIX}{index_id}:"
        for key in self.redis.scan_iter(match=f"{PREFIX}*", count=500):
            name = key.decode() if isinstance(key, bytes) else key
            if not name.startswith(mine):
                self.redis.delete(key)
                removed += 1
        self._purged_for = index_id
        if removed:
            log.info("cache invalidated after re-index", extra={"removed": removed,
                                                               "index_id": index_id})
        return removed

    def lookup(self, vector: list[float], index_id: str,
               question: str = "") -> tuple[AskResponse | None, float]:
        """Return (cached response, similarity) for the best entry at or above the threshold.
        A semantic match never crosses an explicit year: "2023 rules" must not reuse the answer
        cached for "2026 rules", however close the embeddings are."""
        self.invalidate_other_indexes(index_id)
        keys = list(self.redis.scan_iter(match=f"{PREFIX}{index_id}:*", count=500))
        if not keys:
            return None, 0.0
        best: tuple[float, dict[str, Any]] | None = None
        for raw in self.redis.mget(keys):
            if raw is None:
                continue
            entry = json.loads(raw)
            if set(_YEARS.findall(question)) != set(_YEARS.findall(entry.get("q", ""))):
                continue
            sim = cosine(vector, entry["vec"])
            if best is None or sim > best[0]:
                best = (sim, entry)
        if best is None or best[0] < self.threshold:
            return None, best[0] if best else 0.0
        # request_id and cached are per-request, not stored: the caller overwrites both.
        return AskResponse(request_id="-", **best[1]["resp"]), best[0]

    def store(self, vector: list[float], index_id: str, response: AskResponse,
              question: str = "") -> bool:
        """Callers decide whether an answer is safe to cache (see answer.py); this just writes."""
        count = sum(1 for _ in self.redis.scan_iter(match=f"{PREFIX}{index_id}:*", count=500))
        if count >= MAX_ENTRIES:
            return False
        payload = json.dumps({"q": question, "vec": vector, "resp": response.model_dump(
            exclude={"request_id", "cached"})})
        self.redis.set(f"{PREFIX}{index_id}:{uuid.uuid4().hex}", payload, ex=self.ttl)
        metrics.CACHE_LOOKUPS.labels(result="stored").inc()
        return True


def connect(host: str, port: int) -> Any:
    import redis

    return redis.Redis(host=host, port=port, socket_connect_timeout=2, socket_timeout=2)
