"""Optional Redis semantic cache. Failures are deliberately fail-open."""

from __future__ import annotations

import hashlib
import json
import math
from typing import cast

from ..config import Settings
from ..schemas import AskResponse


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    denom = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return sum(x * y for x, y in zip(left, right)) / denom if denom else -1.0


class SemanticCache:
    def __init__(self, settings: Settings, retriever: object, provider: str) -> None:
        self.settings = settings
        self.retriever = retriever
        self.provider = provider
        collection = getattr(retriever, "collection", None)
        metadata = getattr(collection, "metadata", None) or {}
        fingerprint = metadata.get("index_fingerprint", "unversioned")
        self.prefix = (f"rahbar:cache:{settings.collection}:{settings.embed_model}:"
                       f"{fingerprint}:{provider}:")

    def _client(self):
        if not self.settings.redis_url:
            return None
        import redis

        return redis.Redis.from_url(self.settings.redis_url, decode_responses=True,
                                   socket_connect_timeout=0.2, socket_timeout=0.2)

    def lookup(self, question: str) -> AskResponse | None:
        client = self._client()
        embed = getattr(self.retriever, "embed", None)
        if client is None or embed is None:
            return None
        try:
            vector = embed(question)
            exact = client.get(self.prefix + hashlib.sha256(question.casefold().strip().encode()).hexdigest())
            best: tuple[float, str] | None = None
            if exact:
                best = (1.0, exact)
            else:
                for key in client.scan_iter(match=self.prefix + "*", count=100):
                    raw = client.get(key)
                    if not raw:
                        continue
                    item = json.loads(raw)
                    score = _cosine(vector, item["vector"])
                    if best is None or score > best[0]:
                        best = score, raw
            threshold = self.settings.cache_similarity
            if best and best[0] >= threshold:
                payload = json.loads(best[1])["response"]
                return AskResponse.model_validate(payload)
        except Exception:
            return None
        return None

    def save(self, question: str, response: AskResponse) -> None:
        client = self._client()
        embed = getattr(self.retriever, "embed", None)
        if client is None or embed is None or never_cache(response):
            return
        try:
            vector = embed(question)
            key = self.prefix + hashlib.sha256(question.casefold().strip().encode()).hexdigest()
            client.setex(key, 3600, json.dumps({"vector": vector,
                                               "response": response.model_dump()}))
        except Exception:
            return


def never_cache(response: AskResponse) -> bool:
    """Declines are not reusable answers; only validated contextual answers are cached."""
    return not response.answered_from_context or not response.citations


def invalidate(settings: Settings, provider: str | None = None) -> int:
    """Delete cached answers for the active model and collection; return deleted keys."""
    if not settings.redis_url:
        return 0
    import redis

    prefix = f"rahbar:cache:{settings.collection}:{settings.embed_model}:"
    if provider:
        prefix += f"{provider}:"
    client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    keys = list(client.scan_iter(match=prefix + "*", count=500))
    return cast(int, client.delete(*keys)) if keys else 0
