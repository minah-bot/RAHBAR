# ADR 0003: Optional Redis semantic cache

- Status: Accepted
- Context: Repeated and near-identical questions can avoid provider generation, but a
  wrong semantic match could return an unrelated policy answer.
- Decision: Redis is opt-in through `REDIS_URL`; question vectors use the pinned embedding
  model, cosine similarity must meet `CACHE_SIMILARITY` (0.97 by default), only
  context-supported answers are stored, entries expire after one hour, and the active
  index fingerprint is included in the key namespace.
- Consequence: Redis outages fail open to retrieval/generation. Thresholds are defaults,
  not empirically tuned values; evaluate false hits before lowering them. Explicit
  invalidation is available in `api/app/services/cache.py`.
