# ADR 0003: Semantic cache in Redis, scoped to the index it was built from

## Status
Accepted · 2026-11-06

## Context
Generation is by far the slowest and most expensive step, and users ask the same things in different
words. A cache keyed on exact text would almost never hit; one keyed on embedding similarity hits on
paraphrases but can return a confident, cited, *wrong* answer if two different questions land close
together ("BS fee" versus "MS fee"). The cache must survive a pod restart, be shared by the three API
replicas, and must not outlive the index it was derived from.

## Decision
Redis (`k8s/base/redis.yaml`: Deployment, `Recreate`, AOF persistence on a PVC, `allkeys-lru`,
ClusterIP only). An entry is `rahbar:cache:<index_id>:<uuid>` holding the question's embedding and the
validated response; it also expires after `CACHE_TTL_SECONDS`. A lookup embeds the question (a memo makes
the second embedding free), compares it with that index's entries and hits at or above `CACHE_THRESHOLD`.
The threshold comes from `eval/tune_cache.py` over labelled paraphrase and near-miss pairs; the value
shipped and the false hit that ruled out a lower one are recorded in PIPELINE.md.
**Invalidation on re-index:** the key carries the `index_id` the indexer wrote into the collection
metadata, the API reads the new id after the pointer swap (ADR 0001), old keys stop matching and are
deleted by the first replica to notice. The indexer never talks to Redis, which keeps the NetworkPolicy
"redis from api only" true.
**Never cached:** follow-ups (`follow_up: true`), questions that ask about an older version (the
filter, not the wording, decides their answer), abstentions, fallbacks, and anything that failed
validation. A Redis failure is logged and skipped; it never fails an answer.

## Options considered
* *Exact-match cache:* safe but nearly useless.
* *A vector index in Redis (RediSearch) or Chroma:* the right move beyond a few thousand entries;
  the cap here is 2,000 entries, where a linear scan is a few tens of milliseconds. The cap is the
  trigger to revisit.
* *An in-process LRU per pod:* three caches, three hit rates, lost on every restart.
* *Invalidate by having the indexer flush Redis:* gives the indexer a network path to the cache.

## Consequences
Easier: repeat questions skip retrieval and generation. Harder: the threshold is a trade between hit rate and
wrong answers and has to be re-tuned if the embedding model changes. Watch: `rahbar_cache_lookups_total`
by result (hit ratio is hit / (hit + miss)), and any user report of an answer that fits a similar but
different question.
