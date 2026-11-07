# ADR 0001: The indexer is a Job, and the API only reads

## Status
Accepted · 2026-11-06

## Context
RAHBAR runs three API replicas against one Chroma volume. The obvious design is "index when the
API starts": the code already exists, and there is nothing extra to deploy. It breaks four ways.
Three pods start together, so three writers race on one volume, each halfway through a different
document, and the result is an index no document produced. Every restart (a rolling update, an
OOMKill, a node drain) pays the whole extraction and embedding cost again. Readiness stops being
true: either a pod says "ready" before the index exists and serves empty retrievals, or it takes
minutes to say it and the kubelet kills it while it is still chunking. And changing a document or
a chunk size would need a redeploy, because indexing lives in the image's startup path.

## Decision
`indexer` is a Kubernetes **Job** (`k8s/base/indexer-job.yaml`, `restartPolicy: OnFailure`,
`backoffLimit: 3`), and a **CronJob** `reindex` runs the identical pod template weekly with
`concurrencyPolicy: Forbid`, so two indexers can never run at once. The API container has no write
path to Chroma at all: `ChromaRetriever` only calls `get_collection`, `query`, `count` and `get`.
The API's `/readyz` (`api/app/routes/health.py`) returns 503, naming the failed dependency, until
Chroma answers, a non-empty collection exists, the embedding pin holds and Redis answers, so a pod
without an index is withheld from the Service instead of serving nothing.

**Re-indexing never touches the index being served.** Each run builds a brand-new collection
`rahbar-<timestamp>`, writes every chunk, checks the count, and only then moves a pointer collection
`rahbar-active` to the new name (`indexer/pipeline/store.py`). The API follows the pointer (re-read
at most every 5 s), so pods keep answering from the old index until the instant of the swap and
never see a half-built one. The previous collection is kept for one generation as a rollback and
older ones are deleted. The same swap changes the semantic cache's index id, which invalidates
cached answers (ADR 0003). One command re-indexes with no redeploy and no downtime:
`kubectl create job --from=cronjob/reindex manual-1 -n rahbar`.

The Job's first init container waits for Chroma and for documents to appear on the corpus PVC, so
`kubectl apply -k` followed by `make corpus-load` works in either order.

## Options considered
* *Index in API startup:* rejected for the four reasons above.
* *Index in an init container of the API pods:* still three writers, and still a restart cost.
* *Upsert in place, then delete stale ids:* simpler, but readers briefly see a mix of old and new
  chunks, and a change of embedding model (a different vector dimension) cannot be done in place.
* *One writer through a leader-elected API pod:* needs leader election code and still couples
  indexing to the serving lifecycle.

## Consequences
Easier: restarts and rollouts are cheap, a bad document fails a Job (visible, retried, logged) instead
of crashing serving pods, re-indexing is a command. Harder: there are two images' worth of lifecycle
to reason about, and a Job's pod template is immutable, so `make apply` deletes the finished Job
before re-applying. Watch: the indexer's peak memory (the OOMKill risk lives here, not in the API) and
disk growth on the Chroma PVC if old collections are not being deleted.
