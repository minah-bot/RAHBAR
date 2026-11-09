# Engineering notes

Questions 1, 2, 3 and 6 are answered from this repository's code. Questions 4, 5 and 7 need numbers or a story that only
your own runs can provide, so they say exactly what to run and what to write; they are marked **TEAM TODO**.

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?

Three pods start together and each would run the pipeline, so three writers race on the one volume behind
`chroma` (`k8s/base/chroma.yaml`, `volumeClaimTemplates`, line ~83): each is halfway through a different document and the
resulting collection is one no document produced. Every restart repeats the whole extraction and embedding cost, readiness
becomes untrue (ready before the index exists, or not ready for minutes while the kubelet kills it), and a document change
needs a redeploy. The object that prevents it is the **Job** `indexer` (`k8s/base/indexer-job.yaml:5`, `backoffLimit: 3` at
line 11, `restartPolicy: OnFailure` at line 17), with the CronJob `concurrencyPolicy: Forbid` in
`k8s/base/reindex-cronjob.yaml` guaranteeing a single writer. The API has no write path at all: `ChromaRetriever`
(`api/app/retrieval/chroma.py`) only reads, and `/readyz` (`api/app/routes/health.py:22`) holds a pod back until the index exists.

## 2. Readiness depends on Chroma, liveness does not: what if they were swapped and Chroma is slow for 90 seconds?

`/readyz` calls `check_ready()` (`api/app/retrieval/chroma.py:78`) which touches Chroma; `/healthz`
(`api/app/routes/health.py:16`) touches nothing. In the manifest the liveness probe is `k8s/base/api.yaml:39`
(`/healthz`, every 20 s, 3 failures) and readiness is line 43 (`/readyz`, every 10 s, 3 failures). Swapped, a 90-second Chroma
slowdown makes liveness fail 3 times in 60 s on **all three pods at once**, so the kubelet kills and restarts every
replica; they restart into the same slow Chroma, fail their startup and liveness again, and what was merely slow becomes
a full outage with no pod left to serve cached answers or a clear 503. As shipped, the same slowdown only fails readiness:
the pods leave the Service's endpoints (users get a 503 from the Ingress), nothing is restarted, and the pods rejoin by
themselves 10-30 s after Chroma recovers.

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?

`k8s/base/api.yaml:87` scales on CPU `averageUtilization: 70` of `requests.cpu` (100m, line 51). While a request waits for
the model the Python worker is idle (the call is I/O from the pod's point of view), so CPU stays low even when every
worker is blocked and users are queueing: the HPA measures the cost of *our* code (embedding the question, JSON
parsing, cosine over the cache), which is small, not the saturation of the service. The signal that should scale it is
**concurrency or queue depth**: in-flight requests per pod, or generation latency p95 (`rahbar_generation_seconds`), and
in practice the capacity of the model backend, which is where the work happens (a laptop Ollama does not get faster with more
API pods). To use it: expose an `rahbar_inflight_requests` gauge, scrape `/metrics` with Prometheus, add the Prometheus
Adapter (or KEDA) so the HPA can read a Pods/External metric, and put a ceiling on `maxReplicas` equal to what the
model backend can actually serve.

## 4. kubectl top pods shows the indexer's peak memory. What is your limit, what is your margin, and which exit code if wrong?

> **TEAM TODO:** run the indexer on your real corpus and sample while it runs:
> `make index; for i in $(seq 1 60); do kubectl top pod -n rahbar -l app=indexer --containers; sleep 5; done | tee docs/evidence/kubectl-top-indexer.txt`
> Then write: peak memory ___ Mi, limit in `k8s/base/indexer-job.yaml:64` (currently 1Gi, a starting estimate) ___ Mi,
> margin = limit / peak = ___. Too low a limit is an OOMKill: exit code **137** (`kubectl describe pod` shows
> `Reason: OOMKilled`, and the Job retries up to `backoffLimit`). A CPU limit set too low only slows the job (throttling).
> Do the same for `api`, `chroma`, `redis`, `ui` (`make top`) and set each request near the observed steady state.

## 5. recall@5 by question type: where is it worst, and why?

> **TEAM TODO:** after `make eval`, read `eval/results.json` -> `recall_at_5_by_type` and write which type is lowest.
> Expected shape (verify, do not copy): `exact` is usually the worst, because an embedding places "clause 5.4" and "clause 5.3"
> almost on top of each other (numbers carry little semantic signal) while definitional questions match on meaning.
> The technique from the RAG lecture that addresses it is **hybrid retrieval**: add a keyword (BM25) leg and merge the two
> rankings (reciprocal rank fusion), optionally a cross-encoder re-ranker; metadata filtering on clause number is a cheaper partial fix.
> Back it with your numbers and one failing question's retrieved list from the request log.

## 6. EMBED_MODEL is changed in the ConfigMap and only the API is restarted. What happens?

The index was built with `nomic-embed-text` and its collection metadata says so (`indexer/pipeline/store.py:42`). The restarted
API pod reads `EMBED_MODEL=<other>` from `k8s/base/configmap.yaml`. `/readyz` calls `pin_conflicts()`
(`api/app/retrieval/index.py:34`) via `check_ready()` (`api/app/retrieval/chroma.py:78`), finds
`embedding model mismatch: the index was built with 'nomic-embed-text' but this API is configured with EMBED_MODEL='...'`,
returns 503 `failed: embedding-pin`, and logs `embedding pin violated` (`chroma.py:86`) with both names. The new pod sits at
**0/1 READY**; with `maxUnavailable: 0` the rolling update never removes an old pod, so users keep being served by the old
replicas, which still hold the old model. `retrieve()` repeats the check on every query and the vector dimension is compared too.
Without the guardrail the pod would be ready, embed questions with model B and compare them to chunks embedded by model A:
if the dimensions differ Chroma raises an error per request; if they match (two 768-dimension models) it silently returns
nearest neighbours in the wrong space, so the answers are fluent, cited and wrong, and no probe, metric or log says so.

## 7. The failure that cost us more than an hour

> **TEAM TODO (write this yourselves):** it must be a real one from your own work: symptoms, what you wrongly believed first,
> and the exact command, metric or log line that finally told you the truth. Candidates that are likely to be real for this
> stack, if you hit them: `ImagePullBackOff` for a locally built image (needs `minikube image load`), every Service name failing
> after default-deny (DNS egress), the API unable to reach Ollama (host bound to 127.0.0.1, or the `ipBlock` address), Chroma
> `CrashLoopBackOff` under `readOnlyRootFilesystem` (needs a writable `/tmp`), or HPA `<unknown>/70%` (metrics-server).
> Do not write one you did not have: this answer is graded on being specific.
