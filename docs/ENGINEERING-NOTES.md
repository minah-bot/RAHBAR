# Engineering notes

# Engineering notes

These notes distinguish code behavior from cluster measurements. Kubernetes manifests,
cluster evidence, production corpus, and a live indexed API are not part of this scoped
implementation, so resource numbers and deployment observations are intentionally absent.

## 1. Indexing in API startup: what breaks with 3 replicas and one Chroma volume?

If every API replica writes the index on startup, they race to parse/embed the same corpus,
duplicate expensive work, and may mutate a shared collection while requests query it.
Startup and readiness become coupled to the embedding server and corpus. Here indexing is
in `indexer/pipeline/run.py`; API construction only selects a retriever/provider in
`api/app/main.py`. Run the indexer as a separate one-shot Job and make the API read-only.

## 2. Readiness and liveness swapped, Chroma slow for 90 seconds: what happens?

`/healthz` is dependency-free. It should be used for liveness: a temporary Chroma outage
must not restart every API pod. `/readyz` checks Chroma count and embedding metadata, and
checks Redis when configured. A slow readiness dependency can remove a pod from service;
timeouts and probe thresholds must be set in the deployment. There is no deployment here
to measure that 90-second scenario.

## 3. What does a CPU-based HPA measure on a generation service, and what should scale it?

CPU is a weak proxy when requests spend much of their time waiting on Ollama or a hosted
model. Queue depth, in-flight requests, request rate, and generation latency are more
direct signals. This repository defines request/latency Prometheus metrics but has no HPA
or observed `kubectl top` output; do not infer production capacity from it.

## 4. The indexer's peak memory, its limit, the margin, and the exit code when it is wrong

Peak memory has not been measured; the size and format of the real corpus are not present.
Measure the indexer Job while processing the chosen corpus, then set its memory limit with
headroom and record `kubectl top pod` output. A failure exits non-zero (including the
explicit OCR, embedding, or storage exceptions), allowing a Job controller to retry.
Do not invent a peak or limit before this measurement.

## 5. recall@5 by question type: where is it worst, and why?

No live fixture or production evaluation was run in this environment, so no type has a
measured worst score. `eval/run_eval.py` reports `recall_at_5_by_type`; run the fixture
golden set against an indexed API, inspect per-question retrieval, and document the actual
lowest type and likely cause. Exact clause/code questions can be hurt by embedding-only
search, which is why retrieval also includes lexical token overlap.

## 6. EMBED_MODEL changed on the API only: what does your system do?

`/readyz` compares the configured model against collection metadata and returns HTTP 503
with expected and indexed values on mismatch. Both API and indexer must use the same pin.
Changing only the API model must not be treated as a successful rollout.

## 7. The failure that cost us more than an hour

No personal debugging incident is claimed for this AI-assisted implementation. Complete
this section from your own work: record the actual failure, the misleading symptom, the
log/metric/command that isolated it, and the fix. Explain the result in your own words for
the viva rather than presenting a hypothetical incident as observed evidence.
