# Failure analysis

Break **four** of the six stages on purpose, one at a time, restore each, and fill one block per stage. Every number below
must come from your own run: the blocks list what to do and where the evidence appears, and leave the results to you.
Before each experiment: `make eval` once as the healthy reference (`cp eval/results.json /tmp/healthy.json`).
After each: restore, re-index if needed (`make index`), and check `make eval` returns to the reference.

How the system exposes each stage (so you can diagnose it *as if you had not caused it*):

| Where it shows | What it tells you |
| --- | --- |
| request log line `answered` | `chunk_ids`, `scores`, `abstained`, `reason`: the retrieved context, where five of six failures are visible |
| `rahbar_abstentions_total{reason}` | `below_relevance_floor` rising = retrieval is returning weak passages |
| `rahbar_citations_dropped_total`, `rahbar_validation_failures_total`, `rahbar_fallbacks_total` | the model is inventing sources, or breaking the schema |
| indexer log `chunked` / `chunking summary` | `ocr_pages`, `empty_pages`, `chunks`, `amplification` per document |
| `eval/results.json` | `recall_at_5_by_type`, `citation_validity`, `abstention_correctness`, `answer_rate_on_answerable` |

## Stage A · Extraction
- **What we broke:** index the scanned PDF without OCR: set `OCR_ENABLED: "false"` in the indexer's environment
  (a temporary `env` entry on the Job, or `kubectl set env`), `make index`.
- **Symptom:** **TEAM TODO** (expected: questions about the scanned document abstain or retrieve unrelated text).
- **Diagnosis, as if we had not caused it:** the indexer log line for that document shows `empty_pages` equal to its page
  count and `chunks: 0` (or very few); `ocr_pages: 0`. The metric that moves: `must-abstain`-style abstentions on questions that
  *should* be answerable, i.e. `answer_rate_on_answerable` falls and `recall_at_5` for that document's questions is 0.
- **Remedy:** **TEAM TODO** (remove the switch; OCR is on by default; add a check that fails the Job when a document
  yields no chunks).

## Stage B · Chunking
- **What we broke:** `CHUNK_STRATEGY=fixed`, `CHUNK_SIZE=100`, `CHUNK_OVERLAP=0` on the indexer (fixed windows, no sections).
- **Symptom:** **TEAM TODO** (expected: exact-term and clause questions fail; citations lose their section names).
- **Diagnosis, as if we had not caused it:** `chunking summary` shows `chunks` jumping and `amplification` near 1.0 with no
  overlap; `recall_at_5_by_type.exact` collapses first; retrieved `section` values in the request log are empty strings.
- **Remedy:** **TEAM TODO** (restore the structure-aware chunker; the CI gate would have blocked this change).

## Stage C · Retrieval
- **What we broke:** (a) `TOP_K: "1"` in the ConfigMap, restart api; and (b) the API-only `EMBED_MODEL` change that is also
  the section 4.4 demonstration: `kubectl edit configmap rahbar-config -n rahbar`, change `EMBED_MODEL`,
  `kubectl rollout restart deployment/api -n rahbar`.
- **Symptom:** **TEAM TODO** (a) recall@5 is capped at what one chunk can hold and multi-clause answers are incomplete;
  (b) the new pods sit at `0/1 READY` and the old ones keep serving.
- **Diagnosis, as if we had not caused it:** (a) `scores` list has a single entry per request in the `answered` log line and
  `recall_at_5` equals recall@1; (b) `kubectl get pods -n rahbar` shows `0/1`, `kubectl logs <pod>` shows
  `embedding pin violated` naming both models, and `GET /readyz` returns 503 with `failed: embedding-pin`.
- **Remedy:** **TEAM TODO** (restore the values; the pin makes (b) impossible to miss).
- Evidence to commit: `docs/evidence/pin-0of1.png` (the pod at 0/1 and the log line).

## Stage D · Prompt
- **What we broke:** on a throwaway branch, delete rule 1 ("Answer ONLY from the passages...") and rule 3 from `SYSTEM` in
  `api/app/services/prompt.py`, rebuild and load the api image, `kubectl set image`.
- **Symptom:** **TEAM TODO** (expected: must-abstain questions get a fluent answer from the model's own knowledge).
- **Diagnosis, as if we had not caused it:** `abstention_correctness` drops below 1.0 in the eval and the gate fails;
  `rahbar_abstentions_total{reason="model_declined"}` stops increasing while retrieval scores for those questions stay low.
- **Remedy:** **TEAM TODO** (restore the prompt; the gate is the guard).

## Stage E · Source
- **What we broke:** on a throwaway branch, make `answer_question` pass `include_superseded=True` always (or remove the
  `where` filter in `ChromaRetriever.retrieve`), then ask a version-sensitive question.
- **Symptom:** **TEAM TODO** (expected: the answer quotes the superseded number, with a real citation to the old PDF).
- **Diagnosis, as if we had not caused it:** the retrieved `source` in the log line is the superseded document; the
  citation validity is still 100%, because the citation is genuine, which is why this needs the `version` golden questions.
- **Remedy:** **TEAM TODO** (restore the filter; `retrieval/version.py` is the only door back to old versions).

## Stage F · Generation
- **What we broke:** `TEMPERATURE: "1.2"` in the ConfigMap, restart api.
- **Symptom:** **TEAM TODO**
- **Diagnosis, as if we had not caused it:** retrieved chunks and scores are unchanged (retrieval is fine) while
  `rahbar_validation_failures_total` / `rahbar_citations_dropped_total` rise: the model, not the pipeline, got worse.
- **Remedy:** **TEAM TODO**

> Remember the lecture's rule: only F is the model's fault, so check A to E first.
