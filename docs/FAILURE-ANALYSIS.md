# Failure analysis

# Failure analysis

The following are failure scenarios derived from the implemented contracts. They are
diagnostic guidance, not claims that fault injection was run against a live deployment.
Run these checks and add observed timestamps/logs before submission.

## Extraction · image-only PDF without OCR runtime

- **What fails:** A PDF page has no text layer and poppler or Tesseract is unavailable.
- **Symptom:** The indexer exits non-zero with the page number and OCR error. It must not
  silently index an empty page as if extraction succeeded.
- **Diagnosis:** Check the indexer Job log, `Page.ocr`, and whether `pdftotext` returns
  useful content. The indexer Dockerfile installs the OCR binaries; a local environment
  must install them separately.
- **Remedy:** Install the declared system packages and language data, then inspect OCR
  output for the scan. Handwriting and low-resolution scans may need manual correction.

## Chunking · oversized or repeated text

- **What fails:** A large paragraph is split at the configured word boundary, or a
  repeated short line is incorrectly classified as page furniture.
- **Symptom:** An answer-bearing clause spans neighboring chunks, or a phrase disappears.
- **Diagnosis:** Inspect the cleaned page text and chunk boundaries for the affected source;
  compare `CHUNK_SIZE` and `CHUNK_OVERLAP`. The overlap is in whitespace-delimited words.
- **Remedy:** Adjust chunk settings based on measured recall and review repeated-line
  removal on that document. Do not claim a recall improvement without rerunning evaluation.

## Retrieval · stale embedding pin or weak relevance

- **What fails:** Runtime `EMBED_MODEL` differs from collection metadata, or the selected
  similarity falls below `RELEVANCE_FLOOR`.
- **Symptom:** `/readyz` returns 503 and names the model mismatch; alternatively retrieval
  returns no evidence and the answer path abstains. Retrieval scores appear in request logs.
- **Diagnosis:** Compare `EMBED_MODEL` in both processes with the Chroma collection
  metadata; check collection count and the configured score floor.
- **Remedy:** Use the pinned model consistently and re-index with the separate indexer.
  Lower a relevance floor only after measuring false positives and recall.

## Source filtering · superseded policy content

- **What fails:** A retrieved chunk is erroneously marked current, or a historical query
  lacks a matching `doc_version`.
- **Symptom:** Current queries may cite an obsolete rule, or version-specific retrieval
  returns no passage. Each retrieved response exposes source, section, page and score.
- **Diagnosis:** Inspect `_index.json` or fixture front matter and the stored Chroma
  metadata. Compare the query year with `doc_version`.
- **Remedy:** Correct source metadata and re-run the indexer. For a query with a year the
  retriever searches that version; otherwise it filters superseded records.

## Generation · malformed output or fabricated citation

- **What fails:** The provider returns invalid schema data, cites outside the retrieved
  passages, or answers without evidence.
- **Symptom:** The service retries generation once, then returns a safe abstention with no
  citations; the fallback and abstention Prometheus counters increase.
- **Diagnosis:** Inspect provider/model configuration and the counter names at `/metrics`.
  Keys are not logged. Stub mode can validate API wiring without an external model.
- **Remedy:** Fix the provider endpoint/model or prompt issue and verify citations against
  retrieved source/section/page. Never weaken citation validation to make a result pass.
