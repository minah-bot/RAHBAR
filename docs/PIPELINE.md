# Indexing and retrieval pipeline

Items marked **TEAM TODO** need numbers only your cluster or your documents can produce. Everything else
describes what the code in this repository does.

## Extraction
`indexer/pipeline/extract.py`. PDFs are read page by page with pdfplumber. Tables are found with
`find_tables()`, rendered as Markdown rows (`| a | b |`) and the text *outside* the table bounding boxes is extracted
separately, so columns stay columns instead of the jumbled "a b c d" a plain text dump gives. A page whose text layer
has fewer than 25 characters is rasterised (pdf2image, 200 dpi) and OCR'd with Tesseract, and the page is flagged
`ocr` (the indexer logs `ocr_pages` per document). HTML drops script/nav/footer and converts tables the same way;
Markdown loses its front matter.

* **Documents that needed OCR:** **TEAM TODO** (`kubectl logs job/indexer -n rahbar | grep ocr_pages`).
* **One before-and-after table example:** **TEAM TODO** (run `python eval/find_sections.py --grep "<a word in the table>"`
  and paste the chunk next to the PDF page).
* **What cannot be recovered:** tables are appended after the page's other text, so their position on the page is lost;
  merged cells become repeated or empty cells; OCR errors on low-quality scans are kept as OCR read them.

## Cleaning
`indexer/pipeline/clean.py`: (1) lines that repeat in the first or last two lines of at least 40% of pages are removed as
running headers and footers (compared with digits masked, so "Page 3" and "Page 4" match); (2) bare page-number lines;
(3) words hyphenated across a line break are re-joined; (4) control characters and runs of blanks are normalised.
**Deliberately not stripped:** clause numbers (`5.4`) and the digits inside table cells. They look like noise but they are
the headings and the exact terms (fees, thresholds) that chunking and exact-term retrieval depend on.

## Chunking
ADR 0002. Section, then paragraph/clause, then sentence; tables by rows with the header repeated; overlap inside a
section. Size and overlap are in tokens where a token is a regex word or punctuation mark (`pipeline/tokens.py`).

* **Shipped:** **TEAM TODO** size ___ / overlap ___ (the ConfigMap values).
* **Storage amplification (stored tokens / original tokens):** **TEAM TODO** (the indexer logs it:
  `kubectl logs job/indexer -n rahbar | grep "chunking summary"`). Note: `stored_tokens` counts chunk bodies; the heading
  is kept in metadata and prepended only to the text that is embedded.

Measure three or more configurations on the golden set with `make sweep` (offline, real Ollama, no cluster) and paste:

| Size (tokens) | Overlap (tokens) | recall@5 | Chunks stored |
| --- | --- | --- | --- |
| **TEAM TODO** | | | |

## Embedding
`nomic-embed-text` through Ollama (`OLLAMA_URL`), prefixed `search_document:` for chunks and `search_query:` for questions
(the model was trained with those prefixes). Vector dimension: **TEAM TODO** (768 for nomic-embed-text; read it from the
collection metadata). **The pin:** `EMBED_MODEL` is one ConfigMap key read by both the indexer and the API. The indexer
writes `embed_model` and `dimension` into the collection metadata (`indexer/pipeline/store.py`). The API compares them with
its own `EMBED_MODEL` in `/readyz` and again on every query (`api/app/retrieval/index.py:pin_conflicts`,
`retrieval/chroma.py:check_ready`); on a mismatch it answers 503 with `failed: embedding-pin`, stays at 0/1 READY, and
logs `embedding pin violated` with both model names. It also compares the question vector's dimension with the index's.

## Semantic cache
ADR 0003. Threshold chosen with `eval/tune_cache.py` (paraphrase pairs that should hit, near-miss pairs that must not).

| Threshold | Paraphrases hit | Near-misses wrongly hit |
| --- | --- | --- |
| **TEAM TODO** (run `python eval/tune_cache.py`) | | |

* **The false hit that ruled one out, and the value shipped:** **TEAM TODO**.

## Relevance floor
`RELEVANCE_FLOOR` is the best-passage similarity below which the system abstains without calling the model. **TEAM TODO:**
the distribution of top-1 scores for answerable versus must-abstain golden questions (`eval/results.json` has them in
`retrieved`), and the floor chosen between them. Too high abstains on real questions (`answer_rate_on_answerable` falls);
too low lets off-topic questions reach the model.

## Evaluation method
`eval/run_eval.py`. A golden question lists its relevant passages as `(source, section)` pairs, not chunk ids, because chunk
ids change whenever chunking changes while the section a clause lives in does not. A retrieved passage matches a golden one
when the source file names are equal and the sections are equal, or the golden section is a clause-number prefix of the
retrieved one (`5.4` matches `5.4 Academic Warning` and `5.4.1 ...`, not `5.40`); case and whitespace are ignored.
`recall@k` is the share of relevant pairs found in the top k. Citation validity is the share of citations naming a
retrieved (source, section) over the whole run; the gate (section 5.4) also fails if any must-abstain question is answered.
Two companion numbers are reported but not gated: `answer_rate_on_answerable` (catches a system that "passes" abstention by
refusing everything) and `grounded_answers_without_citation`.
