# Indexing and retrieval pipeline

## Extraction

`pipeline.extract.extract` supports PDF, Markdown, plain text, and HTML. PDF text is
extracted per page with pdfplumber. Pages with an empty text layer are rendered with
poppler and sent to Tesseract; the `Page.ocr` flag records that path. The image-processing
system packages are installed in the indexer image. OCR depends on readable scans and
installed language data; handwriting and damaged pages are not recovered reliably.
No FAST-NUCES source documents were present in the workspace, so no source-document OCR
or table before/after result is claimed. Markdown fixture front matter is removed from
content and its version/superseded values are preserved by discovery metadata.

## Cleaning

Whitespace and null bytes are normalized. Short, repeated lines appearing on most pages
are removed as likely headers or footers. Headings, numbered clauses, tables, paragraph
breaks, page numbers, source names, versions, and superseded flags are retained. This
heuristic can mistake a repeated short legal phrase for page furniture; inspect a newly
fetched corpus before indexing it.

## Chunking

Chunk size and overlap are measured in whitespace-delimited word tokens, not model tokens.
Headings set section metadata, paragraphs are split independently, and overlap is carried
between neighboring windows within a paragraph. Defaults are 400 words and 50 words.
The configured storage amplification is approximately `size / (size - overlap)` for long
paragraphs (400 / 350 = 1.14); short paragraphs incur less overlap. This is a planning
estimate, not a measured corpus result. The three required recall measurements have not
been run because there is no running indexed API in this environment.

| Size (words) | Overlap (words) | recall@5 | Chunks stored |
| ---: | ---: | --- | --- |
| 250 | 25 | Not measured | Not measured |
| 400 | 50 (configured default) | Not measured | Not measured |
| 600 | 75 | Not measured | Not measured |

## Embedding and index replacement

Both programs read `EMBED_MODEL` (default `nomic-embed-text`). Indexer embeddings and API
query embeddings use Ollama's `/api/embed` endpoint. The indexer records the model name,
dimension, and a fingerprint of chunk IDs as collection metadata. `/readyz` rejects an
empty Chroma collection or a model-name mismatch. The API never writes to Chroma.
Indexing stages into a temporary collection before replacing the active collection. Chroma
does not provide an atomic rename for collections; the current copy-over strategy has a
brief replacement window and should be upgraded to a blue/green alias scheme for zero
downtime. See ADR 0001.

## Retrieval, cache, and evaluation

Retrieval uses cosine similarity (`1 - distance`) and adds a small exact-token overlap
component to help with clause numbers, codes, and fees. By default superseded documents
are excluded; a four-digit year in the query filters by `doc_version` so historical policy
questions can target that version. `RELEVANCE_FLOOR` defaults to 0.25. The threshold is a
configurable starting point, not a measured optimum.

Redis caching is optional through `REDIS_URL`. It stores only context-supported answers,
uses the same pinned embedding model, and compares cosine similarity against
`CACHE_SIMILARITY` (default 0.97). The index fingerprint is part of the namespace, so a
new index naturally invalidates prior answers. `invalidate()` can explicitly clear cached
entries for a collection/model/provider.

Evaluation recall matches case-insensitive `(source, section)` pairs against top-k results;
it does not claim chunk-level recall. Citation validity checks the same source/section
pair against retrieved passages. Baseline comparison allows a five-point recall@5 drop;
citation validity must be at least 0.95 and every marked must-abstain question must be
declined. No results or baseline are checked in because that would imply an unrun
measurement. Use `eval/fixtures/golden.fixture.yaml` to measure the local fixture corpus.
