# ADR 0002: Structure-aware chunking with overlap

## Status
Accepted · 2026-11-06 (shipped size and overlap: fill in from `make sweep`; see PIPELINE.md)

## Context
The corpus is policy text: numbered clauses ("5.4 Academic Warning"), tables of fees, and long
definitions. Questions are of two kinds that pull in opposite directions: definitional ones want
enough surrounding text to be answerable, exact-term ones ("clause 5.4", "PKR 3,000") want the
clause number and its number kept together in one chunk. Fixed-size slicing cuts clauses in half,
separates a heading from its text, and cuts a table mid-row.

## Decision
`indexer/pipeline/chunk.py` splits in this order: **section** (a heading is a hard boundary; the
heading text becomes `metadata.section` and is also placed in front of the text that is embedded),
then **paragraph / numbered clause**, then **sentence**, then words as a last resort. Chunks are packed
up to `CHUNK_SIZE` tokens and neighbours inside a section share `CHUNK_OVERLAP` tokens. A table is kept
whole when it fits; otherwise it is split by rows and the header row is repeated in every piece, so a
row is never cut and a fragment is never a column of unlabelled numbers. Every chunk carries `source`,
`section`, `page`, `doc_version` and `superseded`. Ids are `<document>-<NNNN>-<hash>`, stable for the same
document, position and content, so a re-index of unchanged files produces the same ids and an unchanged chunk keeps its vector (see ADR 0001). Two files with the same name in different folders get different ids (the hash covers the document's corpus-relative identity). Pages that are not content, such as a table of contents, are left out by data and not by code: `exclude_pages: [1, 2, 3]` on a document in `corpus/sources.yaml`, which `fetch.py` copies into `_index.json`.
The size and overlap are measured, not guessed: `eval/chunk_sweep.py` compares at least three
configurations on the golden set (table in PIPELINE.md).

## Options considered
* *Fixed 100-token windows, no overlap:* also the deliberate failure for Stage B in FAILURE-ANALYSIS.md.
* *Sentence windows:* good for definitions, loses the clause number for exact-term questions.
* *Whole sections as chunks:* best for coherence, but a long section dilutes the embedding and
  wastes the model's context on irrelevant text.
* *A semantic splitter (embedding-based):* costs an embedding pass per candidate boundary for little
  gain on text that is already structured by numbered headings.

## Consequences
Easier: citations are meaningful (a section name a user can open), exact clause numbers survive,
tables stay readable. Harder: heading detection is a heuristic (`heading_of`): a PDF that numbers
paragraphs without headings, or a scanned page OCR'd with errors, will chunk worse. Watch: storage
amplification (stored tokens / original tokens, logged by every indexer run) and recall on exact-term
questions after any change here; the CI gate exists to catch a chunking tweak that quietly breaks them.
