# ADR 0002: Word-based, section-aware chunking

- Status: Accepted
- Context: Chunks need to retain headings and document/page/version identity, while
  configuration must be usable without each provider's tokenizer.
- Decision: Split by page and paragraph, attach the current heading, then use
  whitespace-delimited word windows with configurable overlap. Defaults are 400/50 words.
- Consequence: Chunk boundaries are simple and reproducible, but word count is not model
  token count and long paragraphs can lose finer sentence boundaries. Recall comparisons
  across size/overlap settings remain to be measured on the selected corpus.
