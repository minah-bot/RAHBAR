# RAHBAR corpus setup — FAST-NUCES student and academic documents

`sources.yaml` lists six verified official FAST-NUCES PDFs. Together they contain
260 pages and include the 2023 and June 2026 editions of Undergraduate Academic
Rules and Regulations. The 2023 edition is retained for historical questions and
is marked `superseded: true`; use the 2026 edition for current policy questions.

## Download and verify

From the repository root, run:

```bash
make install
make fetch
.venv/bin/python corpus/fetch.py --verify
```

`corpus/fetch.py` writes the PDFs to `corpus/raw/`, creates `corpus/raw/_index.json`
for the indexer, and regenerates `corpus/MANIFEST.md` with byte counts and SHA-256
hashes. The raw PDFs are gitignored, so each indexing environment must fetch them.

## Scanned source

The preserved August 2023 rules PDF has 26 pages but only 243 bytes of extractable
text; its policy body is image-based. A page-level OCR spot check recovered academic
rules text. The indexer's PDF extractor OCRs pages with fewer than 150 text
characters by default. Review OCR output for clause numbers, CGPA thresholds, and
exceptions before relying on historical passages.

## Corpus coverage

- The 2026 academic rules provide numbered clauses, including clause 5.4 and the
  2.00 CGPA academic-warning threshold.
- The 2026 fee sheet has fee and refund tables; the computing prospectus has
  course-code, credit-hour, and prerequisite tables (including `CS1002`).
- The Student Handbook includes multi-part definitions of cheating, plagiarism,
  and harassment.
- The set does not contain current campus shuttle stops or departure times. The
  annual report's general transport discussion is not a route timetable.

See [docs/CORPUS.md](../docs/CORPUS.md) for evidence and [MANIFEST.md](MANIFEST.md)
for the verified file inventory.
