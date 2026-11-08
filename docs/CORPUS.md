# Corpus

> **TEAM TODO:** this file records *your* corpus. The starter list is `corpus/sources.example.yaml`; several URLs may have
> moved, so every row must be checked by opening the downloaded file. Nothing below is filled in for you on purpose.

Source: FAST-NUCES public website · Documents: ? · Pages: ?  (`make fetch` writes `corpus/raw/_index.json`; the documents
themselves are gitignored: commit `corpus/sources.yaml` and `fetch.py`, not the PDFs.)

| # | Required property | Document(s) that provide it | Evidence or notes |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | | `ls corpus/raw`, `pdfinfo` page counts |
| 2 | Two versions of one policy | | Which supersedes which; `superseded: true` in `corpus/sources.yaml` becomes chunk metadata, and the retriever filters it out unless the question asks about an older version |
| 3 | Scanned, image-only PDF | | `pdffonts file.pdf` printing no fonts, or `pdftotext` returning nothing; the indexer logs `ocr_pages` for it |
| 4 | A table with real information | | |
| 5 | Long definitions | | |
| 6 | A deliberately absent topic | | What you left out and why a user would plausibly ask (`python eval/find_sections.py --grep <topic>` must find nothing) |
| 7 | Exact-term content | | Clause numbers, codes, thresholds, fees |
