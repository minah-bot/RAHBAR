# Corpus

## Current state

The only available corpus in this checkout is the three-document fictional evaluation
fixture in `eval/fixtures/`. It contains 2023 and 2026 academic rules and a 2026 fee
schedule. It is suitable for local integration testing, but it is not a FAST-NUCES corpus
and does not meet the assignment's six-document / forty-page collection requirement.
No public documents have been downloaded into this repository; no page counts, checksums,
or scanned-document evidence are asserted. `corpus/fetch.py` and
`corpus/sources.example.yaml` are provided to fetch approved source material. Keep fetched
PDFs in ignored `corpus/raw/` and commit the generated, verified `corpus/MANIFEST.md`.

| # | Required property | Evidence currently available | Status / student action |
| --- | --- | --- | --- |
| 1 | At least 6 documents, 40 pages | Three short Markdown fixtures | Incomplete; select and verify at least six source documents and 40 pages. |
| 2 | Two versions of one policy | `academic-rules-2023.md` is superseded by `academic-rules-2026.md` | Fixture only; front matter sets `doc_version` and `superseded`. |
| 3 | Scanned, image-only PDF | No scanned PDF in the checkout | Incomplete; include an authorized scan and record text-layer/OCR evidence. |
| 4 | A table with real information | The fictional fee table in `fees-2026.md` | Fixture demonstrates table extraction, not real university information. |
| 5 | Long definitions | No long definition in the fixtures | Incomplete; identify one in the selected corpus. |
| 6 | Deliberately absent topic | Hostel application and library hours are absent from fixtures | The evaluation uses these plausible must-abstain questions. |
| 7 | Exact-term content | Fixture has 80%, 2.00 CGPA, FA and PKR fee amounts | Fixture only; add verified source clauses, codes and thresholds. |

Before submission, document source URLs, versions, page counts, SHA-256 checksums, and
permissions in the generated corpus manifest. Do not treat fixture data as university policy.
