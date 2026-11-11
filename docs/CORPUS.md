# Corpus

**Source:** official FAST-NUCES public documents. The local corpus contains six
verified PDFs with **260 pages total**. `corpus/fetch.py` writes their local
index to `corpus/raw/_index.json`; `corpus/MANIFEST.md` records each source URL,
version, file size, and SHA-256. The PDFs are gitignored and must be fetched in
each environment before indexing.

## Verified documents

| Document | Version | Pages | Local file | Source |
| --- | --- | ---: | --- | --- |
| Undergraduate Academic Rules and Regulations | 2023-08 (superseded) | 26 | `undergraduate-academic-rules-and-regulations-august-2023.pdf` | [FAST-NUCES PDF](http://nu.edu.pk/public/downloads/Academic-Rules-Undergrad-Aug-2023.pdf) |
| Undergraduate Academic Rules and Regulations | 2026-06 (current in this corpus) | 58 | `undergraduate-academic-rules-and-regulations-june-2026.pdf` | [FAST-NUCES PDF](https://nu.edu.pk/public/downloads/Academic_Rules_Handbook_2026.pdf) |
| Student Handbook | 2026-06 | 64 | `student-handbook-june-2026.pdf` | [FAST-NUCES PDF](https://nu.edu.pk/public/downloads/Student_Handbook_2026.pdf) |
| FAST-NUCES Annual Report | 2023-24 | 85 | `fast-nuces-annual-report-2023-2024.pdf` | [FAST-NUCES PDF](https://nu.edu.pk/public/downloads/AnnualReportForm/Annual%20Report%202023-24.pdf?261108=) |
| Fee Information (Prospectus) | 2026-27 | 1 | `fee-information-prospectus-2026.pdf` | [FAST-NUCES PDF](https://www.nu.edu.pk/Admissions/ProspectusUpload/Fee%20Information.pdf) |
| Computing Programs (Prospectus) | 2026 | 26 | `computing-programs-prospectus-2026.pdf` | [FAST-NUCES PDF](https://www.nu.edu.pk/Admissions/ProspectusUpload/Computing%20Programs.pdf) |

The two 2026 handbooks are listed as June 2026 editions on the [official FAST-NUCES Downloads page](https://www.nu.edu.pk/Downloads). Page counts were checked with `pdfinfo`; all six files have a PDF header, a successful `pdfinfo` read, and recorded SHA-256 hashes in the manifest. The existing 2023 file remains byte-for-byte unchanged.

## Evidence for the seven corpus requirements

| # | Requirement | Evidence |
| --- | --- | --- |
| 1 | At least 6 genuine documents and 40 pages | Six official PDFs, 260 pages in total: 26 + 58 + 64 + 85 + 1 + 26. All six were downloaded from FAST-NUCES URLs and validated as PDFs; their names, versions, byte sizes, source URLs, and SHA-256 hashes are in `corpus/MANIFEST.md`. |
| 2 | Old and new versions of one policy | The 2023 and June 2026 Undergraduate Academic Rules & Regulations are successive editions of the same policy. The FAST downloads page lists the June 2026 edition. The 2023 item is `superseded: true`; the 2026 item is `superseded: false` in `corpus/sources.yaml` and `corpus/raw/_index.json`. |
| 3 | Scanned/image-only PDF | The preserved 2023 rules PDF is 26 pages and `pdftotext` extracts only 243 bytes, consisting mainly of cover/footer text; `pdfimages -list` shows page images while the policy body has no usable text layer. `pdffonts` reports only three font records, insufficient to represent the policy body. A Tesseract spot check of PDF page 8 recovered numbered rules. The indexer's extractor OCRs pages with fewer than 150 text characters; no indexing run was performed. |
| 4 | Informative table | The one-page 2026 Fee Information PDF contains a miscellaneous fee table with 12 rows and a fee-refund schedule with 6 rows; `pdfplumber` extracted both, including amounts such as Rs. 5,000 and the percentage schedule. The 2026 Computing Programs PDF also contains course-code/credit-hour/prerequisite tables; `CS1002` appears in the extracted table. |
| 5 | Long, multi-part definitions that a fixed window could split | The 2026 Student Handbook defines cheating, plagiarism, lying, stealing, and harassment in structured subparts (PDF pages 12–14). The cheating list starts on PDF page 12 and continues onto page 13; item (b) itself breaks across the page boundary after “copying from another.” The harassment definition has five lettered elements. A fixed window can split these blocks and separate a definition from one or more of its cases. |
| 6 | Realistic topic deliberately absent | Current campus shuttle stop-by-stop routes and departure/pickup times are absent. A student could reasonably ask when a bus leaves a particular stop. Searches of extracted text from all selectable-text PDFs and OCR text from all 26 pages of the 2023 scan found no shuttle timetable or departure-time details. The annual report mentions transport generally, but gives no such operational timetable. |
| 7 | Exact searchable terms | The Computing Programs table contains `CS1002` (Programming Fundamentals, PDF page 4). The 2026 academic rules contain clause `5.4` and the exact academic-warning threshold `CGPA ... 2.00` (PDF page 24; printed page 18). The 2026 Fee Information PDF states `Rs. 12,000 per credit hour` (PDF page 1). These exact strings were confirmed in extracted PDF text/table output. |

## Verification notes

- The five new corpus PDFs were downloaded and checked for status, PDF signature,
  page count, text extraction, and embedded fonts. SHA-256 values are in the
  manifest. `python corpus/fetch.py --verify` checks the local bytes against it.
- The Annual Report includes campus/program/enrollment tables and general
  transport references. It is contextual evidence, not a current timetable or
  policy source.
- The eight-page ORIC PDF at `OricResources/ResearchPolicies/27.pdf` was also
  inspected, but it is a scanned faculty entrepreneurship policy with no text
  layer and was not added: it is less relevant to the student academic corpus,
  and it is not needed to satisfy the requirements.
- The indexer has not been run. OCR accuracy for all pages of the 2023 scan has
  not been manually proofread; only a page-level OCR spot check was made.
