"""Stage 2: Clean. Remove artefacts without destroying structure.

Rules (also listed in docs/PIPELINE.md):
 1. Drop lines that repeat in the first/last two lines of many pages (running headers and
    footers), compared with digits masked so "Page 3" and "Page 4" count as the same line.
 2. Drop bare page-number lines ("3", "Page 3", "- 3 -").
 3. Re-join words hyphenated across a line break ("regis-\\ntration").
 4. Remove control characters and collapse runs of spaces and blank lines.
Deliberately NOT stripped: clause numbers ("5.4"), digits inside table cells and short
ALL-CAPS lines. They look like noise but they are the headings and exact terms that chunking
and exact-term retrieval depend on.
"""

from __future__ import annotations

import re
from collections import Counter

from .models import Document, Page

REPEAT_SHARE = 0.4      # a header/footer line appears on at least 40% of pages
MIN_PAGES_FOR_REPEATS = 2

_PAGE_NO = re.compile(r"^\s*(?:page\s*)?[-–]?\s*\d{1,4}\s*(?:of\s*\d{1,4})?\s*[-–]?\s*$", re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u00ad\u200b]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-z])")


def _mask(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def _repeated_lines(pages: list[Page]) -> set[str]:
    if len(pages) < MIN_PAGES_FOR_REPEATS:
        return set()
    seen: Counter[str] = Counter()
    for page in pages:
        lines = [ln for ln in page.text.splitlines() if ln.strip()]
        for ln in {_mask(x) for x in lines[:2] + lines[-2:]}:
            seen[ln] += 1
    limit = max(2, int(len(pages) * REPEAT_SHARE))
    return {ln for ln, n in seen.items() if n >= limit and len(ln) < 120}


def clean(doc: Document) -> Document:
    repeats = _repeated_lines(doc.pages)
    pages: list[Page] = []
    for page in doc.pages:
        text = _CONTROL.sub("", page.text.replace("\r\n", "\n").replace("\r", "\n"))
        lines = text.split("\n")
        edge = {i for i, ln in enumerate(lines) if ln.strip()}
        keep_edges = sorted(edge)
        edge_idx = set(keep_edges[:2] + keep_edges[-2:])
        kept = []
        for i, ln in enumerate(lines):
            if i in edge_idx and _mask(ln) in repeats:
                continue
            if _PAGE_NO.match(ln):
                continue
            kept.append(ln.rstrip())
        text = _HYPHEN_BREAK.sub(r"\1", "\n".join(kept))
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        pages.append(Page(source=page.source, page=page.page, text=text, ocr=page.ocr))
    doc.pages = pages          # cleaned in place, and returned for callers that chain
    return doc
