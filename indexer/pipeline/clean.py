"""Stage 2: Clean. Remove artefacts without destroying structure."""

from __future__ import annotations

import re

from .models import Document


def clean(doc: Document) -> Document:
    """Strip headers and footers that repeat on every page; keep headings and
    section boundaries, because chunking and citations depend on them."""
    # Remove extraction artifacts and repeated page furniture while retaining headings,
    # numbered clauses, tables, and page boundaries.
    texts = [re.sub(r"[\t\u00a0 ]+", " ", p.text).replace("\x00", "").strip()
             for p in doc.pages]
    counts: dict[str, int] = {}
    for text in texts:
        for line in set(text.splitlines()):
            normalized = line.strip()
            if len(normalized) < 4 or len(normalized) > 120:
                continue
            counts[normalized] = counts.get(normalized, 0) + 1
    repeated = {line for line, count in counts.items() if count >= max(2, len(texts) * 0.6)}
    for page, text in zip(doc.pages, texts):
        page.text = "\n".join(line for line in text.splitlines()
                               if line.strip() not in repeated).strip()
    return doc
