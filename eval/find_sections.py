#!/usr/bin/env python3
"""List the (source, section) pairs the indexer produces, to write eval/golden.yaml against.

    python eval/find_sections.py                       # every section of every document
    python eval/find_sections.py --grep "academic warning" --grep CGPA

The golden set must name sections exactly as the indexer stores them. This prints those names
(and where a phrase occurs), so you read the document, then copy the section string. Write the
golden set BEFORE tuning chunk size or the relevance floor (section 5.1).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))

from pipeline.chunk import chunk  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.run import discover  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "raw"))
    ap.add_argument("--grep", action="append", default=[], help="case-insensitive phrase")
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--overlap", type=int, default=50)
    a = ap.parse_args()
    for path, version, superseded in discover(a.corpus):
        doc = clean(extract(str(path), version, superseded))
        chunks = chunk(doc, a.size, a.overlap)
        tag = "SUPERSEDED" if superseded else "current"
        print(f"\n=== {path.name}  (doc_version {version}, {tag}, {len(doc.pages)} pages, "
              f"{sum(p.ocr for p in doc.pages)} OCR)")
        if a.grep:
            for c in chunks:
                for term in a.grep:
                    m = re.search(re.escape(term), c.text, re.I)
                    if m:
                        s = max(0, m.start() - 60)
                        print(f"  [{term}] section={c.section!r} page={c.page}\n"
                              f"      ...{c.text[s:m.end() + 80]!r}")
        else:
            seen: dict[str, int] = {}
            for c in chunks:
                seen[c.section] = seen.get(c.section, 0) + 1
            for section, n in seen.items():
                print(f"  {section!r}  ({n} chunk{'s' if n > 1 else ''})")


if __name__ == "__main__":
    main()
