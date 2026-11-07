"""Does the user explicitly ask about an older version of a policy? (section 3.2, step 4)

By default superseded documents are filtered out in the Chroma query itself. This is the
only door back in. A heuristic, deliberately narrow: a keyword that means "not the current
one", or a year before the current edition. It is tested in tests/test_version.py.
"""

from __future__ import annotations

import re

CURRENT_EDITION_YEAR = 2026

_KEYWORDS = re.compile(
    r"\b(old|older|previous|previously|prior|earlier|former|formerly|superseded|repealed|"
    r"outdated|archived|used to be|was replaced)\b",
    re.I,
)
_YEAR = re.compile(r"\b(20[0-2]\d)\b")


def wants_older_version(question: str) -> bool:
    if _KEYWORDS.search(question):
        return True
    return any(int(y) < CURRENT_EDITION_YEAR for y in _YEAR.findall(question))
