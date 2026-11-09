"""Token counting. One definition, used by the chunker, the amplification report and the sweep.

A "token" here is a word or a punctuation mark (regex `\\w+|[^\\w\\s]`). It is not the model's
tokenizer, but it is cheap, deterministic and within ~25% of nomic's WordPiece count on English
prose. CHUNK_SIZE and CHUNK_OVERLAP are in these tokens.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count(text: str) -> int:
    return len(_TOKEN.findall(text))


def words(text: str) -> list[str]:
    return text.split()
