"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on sections, paragraphs, and sentence boundaries, keeping each chunk under
    `size` whitespace-delimited words, with `overlap` words shared between neighbours. Every chunk keeps
    the section heading it sits under, its page, doc_version and superseded."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    result: list[Chunk] = []
    # Word tokens make configuration portable without a model tokenizer.
    for page in doc.pages:
        section = ""
        paragraphs: list[tuple[str, str]] = []
        for block in re.split(r"\n\s*\n", page.text):
            block = block.strip()
            if not block:
                continue
            heading = re.match(r"^(#{1,6}\s+.+|(?:\d+(?:\.\d+)*[.)]?\s+).{2,100})$", block)
            if heading and len(block.split()) <= 18:
                section = re.sub(r"^#+\s*", "", block).strip()
            else:
                paragraphs.append((section or "General", block))
        for section, paragraph in paragraphs:
            step = size - overlap
            sentences = [part.split() for part in
                         re.split(r"(?<=[.!?])\s+|\n+", paragraph) if part.split()]
            parts: list[list[str]] = []
            current: list[str] = []
            for sentence in sentences:
                if len(sentence) > size:
                    if current:
                        parts.append(current)
                        current = []
                    for start in range(0, len(sentence), step):
                        parts.append(sentence[start:start + size])
                    continue
                if current and len(current) + len(sentence) > size:
                    parts.append(current)
                    tail = current[-overlap:] if overlap else []
                    current = tail + sentence if len(tail) + len(sentence) <= size else sentence.copy()
                else:
                    current.extend(sentence)
            if current:
                parts.append(current)
            offset = 0
            for part in parts:
                text = " ".join(part)
                digest = hashlib.sha1(
                    f"{doc.source}:{page.page}:{section}:{offset}:{len(result)}:{text}".encode()
                ).hexdigest()[:16]
                result.append(Chunk(digest, text, doc.source, section, page.page,
                                    doc.doc_version, doc.superseded))
                offset += max(1, len(part) - overlap)
    return result
