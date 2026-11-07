"""Stage 3: Chunk. Structure-aware, with overlap."""

from __future__ import annotations

import hashlib
import re

from .models import Chunk, Document


EXCLUDED_PAGES = {1, 2, 3, 4, 5, 26}
EXCLUDED_SOURCES = {
    "undergraduate-academic-rules-and-regulations-august-2023.pdf",
    "rahbar-rules-ocr.pdf",
}


def _is_heading_line(line: str) -> bool:
    """Recognize standalone section headings, not numbered policy clauses."""
    line = line.strip()
    if not line or len(line) > 100 or len(line.split()) > 12:
        return False

    # Markdown headings.
    if re.match(r"^#{1,6}\s+", line):
        return True

    # Numbered section headings, e.g. "1. Introduction".
    # Decimal clauses such as "1.21 Students..." are deliberately excluded.
    if re.match(r"^\d{1,2}\.\s+[A-Za-z]", line):
        return True

    if re.match(r"^\d+\.\d+", line):
        return False

    # Standalone all-uppercase headings.
    letters = [char for char in line if char.isalpha()]
    if letters and all(char.isupper() for char in letters):
        return True

    # Short title-case headings, e.g. "Attendance Requirements".
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", line)
    if not words or len(words) > 7 or line[-1] in ".!?;:":
        return False

    small_words = {"and", "or", "of", "the", "for", "in", "to", "a", "on"}
    title_case = all(
        word[0].isupper() or word.lower() in small_words
        for word in words
    )
    return title_case


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split pages into overlapping chunks while retaining section metadata."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")

    result: list[Chunk] = []
    step = size - overlap

    for page in doc.pages:
        source_name = doc.source.replace("\\", "/").split("/")[-1].lower()
        if source_name in EXCLUDED_SOURCES and page.page in EXCLUDED_PAGES:
            continue

        section = "General"
        body_lines: list[str] = []

        def flush_body() -> None:
            if not body_lines:
                return

            body = " ".join(line.strip() for line in body_lines if line.strip())
            body_lines.clear()
            if not body:
                return

            sentences = [
                part.split()
                for part in re.split(r"(?<=[.!?])\s+", body)
                if part.split()
            ]

            parts: list[list[str]] = []
            current: list[str] = []

            for sentence in sentences:
                if len(sentence) > size:
                    if current:
                        parts.append(current)
                        current = []
                    for start in range(0, len(sentence), step):
                        parts.append(sentence[start:start + size])
                elif current and len(current) + len(sentence) > size:
                    parts.append(current)
                    tail = current[-overlap:] if overlap else []
                    current = tail + sentence
                else:
                    current.extend(sentence)

            if current:
                parts.append(current)

            offset = 0
            for part in parts:
                text = " ".join(part)
                digest = hashlib.sha1(
                    f"{doc.identity or doc.source}:{doc.doc_version}:{doc.superseded}:"
                    f"{page.page}:{section}:{offset}:{text}".encode()
                ).hexdigest()[:16]

                result.append(
                    Chunk(
                        digest,
                        text,
                        doc.source,
                        section,
                        page.page,
                        doc.doc_version,
                        doc.superseded,
                    )
                )
                offset += max(1, len(part) - overlap)

        for line in page.text.splitlines():
            line = line.strip()
            if not line:
                continue

            if _is_heading_line(line):
                flush_body()
                section = re.sub(r"^#{1,6}\s*", "", line).strip()
            else:
                body_lines.append(line)

        flush_body()

    return result
