"""Stage 3: Chunk. Structure-aware, with overlap.

Order of preference when splitting: sections (headings), then paragraphs / clauses, then
sentences, then words as a last resort. A chunk never crosses a heading. Tables are kept whole
when they fit, otherwise split by rows with the header row repeated, so a fee table is never
cut mid-row. Overlap is carried between neighbouring chunks of the same section.
`size` and `overlap` are in tokens (see tokens.py).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .models import Chunk, Document
from .tokens import count

_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,4}\.?)\s+([A-Z][^\n]{1,70})$")
_CAPS = re.compile(r"^[A-Z][A-Z0-9 ,&/()'\-]{3,70}$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])")


@dataclass
class _Unit:
    text: str
    page: int


def heading_of(line: str) -> str | None:
    """Return the heading text if the line is one, else None."""
    s = line.strip()
    m = _MD_HEADING.match(s)
    if m:
        return m.group(1)
    if s.startswith("|"):
        return None
    m = _NUMBERED.match(s)
    if m and not s.endswith((".", ",", ";", ":")) and len(s.split()) <= 12:
        return s
    if _CAPS.match(s) and not s.replace(" ", "").isdigit() and len(s.split()) <= 10:
        return s
    return None


def _split_sections(doc: Document) -> list[tuple[str, list[_Unit]]]:
    """[(section heading, [paragraph-level units])] in reading order."""
    sections: list[tuple[str, list[_Unit]]] = [("Preamble", [])]
    for page in doc.pages:
        buf: list[str] = []

        def flush(buf: list[str] = buf, page_no: int = page.page) -> None:
            if buf:
                sections[-1][1].append(_Unit("\n".join(buf).strip(), page_no))
                buf.clear()

        for line in page.text.split("\n"):
            h = heading_of(line)
            if h is not None:
                flush()
                sections.append((h, []))
            elif not line.strip():
                flush()
            elif re.match(r"^\d+(\.\d+)+\s", line) and buf and not buf[-1].startswith("|"):
                flush()   # a new numbered clause starts its own paragraph
                buf.append(line)
            else:
                buf.append(line)
        flush()
    return [(h, [u for u in units if u.text]) for h, units in sections if units]


def _split_long(unit: _Unit, size: int) -> list[_Unit]:
    """Break an over-long unit: table rows first, then sentences, then words."""
    lines = unit.text.split("\n")
    if all(ln.startswith("|") for ln in lines) and len(lines) > 3:
        header, rows = lines[:2], lines[2:]
        out, cur = [], list(header)
        for row in rows:
            if count("\n".join([*cur, row])) > size and len(cur) > 2:
                out.append(_Unit("\n".join(cur), unit.page))
                cur = [*header]
            cur.append(row)
        out.append(_Unit("\n".join(cur), unit.page))
        return out
    pieces = [s for s in _SENTENCE.split(unit.text) if s.strip()]
    out = []
    for piece in pieces:
        if count(piece) <= size:
            out.append(_Unit(piece, unit.page))
        else:
            ws = piece.split()
            step = max(size // 2, 1)
            for i in range(0, len(ws), step * 2 or 1):
                out.append(_Unit(" ".join(ws[i:i + step * 2]), unit.page))
    return out


def _chunk_id(doc: Document, seq: int, page: int, section: str, text: str) -> str:
    """`<document>-<NNNN>-<hash>`. The hash covers the document's identity, version and the chunk's
    page, section and text, so an unchanged chunk keeps its id across re-indexes (its vector can be
    reused), a changed one gets a new id, and same-named files in different folders never collide."""
    digest = hashlib.sha1(
        f"{doc.identity or doc.source}\0{doc.doc_version}\0{doc.superseded}\0{page}\0"
        f"{section}\0{text}".encode()).hexdigest()[:8]
    return f"{_slug(doc.source.rsplit('.', 1)[0])}-{seq:04d}-{digest}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def chunk(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """Split on sections, then paragraphs, then sentences, keeping each chunk under
    `size` tokens, with `overlap` tokens shared between neighbours. Every chunk keeps
    the section heading it sits under, its page, doc_version and superseded."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for heading, units in _split_sections(doc):
        flat: list[_Unit] = []
        for u in units:
            flat.extend(_split_long(u, size) if count(u.text) > size else [u])
        cur: list[_Unit] = []
        cur_tokens = 0
        tail = ""
        for u in flat + [_Unit("", 0)]:
            n = count(u.text)
            if cur and (not u.text or cur_tokens + n > size):
                body = "\n\n".join(x.text for x in cur)
                text = (tail + "\n\n" + body).strip() if tail else body
                chunks.append(Chunk(
                    id=_chunk_id(doc, len(chunks), cur[0].page, heading, text),
                    text=text, source=doc.source,
                    section=heading, page=cur[0].page, doc_version=doc.doc_version,
                    superseded=doc.superseded))
                ws = body.split()
                tail = " ".join(ws[-overlap:]) if overlap and not body.startswith("|") else ""
                cur, cur_tokens = [], 0
            if u.text:
                cur.append(u)
                cur_tokens += n
    return chunks


def chunk_fixed(doc: Document, size: int) -> list[Chunk]:
    """The naive baseline: fixed windows of `size` tokens, no overlap, no sections. Used only
    to reproduce Stage B of docs/FAILURE-ANALYSIS.md (CHUNK_STRATEGY=fixed); never the default."""
    chunks: list[Chunk] = []
    for page in doc.pages:
        ws = page.text.split()
        for i in range(0, len(ws), size):
            chunks.append(Chunk(
                id=_chunk_id(doc, len(chunks), page.page, "", " ".join(ws[i:i + size])),
                text=" ".join(ws[i:i + size]),
                source=doc.source, section="", page=page.page, doc_version=doc.doc_version,
                superseded=doc.superseded))
    return chunks
