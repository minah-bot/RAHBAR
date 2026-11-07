"""Stage 1: Extract. PDF, HTML, Markdown or text to text, page by page.

Tables become Markdown rows (`| a | b |`) so columns stay columns, instead of the jumbled
"a b c d" a plain text dump gives. Pages with no text layer are OCR'd and marked ocr=True.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from pathlib import Path

from .models import Document, Page

MIN_TEXT_CHARS = 25     # a page with fewer characters than this has no usable text layer
OCR_DPI = 200


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    p = Path(path)
    name = p.name
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages = _pdf_pages(p)
    elif suffix in (".html", ".htm"):
        pages = [Page(source=name, page=1, text=_html_to_text(p.read_text(encoding="utf-8",
                                                                          errors="replace")))]
    elif suffix in (".md", ".txt"):
        pages = [Page(source=name, page=1, text=_strip_front_matter(
            p.read_text(encoding="utf-8")))]
    else:
        raise ValueError(f"unsupported file type: {name}")
    return Document(source=name, doc_version=doc_version, superseded=superseded, pages=pages)


# -- Markdown ---------------------------------------------------------------------
def _strip_front_matter(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:]).lstrip("\n")
    return text


# -- PDF --------------------------------------------------------------------------
def _md_table(rows: list[list[str | None]]) -> str:
    clean = [[(c or "").replace("\n", " ").strip() for c in row] for row in rows if row]
    clean = [r for r in clean if any(r)]
    if len(clean) < 2:
        return ""
    width = max(len(r) for r in clean)
    clean = [r + [""] * (width - len(r)) for r in clean]
    out = ["| " + " | ".join(clean[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in clean[1:]]
    return "\n".join(out)


def _page_text_and_tables(page: object) -> str:
    """Text outside tables, then each table as Markdown. Falls back to plain text."""
    import pdfplumber  # noqa: F401  (imported for the type of `page`)

    try:
        tables = page.find_tables()  # type: ignore[attr-defined]
        rendered = []
        outside = page
        for t in tables:
            md = _md_table(t.extract())
            if md:
                rendered.append(md)
                outside = outside.outside_bbox(t.bbox)  # type: ignore[attr-defined]
        text = (outside.extract_text() or "")  # type: ignore[attr-defined]
        return "\n\n".join([text.strip(), *rendered]).strip()
    except Exception:
        return (page.extract_text() or "").strip()  # type: ignore[attr-defined]


def _ocr_page(path: Path, number: int) -> str:
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(str(path), dpi=OCR_DPI, first_page=number, last_page=number)
    return "\n".join(pytesseract.image_to_string(img) for img in images).strip()


def _pdf_pages(path: Path) -> list[Page]:
    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = _page_text_and_tables(page)
            ocr = False
            # OCR_ENABLED=false exists only to reproduce Stage A of docs/FAILURE-ANALYSIS.md.
            if len(text) < MIN_TEXT_CHARS and os.environ.get("OCR_ENABLED", "true") != "false":
                text, ocr = _ocr_page(path, i), True
            pages.append(Page(source=path.name, page=i, text=text, ocr=ocr))
    return pages


# -- HTML -------------------------------------------------------------------------
class _Html(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "header", "noscript", "svg", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.skip = 0
        self.table: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("p", "li", "br", "div"):
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None and self.table is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.out.append("\n\n" + _md_table(list(self.table)) + "\n\n")  # type: ignore[arg-type]
            self.table = None
        elif re.fullmatch(r"h[1-6]", tag):
            self.out.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        else:
            self.out.append(data)


def _html_to_text(html: str) -> str:
    parser = _Html()
    parser.feed(html)
    text = "".join(parser.out)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()
