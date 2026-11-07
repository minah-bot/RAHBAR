"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page."""

from __future__ import annotations

import re
from pathlib import Path

from .models import Document, Page

MIN_TEXT_CHARS = 150


def _needs_ocr(text: str) -> bool:
    """OCR pages with no text layer or an unusually small text layer."""
    return len(text.strip()) < MIN_TEXT_CHARS


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Pages with missing or insufficient text are OCR'd.
    OCR requires tesseract-ocr and poppler-utils system packages.
    """
    file = Path(path)
    suffix = file.suffix.lower()

    if suffix in {".md", ".txt"}:
        text = file.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
        return Document(file.name, doc_version, superseded, [Page(file.name, 1, text)])

    if suffix == ".html":
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            parts: list[str]
            hidden = 0

            def __init__(self) -> None:
                super().__init__()
                self.parts = []

            def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                if tag in {"script", "style", "noscript"}:
                    self.hidden += 1

            def handle_endtag(self, tag: str) -> None:
                if tag in {"script", "style", "noscript"} and self.hidden:
                    self.hidden -= 1

            def handle_data(self, data: str) -> None:
                if not self.hidden:
                    self.parts.append(data)

        parser = TextExtractor()
        parser.feed(file.read_text(encoding="utf-8", errors="replace"))
        return Document(file.name, doc_version, superseded,
                        [Page(file.name, 1, "\n".join(parser.parts))])

    if suffix != ".pdf":
        raise ValueError(f"unsupported document type: {suffix}")

    import pdfplumber

    pages: list[Page] = []
    with pdfplumber.open(file) as pdf:
        for number, page in enumerate(pdf.pages, 1):
            text = page.extract_text(layout=False) or ""
            ocr = _needs_ocr(text)

            if ocr:
                try:
                    import pytesseract
                    from pdf2image import convert_from_path

                    image = convert_from_path(
                        str(file), first_page=number, last_page=number, dpi=200
                    )[0]
                    text = pytesseract.image_to_string(image)
                except Exception as exc:
                    raise RuntimeError(
                        f"page {number} has no usable text and OCR failed: {exc}"
                    ) from exc

            pages.append(Page(file.name, number, text, ocr))

    return Document(file.name, doc_version, superseded, pages)
