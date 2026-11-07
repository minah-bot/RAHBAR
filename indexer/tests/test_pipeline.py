from pipeline.chunk import chunk
from pipeline.clean import clean
from pipeline.extract import extract
from pipeline.models import Document, Page


def test_extract_markdown_strips_fixture_front_matter(tmp_path):
    path = tmp_path / "rules.md"
    path.write_text("---\ndoc_version: 2026\nsuperseded: false\n---\n# 1 Attendance\n\n80 percent required.")
    result = extract(str(path), "2026", False)
    assert result.pages[0].text.startswith("# 1 Attendance")


def test_clean_removes_repeated_page_furniture_but_keeps_content():
    doc = Document("x.pdf", "v1", False, [
        Page("x.pdf", 1, "University Handbook\nHeading\nUseful text"),
        Page("x.pdf", 2, "University Handbook\nOther heading\nMore useful text"),
    ])
    clean(doc)
    assert all("University Handbook" not in page.text for page in doc.pages)
    assert "Useful text" in doc.pages[0].text


def test_chunks_overlap_and_preserve_metadata():
    doc = Document("x.md", "2026", False,
                   [Page("x.md", 4, "# 2 Policy\n\n" + " ".join(f"w{i}" for i in range(20)))])
    chunks = chunk(doc, size=8, overlap=2)
    assert len(chunks) > 1
    assert chunks[0].section == "2 Policy"
    assert chunks[0].page == 4
    assert chunks[0].doc_version == "2026"
    assert set(chunks[0].text.split()[-2:]) & set(chunks[1].text.split())


def test_chunking_rejects_invalid_overlap():
    try:
        chunk(Document("x", "v", False), size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("invalid overlap should fail")


def test_sparse_pdf_text_requires_ocr():
    from pipeline.extract import _needs_ocr

    assert _needs_ocr("")
    assert _needs_ocr("A short page heading")
    assert not _needs_ocr("Useful policy text. " * 20)


def test_ocr_threshold_covers_sparse_cover_page():
    from pipeline.extract import _needs_ocr

    assert _needs_ocr("UNDERGRADUATE ACADEMIC RULES AND REGULATIONS 2023")
