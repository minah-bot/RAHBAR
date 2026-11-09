"""Indexer stage tests: no Chroma, no model, no PDF fixtures needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.chunk import chunk, heading_of
from pipeline.clean import clean
from pipeline.extract import _html_to_text, extract
from pipeline.models import Document, Page
from pipeline.store import store
from pipeline.tokens import count

FIXTURES = Path(__file__).resolve().parents[2] / "eval" / "fixtures"


def doc(*texts: str) -> Document:
    return Document("d.pdf", "2026", False,
                    [Page("d.pdf", i + 1, t) for i, t in enumerate(texts)])


def test_markdown_front_matter_is_not_indexed() -> None:
    d = extract(str(FIXTURES / "academic-rules-2026.md"), "2026", False)
    assert "doc_version" not in d.pages[0].text and "# Margalla" in d.pages[0].text


def test_chunks_never_cross_a_heading_and_keep_metadata() -> None:
    d = extract(str(FIXTURES / "academic-rules-2023.md"), "2023", True)
    cs = chunk(clean(d), 120, 20)
    sections = {c.section for c in cs}
    assert {"1. Attendance", "2. Academic warning", "3. Repeating a course"} <= sections
    for c in cs:
        assert c.superseded is True and c.doc_version == "2023" and c.source.endswith(".md")
        if c.section == "1. Attendance":
            assert "75 percent" in c.text and "2.1" not in c.text


def test_ids_are_stable_and_unique() -> None:
    d = extract(str(FIXTURES / "fees-2026.md"), "2026", False)
    a, b = chunk(d, 120, 20), chunk(d, 120, 20)
    assert [c.id for c in a] == [c.id for c in b] and len({c.id for c in a}) == len(a)


def test_size_limit_and_overlap_between_neighbours() -> None:
    para = " ".join(f"Sentence number {i} says something about the rule." for i in range(40))
    cs = chunk(doc("# Rules\n\n" + para), 60, 10)
    assert len(cs) > 3 and all(count(c.text) <= 60 + 10 + 12 for c in cs)
    first_tail = cs[0].text.split()[-5:]
    assert " ".join(first_tail) in cs[1].text           # overlap carried forward


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk(doc("# A\n\ntext"), 10, 10)


def test_long_table_splits_by_rows_and_repeats_the_header() -> None:
    rows = "\n".join(f"| Course {i} | {i * 1000} |" for i in range(60))
    table = "| Course | Fee |\n| --- | --- |\n" + rows
    cs = chunk(doc("# Fees\n\n" + table), 80, 0)
    assert len(cs) > 1
    for c in cs:
        assert c.text.startswith("| Course | Fee |")           # header repeated
        assert all(line.startswith("|") for line in c.text.splitlines())   # no cut mid-row


def test_headings_detected() -> None:
    assert heading_of("5.4 Academic Warning") == "5.4 Academic Warning"
    assert heading_of("## 2. Academic warning") == "2. Academic warning"
    assert heading_of("1.1 A student must attend at least 80 percent of lectures.") is None
    assert heading_of("| a | b |") is None


def test_clean_removes_repeated_headers_and_page_numbers_but_keeps_clauses() -> None:
    pages = [f"FAST-NUCES Academic Rules\nIntro {i}\n5.4 Academic Warning\nBody {i}\nMore {i}\n"
             f"Page {i}" for i in range(6)]
    out = clean(doc(*pages))
    for p in out.pages:
        assert "FAST-NUCES Academic Rules" not in p.text and "Page" not in p.text
        assert "5.4 Academic Warning" in p.text         # clause number deliberately kept


def test_clean_rejoins_hyphenated_words() -> None:
    assert "registration" in clean(doc("regis-\ntration is open")).pages[0].text


def test_html_table_becomes_markdown() -> None:
    text = _html_to_text("<h2>Fees</h2><table><tr><th>Item</th><th>PKR</th></tr>"
                         "<tr><td>Late</td><td>3,000</td></tr></table><script>x()</script>")
    assert "## Fees" in text and "| Late | 3,000 |" in text and "x()" not in text


# -- store: blue/green swap against a fake client ----------------------------------
class Col:
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        self.name, self.metadata, self.rows = name, metadata or {}, {}

    def add(self, ids, documents, embeddings, metadatas) -> None:  # type: ignore[no-untyped-def]
        for i, d, m in zip(ids, documents, metadatas, strict=True):
            self.rows[i] = (d, m)

    def upsert(self, ids, documents, embeddings, metadatas) -> None:  # type: ignore[no-untyped-def]
        self.rows[ids[0]] = (documents[0], metadatas[0])

    def get(self, ids):  # type: ignore[no-untyped-def]
        hit = self.rows.get(ids[0])
        return {"documents": [hit[0]] if hit else [], "metadatas": [hit[1]] if hit else []}

    def count(self) -> int:
        return len(self.rows)


class Client:
    def __init__(self) -> None:
        self.cols: dict[str, Col] = {}

    def create_collection(self, name, metadata):  # type: ignore[no-untyped-def]
        self.cols[name] = Col(name, metadata)
        return self.cols[name]

    def get_or_create_collection(self, name):  # type: ignore[no-untyped-def]
        return self.cols.setdefault(name, Col(name))

    def get_collection(self, name):  # type: ignore[no-untyped-def]
        return self.cols[name]

    def list_collections(self):  # type: ignore[no-untyped-def]
        return list(self.cols.values())

    def delete_collection(self, name: str) -> None:
        del self.cols[name]


def test_store_records_the_pin_and_swaps_the_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipeline.store as st
    from pipeline.config import get_settings

    client = Client()
    monkeypatch.setattr(st, "get_client", lambda s: client)
    monkeypatch.setattr(st.time, "strftime", lambda f, t: str(len(client.cols)))
    d = extract(str(FIXTURES / "fees-2026.md"), "2026", False)
    cs = chunk(d, 120, 20)
    vecs = [[0.1, 0.2, 0.3]] * len(cs)
    s = get_settings()
    assert store(cs, vecs, s) == len(cs)
    active = client.cols["rahbar-active"].rows["active"][0]
    meta = client.cols[active].metadata
    assert meta["embed_model"] == s.embed_model and meta["dimension"] == 3
    store(cs, vecs, s)                                   # second run: new index, old kept
    store(cs, vecs, s)                                   # third run: oldest deleted
    names = [n for n in client.cols if n != "rahbar-active"]
    assert len(names) == 2 and client.cols["rahbar-active"].rows["active"][0] in names


def test_store_refuses_an_empty_index() -> None:
    from pipeline.config import get_settings

    with pytest.raises(ValueError):
        store([], [], get_settings())


def test_fixed_strategy_is_the_naive_baseline() -> None:
    from pipeline.chunk import chunk_fixed

    cs = chunk_fixed(doc("# A\n\n" + " ".join(["w"] * 250)), 100)
    assert [len(c.text.split()) for c in cs] == [100, 100, 52] and all(c.section == "" for c in cs)
