from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import IndexerSettings
from pipeline.models import Chunk, Document, Page
from pipeline.run import discover, run_pipeline
from pipeline.store import reuse_embeddings


def settings_for(corpus: Path) -> IndexerSettings:
    return IndexerSettings(
        embed_model="test-embed", chroma_host="localhost", chroma_port=8000,
        collection="rahbar", corpus_dir=str(corpus), chunk_size=100,
        chunk_overlap=10, ollama_url="http://localhost:11434",
    )


def test_manifest_discovery_returns_every_safe_document(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").touch()
    (tmp_path / "b.pdf").touch()
    (tmp_path / "_index.json").write_text(json.dumps([
        {"file": "a.pdf", "doc_version": "2023-08", "superseded": True},
        {"file": "b.pdf", "doc_version": "2026-06", "superseded": "false"},
    ]))

    results = discover(str(tmp_path))
    assert [(path.name, version, old) for path, version, old in results] == [
        ("a.pdf", "2023-08", True), ("b.pdf", "2026-06", False)
    ]


def test_manifest_path_cannot_escape_corpus(tmp_path: Path) -> None:
    (tmp_path / "_index.json").write_text(json.dumps([{"file": "../outside.pdf"}]))
    with pytest.raises(ValueError, match="outside the corpus directory"):
        discover(str(tmp_path))


def test_identical_filenames_in_different_documents_get_distinct_chunk_ids() -> None:
    page = Page("handbook.md", 1, "# Attendance\nAttendance is required.")
    first = Document("handbook.md", "2026", False, [page], identity="campus-a/handbook.md")
    second = Document("handbook.md", "2026", False, [page], identity="campus-b/handbook.md")
    from pipeline.chunk import chunk

    left, right = chunk(first, 100, 10), chunk(second, 100, 10)
    assert left[0].source == right[0].source == "handbook.md"
    assert left[0].id != right[0].id


def test_multi_document_build_preserves_metadata_and_deduplicates_manifest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "north" / "handbook.md"
    second = tmp_path / "south" / "handbook.md"
    first.parent.mkdir()
    second.parent.mkdir()
    text = "# Attendance\n\nAttendance is required for every course session."
    first.write_text(text)
    second.write_text(text)
    entries = [
        {"file": "north/handbook.md", "doc_version": "2026", "superseded": False},
        {"file": "south/handbook.md", "doc_version": "2026", "superseded": False},
        {"file": "north/handbook.md", "doc_version": "2026", "superseded": False},
    ]
    (tmp_path / "_index.json").write_text(json.dumps(entries))

    embedded: list[str] = []
    stored: dict[str, Any] = {}

    def fake_embed(texts: list[str], _settings: IndexerSettings) -> list[list[float]]:
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def fake_store(chunks: list[Chunk], vectors: list[list[float]],
                   _settings: IndexerSettings) -> int:
        stored["chunks"] = chunks
        stored["vectors"] = vectors
        return len(chunks)

    monkeypatch.setattr("pipeline.store.reuse_embeddings", lambda chunks, _settings: {})
    monkeypatch.setattr("pipeline.embed.embed", fake_embed)
    monkeypatch.setattr("pipeline.store.store", fake_store)

    assert run_pipeline(settings_for(tmp_path)) == 2
    chunks = stored["chunks"]
    assert len(chunks) == 2
    assert {chunk.source for chunk in chunks} == {"handbook.md"}
    assert {chunk.doc_version for chunk in chunks} == {"2026"}
    assert {chunk.section for chunk in chunks} == {"Attendance"}
    assert len({chunk.id for chunk in chunks}) == 2
    assert len(embedded) == 2


def test_failed_document_aborts_before_embedding_or_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    valid = tmp_path / "valid.md"
    valid.write_text("# Policy\n\nA valid policy passage with sufficient words.")
    (tmp_path / "_index.json").write_text(json.dumps([
        {"file": "valid.md", "doc_version": "2026", "superseded": False},
        {"file": "missing.pdf", "doc_version": "2026", "superseded": False},
    ]))
    monkeypatch.setattr("pipeline.store.reuse_embeddings", lambda *_args: {})
    monkeypatch.setattr("pipeline.embed.embed", lambda *_args: pytest.fail("must not embed"))
    monkeypatch.setattr("pipeline.store.store", lambda *_args: pytest.fail("must not store"))

    with pytest.raises(RuntimeError, match="missing.pdf"):
        run_pipeline(settings_for(tmp_path))
    assert '"msg": "document failed"' in capsys.readouterr().out


def test_matching_model_reuses_only_existing_chunk_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk_a = Chunk("a", "text a", "a.pdf", "1", 1, "2026", False)
    chunk_b = Chunk("b", "text b", "b.pdf", "1", 1, "2026", False)

    class Collection:
        metadata = {"embed_model": "test-embed", "dimension": 2}

        def get(self, ids: list[str], include: list[str]) -> dict[str, Any]:
            assert include == ["embeddings"]
            requested = [value for value in ids if value == "a"]
            return {"ids": requested, "embeddings": [[0.5, 0.5] for _ in requested]}

    class Client:
        def list_collections(self) -> list[str]:
            return ["rahbar"]

        def get_collection(self, name: str) -> Collection:
            assert name == "rahbar"
            return Collection()

    monkeypatch.setattr("pipeline.store.get_client", lambda _settings: Client())
    result = reuse_embeddings([chunk_a, chunk_b], settings_for(Path("/unused")))
    assert result == {"a": [0.5, 0.5]}


def test_different_embedding_model_never_reuses_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Collection:
        metadata = {"embed_model": "old-model", "dimension": 2}

    class Client:
        def list_collections(self) -> list[str]:
            return ["rahbar"]

        def get_collection(self, name: str) -> Collection:
            return Collection()

    monkeypatch.setattr("pipeline.store.get_client", lambda _settings: Client())
    item = Chunk("a", "text", "a.pdf", "1", 1, "2026", False)
    assert reuse_embeddings([item], settings_for(Path("/unused"))) == {}
