"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, clean, chunk, embed, store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import IndexerSettings, get_settings
from .models import Chunk


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


SUFFIXES = {".pdf", ".md", ".html", ".txt"}


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present, validating every entry (a manifest is input, not trusted). Otherwise read Markdown
    front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError(f"{index} must contain a JSON list of document entries")
        found: list[tuple[Path, str, bool]] = []
        root_resolved = root.resolve()
        for number, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError(f"{index} entry {number} is missing a string 'file' field")
            path = (root / entry["file"]).resolve()
            if path != root_resolved and root_resolved not in path.parents:
                raise ValueError(f"{index} entry {number} points outside the corpus directory")
            if path.suffix.lower() not in SUFFIXES:
                raise ValueError(f"{index} entry {number} has unsupported file type: {path.name}")
            version = str(entry.get("doc_version", "unknown"))
            flag = entry.get("superseded", False)
            found.append((path, version,
                          flag if isinstance(flag, bool) else str(flag).lower() == "true"))
        return found
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in SUFFIXES or path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def load_exclusions(corpus_dir: str) -> dict[Path, set[int]]:
    """Pages to leave out of the index (a table of contents, a cover), per file.

    Set `exclude_pages: [1, 2, 3]` on a document in corpus/sources.yaml; fetch.py copies it into
    _index.json. This is data about the corpus, so it lives with the corpus and not in chunk.py."""
    index = Path(corpus_dir) / "_index.json"
    if not index.exists():
        return {}
    root = Path(corpus_dir)
    out: dict[Path, set[int]] = {}
    for entry in json.loads(index.read_text(encoding="utf-8")):
        pages = entry.get("exclude_pages") if isinstance(entry, dict) else None
        if isinstance(pages, list) and isinstance(entry.get("file"), str):
            out[(root / entry["file"]).resolve()] = {int(p) for p in pages}
    return out


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk, chunk_fixed
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import reuse_embeddings, store
    from .tokens import count

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    exclusions = load_exclusions(settings.corpus_dir)
    corpus_root = Path(settings.corpus_dir).resolve()
    fixed = os.environ.get("CHUNK_STRATEGY", "structure") == "fixed"   # Stage B experiment only
    all_chunks: list[Chunk] = []
    failures: list[str] = []
    original_tokens = 0
    log("documents discovered", documents=len(documents), corpus=settings.corpus_dir)
    for number, (path, doc_version, superseded) in enumerate(documents, 1):
        started = time.perf_counter()
        stage = "extract"
        try:
            doc = extract(str(path), doc_version, superseded)
            doc.identity = (f"{path.resolve().relative_to(corpus_root).as_posix()}\0"
                            f"{doc_version}\0{superseded}")
            skip = exclusions.get(path.resolve(), set())
            if skip:
                doc.pages = [p for p in doc.pages if p.page not in skip]
            stage = "clean"
            doc = clean(doc)
            original_tokens += sum(count(p.text) for p in doc.pages)
            stage = "chunk"
            pieces = chunk_fixed(doc, settings.chunk_size) if fixed else chunk(
                doc, settings.chunk_size, settings.chunk_overlap)
            if not pieces:
                raise ValueError("document produced no chunks")
            all_chunks.extend(pieces)
            log("chunked", source=path.name, document=number, documents=len(documents),
                pages=len(doc.pages), excluded_pages=len(skip), chunks=len(pieces),
                ocr_pages=sum(1 for p in doc.pages if p.ocr),
                empty_pages=sum(1 for p in doc.pages if not p.text.strip()),
                doc_version=doc_version, superseded=superseded,
                seconds=round(time.perf_counter() - started, 2))
        except Exception as exc:  # collect every failure: one bad file must not hide the others
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            log("document failed", source=path.name, document=number, documents=len(documents),
                stage=stage, error_type=type(exc).__name__, error=str(exc),
                seconds=round(time.perf_counter() - started, 2))
    if failures:   # a partial index silently answers from a partial corpus: build nothing
        raise RuntimeError(f"index build aborted; {len(failures)} of {len(documents)} "
                           f"document(s) failed: " + "; ".join(failures))

    unique: dict[str, Chunk] = {}
    for item in all_chunks:
        previous = unique.get(item.id)
        if previous is not None and previous != item:
            raise RuntimeError(f"conflicting chunks share id {item.id!r}")
        unique[item.id] = item          # a manifest row listed twice collapses to one set of chunks
    all_chunks = list(unique.values())

    stored_tokens = sum(count(c.text) for c in all_chunks)
    log("chunking summary", chunks=len(all_chunks), original_tokens=original_tokens,
        stored_tokens=stored_tokens,
        amplification=round(stored_tokens / max(original_tokens, 1), 3),
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    started = time.perf_counter()
    reused = reuse_embeddings(all_chunks, settings)     # unchanged chunks keep their vectors
    missing = [c for c in all_chunks if c.id not in reused]
    log("embedding chunks", chunks=len(all_chunks), reused=len(reused), to_embed=len(missing),
        model=settings.embed_model)
    new_vectors = embed([c.embed_text() for c in missing], settings)
    by_id = dict(reused)
    by_id.update({c.id: v for c, v in zip(missing, new_vectors, strict=True)})
    vectors = [by_id[c.id] for c in all_chunks]
    log("embeddings ready", reused=len(reused), embedded=len(missing),
        seconds=round(time.perf_counter() - started, 2))
    started = time.perf_counter()
    stored = store(all_chunks, vectors, settings)
    log("index stored", chunks=stored, seconds=round(time.perf_counter() - started, 2))
    return stored


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    args = parser.parse_args()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    except Exception as exc:    # a non-zero exit makes the Job retry and `kubectl logs` explain
        log("indexer failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
