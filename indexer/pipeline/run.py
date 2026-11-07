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
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
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
            if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
                raise ValueError(f"{index} entry {number} has unsupported file type: {path.name}")
            version = str(entry.get("doc_version", "unknown"))
            superseded_value = entry.get("superseded", False)
            superseded = (superseded_value if isinstance(superseded_value, bool)
                          else str(superseded_value).lower() == "true")
            found.append((path, version, superseded))
        return found
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(settings: IndexerSettings) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import reuse_embeddings, store

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")
    all_chunks: list[Chunk] = []
    failures: list[str] = []
    corpus_root = Path(settings.corpus_dir).resolve()
    log("documents discovered", documents=len(documents), corpus=settings.corpus_dir)
    for number, (path, doc_version, superseded) in enumerate(documents, 1):
        started = time.perf_counter()
        stage = "extract"
        try:
            doc = extract(str(path), doc_version, superseded)
            doc.identity = (
                f"{path.resolve().relative_to(corpus_root).as_posix()}\0"
                f"{doc_version}\0{superseded}"
            )
            stage = "clean"
            doc = clean(doc)
            page_count = len(doc.pages)
            ocr_pages = sum(page.ocr for page in doc.pages)
            stage = "chunk"
            pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
            if not pieces:
                raise ValueError("document produced no chunks")
            all_chunks.extend(pieces)
            log("document indexed for staging", source=path.name, document=number,
                documents=len(documents), pages=page_count, ocr_pages=ocr_pages,
                chunks=len(pieces), seconds=round(time.perf_counter() - started, 2))
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            log("document failed", source=path.name, document=number,
                documents=len(documents), stage=stage, error_type=type(exc).__name__,
                error=str(exc), seconds=round(time.perf_counter() - started, 2))

    if failures:
        raise RuntimeError(
            f"index build aborted; {len(failures)} of {len(documents)} document(s) failed: "
            + "; ".join(failures)
        )
    if not all_chunks:
        raise RuntimeError("index build aborted; no chunks were produced")

    unique_chunks: dict[str, Chunk] = {}
    for item in all_chunks:
        previous = unique_chunks.get(item.id)
        if previous is not None and previous != item:
            raise RuntimeError(f"conflicting chunks share id {item.id!r}")
        unique_chunks[item.id] = item
    all_chunks = list(unique_chunks.values())

    started = time.perf_counter()
    reused = reuse_embeddings(all_chunks, settings)
    missing = [item for item in all_chunks if item.id not in reused]
    log("embedding chunks", chunks=len(all_chunks), reused=len(reused),
        to_embed=len(missing), model=settings.embed_model)
    new_vectors = embed([item.text for item in missing], settings)
    vectors_by_id = dict(reused)
    vectors_by_id.update({item.id: vector for item, vector in zip(missing, new_vectors)})
    vectors = [vectors_by_id[item.id] for item in all_chunks]
    log("embeddings ready", reused=len(reused), embedded=len(missing),
        seconds=round(time.perf_counter() - started, 2))
    started = time.perf_counter()
    count = store(all_chunks, vectors, settings)
    log("index stored", chunks=count, seconds=round(time.perf_counter() - started, 2))
    return count


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
    except Exception as exc:
        log("indexer failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
