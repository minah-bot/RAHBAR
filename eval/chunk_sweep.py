#!/usr/bin/env python3
"""Offline chunk-size experiment (section 3.1 asks you to try at least three).

    python eval/chunk_sweep.py --corpus corpus/raw --golden eval/golden.yaml \\
        --configs 200:30,400:50,800:100 --ollama-url http://localhost:11434

For each size:overlap it chunks the corpus with the REAL indexer code, embeds the chunks and the
golden questions with the pinned embedding model, ranks by cosine similarity in memory (no
Chroma, no cluster) and reports recall@3 / recall@5 overall and by question type, plus the
token amplification. Needs only Ollama running with the embedding model pulled.
Paste the table into docs/PIPELINE.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "indexer"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "api"))

from pipeline.chunk import chunk  # noqa: E402
from pipeline.clean import clean  # noqa: E402
from pipeline.config import IndexerSettings  # noqa: E402
from pipeline.embed import embed, prepare  # noqa: E402
from pipeline.extract import extract  # noqa: E402
from pipeline.run import discover  # noqa: E402
from pipeline.tokens import count  # noqa: E402
from app.retrieval.version import wants_older_version  # noqa: E402
from run_eval import load_golden, recall_at_k  # noqa: E402


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b, strict=False))
    return d / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "corpus" / "raw"))
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.yaml"))
    ap.add_argument("--configs", default="200:30,400:50,800:100")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--out", default=str(ROOT / "eval" / "chunk_sweep.json"))
    a = ap.parse_args()

    questions = [q for q in load_golden(Path(a.golden)) if q.get("relevant")]
    base = IndexerSettings(embed_model=a.embed_model, chroma_host="", chroma_port=0,
                           collection="", corpus_dir=a.corpus, chunk_size=0, chunk_overlap=0,
                           ollama_url=a.ollama_url)
    docs = [clean(extract(str(p), v, s)) for p, v, s in discover(a.corpus)]
    original = sum(count(pg.text) for d in docs for pg in d.pages)
    import httpx

    qvecs = []
    with httpx.Client(base_url=a.ollama_url, timeout=300) as c:
        for q in questions:
            r = c.post("/api/embed", json={"model": a.embed_model,
                                           "input": [prepare(a.embed_model, q["question"],
                                                             "query")]})
            qvecs.append(r.json()["embeddings"][0])

    rows = []
    for cfg in a.configs.split(","):
        size, overlap = (int(x) for x in cfg.split(":"))
        chunks = [c for d in docs for c in chunk(d, size, overlap)]
        vecs = embed([c.embed_text() for c in chunks], base)
        by_type: dict[str, list[float]] = {}
        r3: list[float] = []
        r5: list[float] = []
        for q, qv in zip(questions, qvecs, strict=True):
            ranked = sorted(((cos(qv, v), c) for v, c in zip(vecs, chunks, strict=True)
                             if not c.superseded or wants_older_version(q["question"])),
                            key=lambda t: t[0], reverse=True)
            retrieved = [{"source": c.source, "section": c.section} for _, c in ranked[:5]]
            x3 = recall_at_k(q["relevant"], retrieved, 3)
            x5 = recall_at_k(q["relevant"], retrieved, 5)
            if x3 is not None and x5 is not None:
                r3.append(x3)
                r5.append(x5)
                by_type.setdefault(q.get("type", "ordinary"), []).append(x5)
        rows.append({
            "size": size, "overlap": overlap, "chunks": len(chunks),
            "amplification": round(sum(count(c.text) for c in chunks) / max(original, 1), 3),
            "recall_at_3": round(sum(r3) / len(r3), 3), "recall_at_5": round(sum(r5) / len(r5), 3),
            "by_type": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_type.items())},
        })
        print(json.dumps(rows[-1]), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("\n| size:overlap | chunks | amplification | recall@3 | recall@5 | by type (recall@5) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['size']}:{r['overlap']} | {r['chunks']} | {r['amplification']} | "
              f"{r['recall_at_3']} | {r['recall_at_5']} | {r['by_type']} |")


if __name__ == "__main__":
    main()
