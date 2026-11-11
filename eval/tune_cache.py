#!/usr/bin/env python3
"""Choose CACHE_THRESHOLD from data, not from a guess (section 3.3).

    python eval/tune_cache.py --ollama-url http://localhost:11434

Reads eval/cache_pairs.yaml: `paraphrases` are question pairs that SHOULD share a cached answer;
`near_misses` look alike but have different answers (a different number, year or course) and must
NOT. For each candidate threshold it prints how many paraphrases hit and how many near-misses
wrongly hit. Pick the lowest threshold with zero wrong hits, and write the table in docs/PIPELINE.md.
A wrong hit serves a confident, cited, WRONG answer, so false hits are the expensive error.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
from app.providers.embedding import prepare  # noqa: E402


def cos(a: list[float], b: list[float]) -> float:
    d = sum(x * y for x, y in zip(a, b, strict=False))
    return d / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--pairs", default=str(ROOT / "eval" / "cache_pairs.yaml"))
    a = ap.parse_args()
    data = yaml.safe_load(Path(a.pairs).read_text(encoding="utf-8"))
    with httpx.Client(base_url=a.ollama_url, timeout=120) as c:
        def sim(pair: list[str]) -> float:
            vs = [c.post("/api/embed", json={"model": a.embed_model, "input": [
                prepare(a.embed_model, t, "query")]}).json()["embeddings"][0] for t in pair]
            return cos(vs[0], vs[1])

        para = [(p, sim(p)) for p in data["paraphrases"]]
        near = [(p, sim(p)) for p in data["near_misses"]]
    print("paraphrases (should hit):")
    for p, s in para:
        print(f"  {s:.3f}  {p[0]!r} ~ {p[1]!r}")
    print("near-misses (must NOT hit):")
    for p, s in near:
        print(f"  {s:.3f}  {p[0]!r} ~ {p[1]!r}")
    print("\n| threshold | paraphrases hit | near-misses wrongly hit |\n|---|---|---|")
    for t in (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
        print(f"| {t:.2f} | {sum(s >= t for _, s in para)}/{len(para)} | "
              f"{sum(s >= t for _, s in near)}/{len(near)} |")


if __name__ == "__main__":
    main()
