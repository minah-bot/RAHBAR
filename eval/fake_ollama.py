#!/usr/bin/env python3
"""A tiny stand-in for Ollama, for CI and for laptops without a model. Standard library only.

    python eval/fake_ollama.py --port 11434

It speaks the two endpoints RAHBAR uses:
  POST /api/embed  bag-of-words hashing vectors (256 dims): deterministic, and similar texts
                   really are close, so recall@k on the fixture corpus is meaningful.
  POST /api/chat   an *extractive* "model": it answers with the best-matching sentence of the
                   best-matching passage and cites it, or declines when nothing overlaps.

What this does and does not test (ADR 0004): the pipeline, the Job, the pin, the cache, the
gate, retrieval quality on the fixtures. It does NOT test a real model's behaviour: abstention
judgement, citation discipline and injection resistance of llama3.2 are measured locally against
real Ollama and reported in docs/FAILURE-ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 256
STOP = set("""a an and are as at be by can do does for from how i if in is it its my of on or
the to was were what when where which who why will with you your me we they this that there
search query document""".split())
WORD = re.compile(r"[a-z0-9]+")
PASSAGE = re.compile(
    r'<passage id="(\d+)" source="([^"]*)" section="([^"]*)" page="([^"]*)"[^>]*>\n(.*?)\n</passage>',
    re.S)
QUESTION = re.compile(r"<question>\n(.*?)\n</question>", re.S)


def content_words(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for w in content_words(text):
        vec[zlib.crc32(w.encode()) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def answer(user: str) -> dict:
    q = QUESTION.search(user)
    question = html.unescape(q.group(1)) if q else ""
    qwords = set(content_words(question))
    best = (0, None)
    for pid, source, section, page, body in PASSAGE.findall(user):
        overlap = len(qwords & set(content_words(html.unescape(body))))
        if overlap > best[0]:
            best = (overlap, (source, section, page, html.unescape(body)))
    if best[1] is None or best[0] < 2:
        return {"reasoning": "fake: no passage overlaps the question",
                "answer": "The documents I have do not cover this question.",
                "citations": [], "answered_from_context": False, "confidence": "low"}
    source, section, page, body = best[1]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", body) if s.strip()]
    sentences = [s for s in sentences if not re.search(r"ignore (all )?previous", s, re.I)]
    top = max(sentences or [body], key=lambda s: len(qwords & set(content_words(s))))
    return {"reasoning": "fake: extractive", "answer": html.unescape(top),
            "citations": [{"source": html.unescape(source), "section": html.unescape(section),
                           "page": int(page) if page.isdigit() else None}],
            "answered_from_context": True, "confidence": "medium"}


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802  (health check for CI)
        self._send({"status": "ok", "fake": True})

    def do_POST(self) -> None:  # noqa: N802
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/api/embed":
            inputs = data.get("input", [])
            inputs = [inputs] if isinstance(inputs, str) else inputs
            self._send({"model": data.get("model"), "embeddings": [embed(t) for t in inputs]})
        elif self.path == "/api/chat":
            user = data["messages"][-1]["content"]
            reply = answer(user)
            self._send({"message": {"role": "assistant", "content": json.dumps(reply)},
                        "prompt_eval_count": len(user.split()),
                        "eval_count": len(reply["answer"].split())})
        else:
            self.send_error(404)

    def log_message(self, *args: object) -> None:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"fake ollama on {a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
