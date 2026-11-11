#!/usr/bin/env bash
# Full pipeline on one machine, no cluster, no real model:
#   fake Ollama -> indexer -> Chroma -> API (3 workers' worth of logic, 1 process) -> eval gate
# Used to check the code before touching Kubernetes, and by CI's fast path.
#   scripts/local_e2e.sh                # needs: make install; pip install chromadb==1.0.12 (server)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/.venv/bin}"
WORK="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$WORK"' EXIT

"$BIN/python" "$ROOT/eval/fake_ollama.py" --port 11500 >"$WORK/ollama.log" 2>&1 &
# CHROMA_RUN overrides the server command, e.g. the Rust binary the container image runs:
#   CHROMA_RUN="/path/to/chroma run --path /tmp/c --host 127.0.0.1 --port 8765" scripts/local_e2e.sh
${CHROMA_RUN:-$BIN/chroma run --path $WORK/chroma --port 8765} >"$WORK/chroma.log" 2>&1 &
for _ in $(seq 1 40); do
  curl -sf localhost:8765/api/v2/heartbeat >/dev/null && curl -sf localhost:11500/ >/dev/null && break
  sleep 1
done

export CHROMA_HOST=localhost CHROMA_PORT=8765 OLLAMA_URL=http://localhost:11500
export EMBED_MODEL=hash-256 CORPUS_DIR="$ROOT/eval/fixtures" CHUNK_SIZE=120 CHUNK_OVERLAP=20
echo "== indexer (first run)"; (cd "$ROOT/indexer" && "$BIN/python" -m pipeline.run | tail -3)
sleep 1   # collection names carry a one-second timestamp
echo "== indexer (second run: unchanged chunks reuse their vectors)"
(cd "$ROOT/indexer" && "$BIN/python" -m pipeline.run | grep -E "embedding chunks|index stored")

export RETRIEVER=chroma LLM_PROVIDER=ollama-host RELEVANCE_FLOOR=0.15 TOP_K=5 TEMPERATURE=0.1
export LLM_MODEL=fake CACHE_ENABLED=false
(cd "$ROOT/api" && "$BIN/uvicorn" app.main:app --port 8000 >"$WORK/api.log" 2>&1) &
for _ in $(seq 1 30); do curl -sf localhost:8000/readyz >/dev/null && break; sleep 1; done
echo "== readyz"; curl -s localhost:8000/readyz; echo
echo "== eval"; "$BIN/python" "$ROOT/eval/run_eval.py" --api http://localhost:8000 \
  --golden "$ROOT/eval/fixtures/golden.fixture.yaml" --out "$WORK/results.json" \
  ${EVAL_ARGS:-}
