#!/usr/bin/env bash
# Deploy RAHBAR's `ci` overlay to the current kind cluster and wait until it can answer.
# Used by ci.yml (images loaded from the build job) and cd.yml (images pulled from Docker Hub
# by commit SHA, when REGISTRY_USER and TAG are set). Needs eval/fake_ollama.py running on the
# host on port 11500; see the workflows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
GATEWAY="$(docker network inspect kind -f '{{range .IPAM.Config}}{{.Gateway}} {{end}}' \
  | tr ' ' '\n' | grep -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$' | head -1)"
echo "kind gateway (where the fake model listens): $GATEWAY"

cp -r "$ROOT/k8s" "$WORK/k8s"
sed -i "s/KIND_GATEWAY/$GATEWAY/g" "$WORK/k8s/overlays/ci/kustomization.yaml"
if [ -n "${REGISTRY_USER:-}" ] && [ -n "${TAG:-}" ]; then
  (cd "$WORK/k8s/overlays/ci" && for img in api ui indexer; do
     kustomize edit set image "rahbar-$img=docker.io/$REGISTRY_USER/rahbar-$img:$TAG"; done)
fi

kubectl apply -f "$ROOT/k8s/base/namespace.yaml"
kubectl create configmap corpus-fixtures -n rahbar \
  --from-file="$ROOT/eval/fixtures/academic-rules-2023.md" \
  --from-file="$ROOT/eval/fixtures/academic-rules-2026.md" \
  --from-file="$ROOT/eval/fixtures/fees-2026.md" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k "$WORK/k8s/overlays/ci"

kubectl rollout status statefulset/chroma -n rahbar --timeout=240s
kubectl rollout status deployment/redis -n rahbar --timeout=120s
kubectl wait --for=condition=complete job/indexer -n rahbar --timeout=300s
kubectl rollout status deployment/api -n rahbar --timeout=240s
kubectl rollout status deployment/ui -n rahbar --timeout=120s
kubectl get pods,svc,endpoints -n rahbar
