# Evidence checklist

Commit these files here (names are suggestions). Each one backs a claim in the documents.

| File | How to produce it | Backs |
| --- | --- | --- |
| `branch-protection.png` | GitHub: Settings > Branches > rule for `main`: no direct pushes, PR required, required checks `lint` `test` `manifests` `build` `scan` `e2e-eval`, 1 approval | section 6 |
| `merge-blocked.png`, `merge-green.png` | open a PR that fails the eval gate (e.g. set `RELEVANCE_FLOOR` to 0.01 in the ci overlay, or break chunking), screenshot the greyed-out merge button, fix it, screenshot green | "show the gate working" |
| `kubectl-top.txt` | `make top > docs/evidence/kubectl-top.txt` while serving, and the indexer sampling loop from ENGINEERING-NOTES 4 | section 4.5 |
| `pin-0of1.png` | the Stage C(b) experiment: pod at `0/1` and the `embedding pin violated` log line | section 4.4 |
| `rollout.txt` | `kubectl set image ... ; kubectl rollout status ... ; kubectl rollout undo ...` with `kubectl get deploy -o wide` before and after | section 4.6 |
| `endpoints.txt` | `kubectl scale deploy/api --replicas=4 -n rahbar; kubectl get endpoints api -n rahbar` | section 4.6 |
| `netpol-ui-to-chroma.txt` | `kubectl exec -n rahbar deploy/ui -- wget -qO- -T3 http://chroma:8000/api/v2/heartbeat` times out; the same from an api pod succeeds | section 4.7 |
| `hpa.txt` | `kubectl get hpa -n rahbar` showing a real percentage, not `<unknown>` | section 4.5 |
| `reindex.txt` | `make index` while `watch kubectl get pods`; the API keeps answering | section 4.2 |
| `chat.png` | the UI answering with citations | README |
