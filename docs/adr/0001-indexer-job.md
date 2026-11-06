# ADR 0001: Run indexing as a separate Job

- Status: Accepted
- Context: API replicas must not race to write Chroma or depend on source documents and
  embedding latency during process startup.
- Decision: `indexer/pipeline.run` is a one-shot command. The API reads the configured
  collection only. Kubernetes resources are outside the current implementation scope.
- Consequence: Operations must run indexing before routing traffic to a new/empty index.
  The indexer builds a temporary collection and copies it to the active name after staging;
  Chroma has no atomic collection rename, so this has a short replacement window. A
  blue/green collection and deployment-level alias/switch is a future improvement.
