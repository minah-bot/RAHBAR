# ADR 0004: LLM provider behind an interface, and what leaves the cluster

## Status
Accepted · 2026-11-06

## Context
Generation needs a model, and the choices differ in cost, memory and where the question text goes.
The assignment offers Ollama on the laptop, Ollama in the cluster, or a free hosted endpoint. Questions
can contain personal details (a roll number, a CGPA, a name), and retrieved passages are university
documents. A hosted endpoint receives both.

## Decision
`LLM_PROVIDER` in the ConfigMap selects an implementation of `LLMProvider.generate(question, passages)`
(`api/app/providers/llm/`). `ollama` (host or in-cluster, only `OLLAMA_URL` differs) is the default and is
fully supported: structure is enforced with Ollama's `format=` schema, then validated again with pydantic.
`hosted` (any OpenAI-compatible endpoint) is implemented behind the same interface using JSON mode with the
schema in the prompt; its key is `LLM_API_KEY` from the Secret `rahbar-secret`, created with
`kubectl create secret` and never committed, and the provider never logs it or puts it in an error message.
Embeddings are always produced through `OLLAMA_URL` (the pinned model in ADR 0001/PIPELINE.md), including when
generation is hosted, so question and chunk vectors always come from the same model.
**Data governance:** with `ollama-host`, question text and passages stay on the user's machine.
With `hosted`, both are sent to a third party, so the default is the local mode, hosted mode needs the extra
egress NetworkPolicy to be applied deliberately (`k8s/base/api-hosted-egress.placeholder.yaml`, not in the
kustomization), and the logs record the question's retrieval metadata (chunk ids, scores) but are
the place to review before pointing real student questions at a hosted model. The system prompt tells the
model that retrieved text and the question are data, not instructions (delimiters, escaping, schema, citation check).
Temperature is 0.1: facts and JSON want near-determinism; exactly 0 does not remove hallucination, because
absent knowledge stays absent.

## Options considered
* *Hosted only:* simplest to run, but every question leaves the machine, and a free tier can be rate limited
  mid-demo.
* *In-cluster Ollama:* the bonus option; needs 8 GB for minikube and a startupProbe with a large threshold.
* *Calling a model SDK directly from the route:* swapping providers would touch the pipeline.

## Consequences
Easier: swapping a provider is one file; the local default keeps data on the laptop. Harder: a small local
model (3B) is weaker at following the citation format, which is why output is validated, retried once and then
replaced by a safe fallback instead of trusted. Watch: `rahbar_validation_failures_total`,
`rahbar_fallbacks_total` and `rahbar_citations_dropped_total`; a rising citations-dropped count is the
model inventing sources.
