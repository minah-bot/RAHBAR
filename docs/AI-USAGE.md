# AI usage

> **TEAM TODO before submitting:** this table is a draft written by the assistant that produced the first version of
> the code. The right-hand column must be filled in by *you*, truthfully, after you have read, run and changed things.
> The rule is honest attribution: a specific disclosure carries no penalty, presenting generated work as your own does.
> You must be able to explain every file below in a viva; if you cannot yet, read it before you submit.

| Tool | What it wrote or shaped | What we changed afterwards, and why |
| --- | --- | --- |
| Claude (Anthropic), chat | `api/app`: `ChromaRetriever`, embedding client, Ollama and hosted providers, prompt builder, citation validation, answer pipeline with retry and fallback, semantic cache, readiness and the embedding pin, new metrics | **TEAM TODO** |
| Claude, chat | `indexer/pipeline`: `extract` (pdfplumber, tables, OCR), `clean`, `chunk`, `embed`, `store` (new collection per run, pointer swap), run summary logging | **TEAM TODO** |
| Claude, chat | `eval/`: `recall_at_k`, `citation_validity`, `summarise`, `gate`, `fake_ollama.py`, `find_sections.py`, `chunk_sweep.py`, `tune_cache.py`, fixture baseline | **TEAM TODO** (the golden set in `eval/golden.yaml` must be yours) |
| Claude, chat | Dockerfiles, all manifests under `k8s/`, `.github/workflows/ci.yml` and `cd.yml`, `scripts/kind_deploy.sh`, `scripts/local_e2e.sh`, Makefile targets | **TEAM TODO** (what you changed after running them on minikube and GitHub Actions) |
| Claude, chat | Unit tests for both packages; ADRs 0001-0004, ARCHITECTURE, README, PIPELINE, ENGINEERING-NOTES answers 1, 2, 3 and 6, FAILURE-ANALYSIS procedure | **TEAM TODO** |
| (add any other tool) | | |

Not generated, and not claimed: the corpus selection and `docs/CORPUS.md`, the golden set, all measured numbers, the screenshots
under `docs/evidence/`, answers 4, 5 and 7 of ENGINEERING-NOTES, and the results of the failure experiments.
