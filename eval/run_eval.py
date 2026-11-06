#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

The runner talks to the API, reads the golden set, writes results.json, and exits
non-zero when the evaluation gate fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

# The gate from section 5.4. Change these only with a reason written in docs/PIPELINE.md.
MAX_RECALL5_DROP = 0.05        # recall@5 may fall at most 5 points below the baseline
MIN_CITATION_VALIDITY = 0.95   # at least 95% of citations must match retrieved passages
REQUIRED_ABSTENTION = 1.0      # every must-abstain question must be declined


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None
    wanted = {(str(x.get("source", "")).casefold(), str(x.get("section", "")).casefold())
              for x in relevant}
    found = {(str(x.get("source", "")).casefold(), str(x.get("section", "")).casefold())
             for x in retrieved[:max(0, k)]}
    return len(wanted & found) / len(wanted)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    citations = response.get("citations") or []
    if not citations:
        return None
    retrieved = {(str(x.get("source", "")).casefold(), str(x.get("section", "")).casefold())
                 for x in response.get("retrieved", [])}
    return sum((str(c.get("source", "")).casefold(), str(c.get("section", "")).casefold())
               in retrieved for c in citations) / len(citations)


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json."""
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    recalls = {k: [value for row in rows
                   if (value := recall_at_k(row.get("relevant", []),
                                            row["response"].get("retrieved", []), k)) is not None]
               for k in (3, 5)}
    citations = [value for row in rows
                 if (value := citation_validity(row["response"])) is not None]
    abstain_rows = [row for row in rows if row.get("must_abstain")]
    abstention = (sum(not row["response"].get("answered_from_context", False)
                      for row in abstain_rows) / len(abstain_rows)) if abstain_rows else None
    by_type: dict[str, float | None] = {}
    for kind in sorted({str(row.get("type", "ordinary")) for row in rows}):
        values = [recall_at_k(row.get("relevant", []), row["response"].get("retrieved", []), 5)
                  for row in rows if row.get("type", "ordinary") == kind]
        by_type[kind] = mean([v for v in values if v is not None])
    return {"recall_at_3": mean(recalls[3]), "recall_at_5": mean(recalls[5]),
            "citation_validity": mean(citations), "abstention_correctness": abstention,
            "recall_at_5_by_type": by_type,
            "must_abstain_questions": len(abstain_rows), "rows": len(rows)}


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []
    citation = results.get("citation_validity")
    if citation is None or citation < MIN_CITATION_VALIDITY:
        failures.append(f"citation validity {citation} is below {MIN_CITATION_VALIDITY}")
    if results.get("must_abstain_questions", 0) and results.get("abstention_correctness") != REQUIRED_ABSTENTION:
        failures.append(f"must-abstain correctness {results.get('abstention_correctness')} is below {REQUIRED_ABSTENTION}")
    if baseline is not None:
        old = baseline.get("recall_at_5")
        current = results.get("recall_at_5")
        if old is not None and current is not None and current < old - MAX_RECALL5_DROP:
            failures.append(f"recall@5 {current:.3f} dropped more than {MAX_RECALL5_DROP:.3f} from baseline {old:.3f}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAHBAR against a golden set.")
    parser.add_argument("--api", required=True, help="base URL, e.g. http://192.168.49.2")
    parser.add_argument("--host", help="Host header, when your Ingress rule names a host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="write these results as the new baseline")
    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []
    headers = {"Host": args.host} if args.host else {}
    with httpx.Client(timeout=120.0, headers=headers) as client:
        for q in questions:
            started = time.perf_counter()
            response = ask(client, args.api, q["question"])
            rows.append({
                "id": q["id"], "type": q.get("type", "ordinary"),
                "must_abstain": bool(q.get("must_abstain", False)),
                "relevant": q.get("relevant") or [],
                "expect": q.get("expect"),
                "response": response,
                "seconds": round(time.perf_counter() - started, 2),
            })
            print(f"  {q['id']:<5} {rows[-1]['seconds']:>6.2f}s  {q['question'][:60]}")

    try:
        results = summarise(rows)
        baseline = json.loads(args.baseline.read_text()) if args.baseline.exists() else None
        failures = gate(results, baseline)
    except NotImplementedError as exc:
        print(f"\nnot implemented yet: {exc}")
        return 2

    results["golden"] = str(args.golden)
    results["questions"] = len(rows)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {args.out}")
    if args.update_baseline:
        args.baseline.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\ngate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
