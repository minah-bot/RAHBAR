#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set, and gate CI on the result (section 5).

    python eval/run_eval.py --api http://<ingress>                 # your corpus
    python eval/run_eval.py --api ... --golden eval/fixtures/golden.fixture.yaml
    python eval/run_eval.py --api ... --update-baseline            # after a deliberate improvement

Exit code 0 when every gate passes, 1 when any fails. CI relies on that exit code.

Talking to the API, reading the golden set, writing results.json and the exit code are the
harness; the four functions recall_at_k, citation_validity, summarise and gate are the evaluation.
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
API_TIMEOUT_SECONDS = 240.0    # allows two 90s model attempts plus API/retry overhead


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        sys.exit(f"no questions in {path}")
    # A golden set with holes gives a flattering number that means nothing (section 5.1).
    unfinished = [q["id"] for q in questions
                  if q.get("type", "ordinary") in ("ordinary", "version", "exact")
                  and not q.get("relevant")]
    if unfinished:
        sys.exit(f"{path}: questions without `relevant` passages: {', '.join(unfinished)}. "
                 "Fill them in (python eval/find_sections.py helps) before running the eval.")
    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    """POST one question and return the AskResponse as a dict (see api/app/schemas.py)."""
    response = client.post(f"{api.rstrip('/')}/api/ask", json={"question": question})
    response.raise_for_status()
    return response.json()


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def section_matches(golden_section: str, retrieved_section: str) -> bool:
    """A golden section matches a retrieved one when they are equal, or when the golden one is
    a clause-number prefix of it: golden "5.4" matches "5.4 Academic Warning" and "5.4.1 ...",
    but not "5.40". Case and whitespace are ignored. Documented in docs/PIPELINE.md."""
    g, r = _norm(golden_section), _norm(retrieved_section)
    if g == r:
        return True
    return r.startswith(g) and not r[len(g)].isalnum()


def _hit(rel: dict, ref: dict) -> bool:
    return _norm(rel["source"]) == _norm(ref.get("source", "")) and section_matches(
        rel["section"], ref.get("section", ""))


def recall_at_k(relevant: list[dict], retrieved: list[dict], k: int) -> float | None:
    """Fraction of the relevant (source, section) pairs found in the top k retrieved.
    Return None when the question has no relevant passages (must-abstain questions)."""
    if not relevant:
        return None
    top = retrieved[:k]
    found = sum(1 for rel in relevant if any(_hit(rel, ref) for ref in top))
    return found / len(relevant)


def citation_counts(response: dict) -> tuple[int, int]:
    """(valid, total) citations: valid means it names a (source, section) that was retrieved."""
    retrieved = {(_norm(r["source"]), _norm(r["section"])) for r in response.get("retrieved", [])}
    cites = response.get("citations") or []
    valid = sum(1 for c in cites if (_norm(c["source"]), _norm(c["section"])) in retrieved)
    return valid, len(cites)


def citation_validity(response: dict) -> float | None:
    """Fraction of the response's citations that match a passage in response['retrieved'].
    Return None when there are no citations."""
    valid, total = citation_counts(response)
    return valid / total if total else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarise(rows: list[dict]) -> dict:
    """Overall recall@3, recall@5, citation validity and abstention correctness, plus
    recall@5 broken down by question type. This dict becomes results.json."""
    r3: list[float] = []
    r5: list[float] = []
    by_type: dict[str, list[float]] = {}
    valid = total = 0
    answered_rows = grounded_without_citation = 0
    abstain_rows = abstain_ok = 0
    answerable = answerable_answered = 0
    expect_rows = expect_hits = 0
    per_question = []
    for row in rows:
        resp = row["response"]
        retrieved = resp.get("retrieved", [])
        a3 = recall_at_k(row["relevant"], retrieved, 3)
        a5 = recall_at_k(row["relevant"], retrieved, 5)
        by_type.setdefault(row["type"], [])      # a type with no relevant passages reports None
        if a3 is not None and a5 is not None:
            r3.append(a3)
            r5.append(a5)
            by_type[row["type"]].append(a5)
        v, t = citation_counts(resp)
        valid, total = valid + v, total + t
        if resp.get("answered_from_context"):
            answered_rows += 1
            grounded_without_citation += 0 if resp.get("citations") else 1
        if row["must_abstain"]:
            abstain_rows += 1
            abstain_ok += 0 if resp.get("answered_from_context") else 1
        else:
            answerable += 1
            answerable_answered += 1 if resp.get("answered_from_context") else 0
        if row.get("expect") and not row["must_abstain"]:
            expect_rows += 1
            expect_hits += 1 if _norm(row["expect"]) in _norm(resp.get("answer", "")) else 0
        per_question.append({
            "id": row.get("id"), "type": row["type"], "recall_at_5": a5,
            "answered_from_context": resp.get("answered_from_context"),
            "must_abstain": row["must_abstain"], "citations_valid": f"{v}/{t}",
            "seconds": row.get("seconds")})
    seconds = sorted(r["seconds"] for r in rows if r.get("seconds") is not None)
    return {
        "recall_at_3": _mean(r3),
        "recall_at_5": _mean(r5),
        "recall_at_5_by_type": {k: _mean(v) for k, v in sorted(by_type.items())},
        "citation_validity": round(valid / total, 4) if total else 1.0,
        "citations_checked": total,
        "grounded_answers_without_citation": grounded_without_citation,
        "abstention_correctness": round(abstain_ok / abstain_rows, 4) if abstain_rows else 1.0,
        "must_abstain_questions": abstain_rows,
        "answer_rate_on_answerable": round(answerable_answered / answerable, 4) if answerable
        else None,
        "expect_hit_rate": round(expect_hits / expect_rows, 4) if expect_rows else None,
        "latency_seconds_p50": seconds[len(seconds) // 2] if seconds else None,
        "latency_seconds_p95": seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))]
        if seconds else None,
        "per_question": per_question,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    """Return one message per failed gate (section 5.4); an empty list means pass."""
    failures: list[str] = []
    r5 = results.get("recall_at_5")
    if baseline is not None and r5 is not None and baseline.get("recall_at_5") is not None \
            and r5 < baseline["recall_at_5"] - MAX_RECALL5_DROP:
        failures.append(f"recall@5 {r5:.3f} is more than {MAX_RECALL5_DROP:.0%} below the "
                        f"baseline {baseline['recall_at_5']:.3f}")
    if results.get("abstention_correctness", 1.0) < REQUIRED_ABSTENTION:
        bad = [q["id"] for q in results.get("per_question", [])
               if q["must_abstain"] and q["answered_from_context"]]
        failures.append(f"must-abstain question(s) answered from context: {', '.join(bad)}")
    if results.get("citation_validity", 1.0) < MIN_CITATION_VALIDITY:
        failures.append(f"citation validity {results['citation_validity']:.3f} is below "
                        f"{MIN_CITATION_VALIDITY:.0%}")
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
    with httpx.Client(timeout=API_TIMEOUT_SECONDS, headers=headers) as client:
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

    try:
        results["golden"] = str(args.golden.resolve().relative_to(HERE.parent))
    except ValueError:
        results["golden"] = args.golden.name
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
