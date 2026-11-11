
#!/usr/bin/env python3
"""Score a running RAHBAR against a golden set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import yaml

HERE = Path(__file__).resolve().parent

MAX_RECALL5_DROP = 0.05
MIN_CITATION_VALIDITY = 0.95
REQUIRED_ABSTENTION = 1.0
API_TIMEOUT_SECONDS = 240.0


def load_golden(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []

    if not questions:
        sys.exit(f"no questions in {path}")

    unfinished = [
        q["id"] for q in questions
        if q.get("type", "ordinary") in ("ordinary", "version", "exact")
        and not q.get("relevant")
    ]

    if unfinished:
        sys.exit(
            f"{path}: questions without relevant passages: "
            f"{', '.join(unfinished)}"
        )

    return questions


def ask(client: httpx.Client, api: str, question: str) -> dict:
    response = client.post(
        f"{api.rstrip('/')}/api/ask",
        json={"question": question}
    )
    response.raise_for_status()
    return response.json()


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def section_matches(golden_section: str, retrieved_section: str) -> bool:
    g = _norm(golden_section)
    r = _norm(retrieved_section)

    if g == r:
        return True

    return r.startswith(g) and not r[len(g)].isalnum()


def _hit(rel: dict, ref: dict) -> bool:
    return (
        _norm(rel["source"]) == _norm(ref.get("source", ""))
        and section_matches(rel["section"], ref.get("section", ""))
    )


def recall_at_k(
    relevant: list[dict],
    retrieved: list[dict],
    k: int
) -> float | None:
    if not relevant:
        return None

    top = retrieved[:k]
    found = sum(
        1 for rel in relevant
        if any(_hit(rel, ref) for ref in top)
    )

    return found / len(relevant)


def citation_counts(response: dict) -> tuple[int, int]:
    retrieved = {
        (_norm(r["source"]), _norm(r["section"]))
        for r in response.get("retrieved", [])
    }

    cites = response.get("citations") or []

    valid = sum(
        1 for c in cites
        if (
            _norm(c["source"]),
            _norm(c["section"])
        ) in retrieved
    )

    return valid, len(cites)


def citation_validity(response: dict) -> float | None:
    valid, total = citation_counts(response)
    return valid / total if total else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarise(rows: list[dict]) -> dict:
    r3 = []
    r5 = []
    by_type = {}

    valid = total = 0
    grounded_without_citation = 0
    abstain_rows = abstain_ok = 0
    answerable = answerable_answered = 0
    expect_rows = expect_hits = 0

    per_question = []

    for row in rows:
        resp = row["response"]
        retrieved = resp.get("retrieved", [])

        a3 = recall_at_k(row["relevant"], retrieved, 3)
        a5 = recall_at_k(row["relevant"], retrieved, 5)

        by_type.setdefault(row["type"], [])

        if a3 is not None and a5 is not None:
            r3.append(a3)
            r5.append(a5)
            by_type[row["type"]].append(a5)

        v, t = citation_counts(resp)
        valid += v
        total += t

        if resp.get("answered_from_context") and not resp.get("citations"):
            grounded_without_citation += 1

        if row["must_abstain"]:
            abstain_rows += 1
            abstain_ok += (
                0 if resp.get("answered_from_context") else 1
            )
        else:
            answerable += 1
            answerable_answered += (
                1 if resp.get("answered_from_context") else 0
            )

        if row.get("expect") and not row["must_abstain"]:
            expect_rows += 1
            expect_hits += (
                1 if _norm(row["expect"]) in _norm(resp.get("answer", ""))
                else 0
            )

        per_question.append({
            "id": row.get("id"),
            "type": row["type"],
            "recall_at_5": a5,
            "answered_from_context": resp.get("answered_from_context"),
            "must_abstain": row["must_abstain"],
            "citations_valid": f"{v}/{t}",
            "seconds": row.get("seconds"),
        })

    seconds = sorted(
        r["seconds"] for r in rows
        if r.get("seconds") is not None
    )

    return {
        "recall_at_3": _mean(r3),
        "recall_at_5": _mean(r5),
        "recall_at_5_by_type": {
            k: _mean(v) for k, v in sorted(by_type.items())
        },
        "citation_validity": round(valid / total, 4) if total else 1.0,
        "citations_checked": total,
        "grounded_answers_without_citation": grounded_without_citation,
        "abstention_correctness": (
            round(abstain_ok / abstain_rows, 4)
            if abstain_rows else 1.0
        ),
        "must_abstain_questions": abstain_rows,
        "answer_rate_on_answerable": (
            round(answerable_answered / answerable, 4)
            if answerable else None
        ),
        "expect_hit_rate": (
            round(expect_hits / expect_rows, 4)
            if expect_rows else None
        ),
        "latency_seconds_p50": (
            seconds[len(seconds) // 2] if seconds else None
        ),
        "latency_seconds_p95": (
            seconds[min(len(seconds) - 1, int(len(seconds) * 0.95))]
            if seconds else None
        ),
        "per_question": per_question,
    }


def gate(results: dict, baseline: dict | None) -> list[str]:
    failures = []

    r5 = results.get("recall_at_5")

    if (
        baseline is not None
        and r5 is not None
        and baseline.get("recall_at_5") is not None
        and r5 < baseline["recall_at_5"] - MAX_RECALL5_DROP
    ):
        failures.append(
            f"recall@5 {r5:.3f} is more than {MAX_RECALL5_DROP:.0%} "
            f"below baseline {baseline['recall_at_5']:.3f}"
        )

    if results.get("abstention_correctness", 1.0) < REQUIRED_ABSTENTION:
        bad = [
            q["id"] for q in results.get("per_question", [])
            if q["must_abstain"] and q["answered_from_context"]
        ]
        failures.append(
            f"must-abstain questions answered: {', '.join(bad)}"
        )

    if results.get("citation_validity", 1.0) < MIN_CITATION_VALIDITY:
        failures.append(
            f"citation validity {results['citation_validity']:.3f} "
            f"is below {MIN_CITATION_VALIDITY:.0%}"
        )

    return failures


def save_results(
    rows: list[dict],
    args,
    total_questions: int,
    complete: bool
) -> dict:
    results = summarise(rows)

    try:
        results["golden"] = str(
            args.golden.resolve().relative_to(HERE.parent)
        )
    except ValueError:
        results["golden"] = args.golden.name

    results["questions"] = len(rows)
    results["total_questions"] = total_questions
    results["complete"] = complete

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAHBAR against a golden set."
    )

    parser.add_argument("--api", required=True)
    parser.add_argument("--host")
    parser.add_argument("--golden", type=Path, default=HERE / "golden.yaml")
    parser.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    parser.add_argument("--out", type=Path, default=HERE / "results.json")
    parser.add_argument("--update-baseline", action="store_true")

    args = parser.parse_args()

    questions = load_golden(args.golden)
    rows = []

    headers = {"Host": args.host} if args.host else {}

    try:
        with httpx.Client(
            timeout=API_TIMEOUT_SECONDS,
            headers=headers
        ) as client:

            for q in questions:
                started = time.perf_counter()

                response = ask(client, args.api, q["question"])

                rows.append({
                    "id": q["id"],
                    "type": q.get("type", "ordinary"),
                    "must_abstain": bool(q.get("must_abstain", False)),
                    "relevant": q.get("relevant") or [],
                    "expect": q.get("expect"),
                    "response": response,
                    "seconds": round(
                        time.perf_counter() - started, 2
                    ),
                })

                print(
                    f"  {q['id']:<5} "
                    f"{rows[-1]['seconds']:>6.2f}s  "
                    f"{q['question'][:60]}",
                    flush=True
                )

                # Save genuine results after every completed question.
                save_results(
                    rows,
                    args,
                    len(questions),
                    complete=False
                )

    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")

        if rows:
            save_results(
                rows,
                args,
                len(questions),
                complete=False
            )
            print(f"Partial results saved to {args.out}")

        return 130

    except Exception as exc:
        print(f"\nEvaluation error: {exc}")

        if rows:
            save_results(
                rows,
                args,
                len(questions),
                complete=False
            )
            print(f"Partial results saved to {args.out}")

        return 2

    results = save_results(
        rows,
        args,
        len(questions),
        complete=(len(rows) == len(questions))
    )

    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline.exists()
        else None
    )

    failures = gate(results, baseline)

    print(f"\nResults written to {args.out}")

    if args.update_baseline:
        args.baseline.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8"
        )
        print(f"Baseline updated: {args.baseline}")

    if failures:
        print("\nGATE FAILED")
        for message in failures:
            print(f"  - {message}")
        return 1

    print("\nGate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
