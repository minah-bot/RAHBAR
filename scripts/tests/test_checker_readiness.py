import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "submission_checker", ROOT / "scripts/check_submission.py"
)
assert spec is not None and spec.loader is not None
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_checker_recognizes_the_real_delegated_empty_collection_guard() -> None:
    health_source = (ROOT / "api/app/routes/health.py").read_text(encoding="utf-8")
    chroma_source = (ROOT / "api/app/retrieval/chroma.py").read_text(encoding="utf-8")
    ready_body = checker._function_body(health_source, r"\w*read(y|iness)\w*")

    assert ready_body is not None
    assert checker._readiness_checks_nonempty(ready_body, chroma_source)


def test_checker_does_not_accept_delegation_without_zero_count_guard() -> None:
    ready_body = 'checks["index"] = check() if check else {"retriever": "stub"}'
    chroma_without_guard = "def check_ready(self):\n    return {\"chunks\": 1}"

    assert not checker._readiness_checks_nonempty(ready_body, chroma_without_guard)


def test_checker_still_accepts_a_direct_count_guard() -> None:
    assert checker._readiness_checks_nonempty("if collection.count() == 0: raise RuntimeError()", "")
