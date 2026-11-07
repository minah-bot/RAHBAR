"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, Response

from ..retrieval.index import IndexNotReady

router = APIRouter()
log = logging.getLogger("rahbar.health")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    """Can this pod take traffic? 200 only if Chroma is reachable, the collection exists and
    is non-empty, the embedding model matches the index, and Redis is reachable. Otherwise
    503 naming the dependency that failed, so `kubectl describe` shows why it is 0/1."""
    state = request.app.state
    checks: dict[str, object] = {}
    try:
        check = getattr(state.retriever, "check_ready", None)   # tolerate plain retrievers
        checks["index"] = check() if check else {"retriever": state.retriever.name}
        if state.cache is not None:
            try:
                state.cache.ping()
            except Exception as exc:
                raise IndexNotReady("redis", f"redis unreachable: {exc!r}") from exc
            checks["redis"] = "ok"
    except IndexNotReady as exc:
        response.status_code = 503
        log.warning("not ready", extra={"dependency": exc.dependency, "detail": exc.detail})
        return {"status": "not ready", "failed": exc.dependency, "detail": exc.detail,
                "checks_passed": len(checks)}
    return {"status": "ready", "checks": checks}
