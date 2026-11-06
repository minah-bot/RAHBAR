"""Liveness and readiness (section 4.3). They answer different questions."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get("/healthz")
def healthz() -> dict[str, str]:
    """Is the process alive? Must never touch Chroma, Redis or the LLM."""
    return {"status": "alive"}


@router.get("/readyz")
def readyz(request: Request, response: Response) -> dict[str, object]:
    settings = request.app.state.settings
    checks: dict[str, object] = {}
    failures: list[str] = []
    if settings.retriever == "chroma":
        try:
            collection = request.app.state.retriever.collection
            count = collection.count()
            metadata = collection.metadata or {}
            checks["chroma"] = {"reachable": True, "count": count}
            if count == 0:
                failures.append("chroma collection is empty")
            if metadata.get("embed_model") != settings.embed_model:
                checks["embedding_model"] = {"expected": settings.embed_model,
                                              "indexed": metadata.get("embed_model")}
                failures.append("embedding model does not match index")
            else:
                checks["embedding_model"] = "matched"
        except Exception as exc:
            checks["chroma"] = {"reachable": False, "error": str(exc)}
            failures.append("chroma unavailable")
    else:
        checks["retriever"] = "stub"
    if settings.redis_url:
        try:
            import redis

            redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5,
                                 socket_timeout=0.5).ping()
            checks["redis"] = "reachable"
        except Exception as exc:
            checks["redis"] = {"reachable": False, "error": str(exc)}
            failures.append("redis unavailable")
    else:
        checks["redis"] = "disabled"
    if failures:
        response.status_code = 503
        return {"status": "not_ready", "checks": checks, "errors": failures}
    return {"status": "ready", "checks": checks}
