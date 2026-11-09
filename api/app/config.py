"""Runtime configuration, read once from environment variables.

On the cluster these variables come from your ConfigMap and Secret (section 4.1).
Nothing here is baked into the image, so one image runs in every environment.
Add your own keys (relevance floor, cache settings, ...) the same way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    # Which implementations to load. "stub" needs no cluster, no model, no network.
    retriever: str = "stub"          # RETRIEVER: stub | chroma
    llm_provider: str = "stub"       # LLM_PROVIDER: stub | ollama-host | ollama-incluster | hosted

    # The embedding pin (section 4.4). The indexer reads the same ConfigMap key.
    embed_model: str = "nomic-embed-text"

    # Vector store
    chroma_host: str = "chroma"
    chroma_port: int = 8000
    collection: str = "rahbar"

    # Retrieval and generation
    top_k: int = 5
    temperature: float = 0.2
    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://host.minikube.internal:11434"
    hosted_base_url: str = ""        # e.g. an OpenAI-compatible endpoint
    llm_api_key: str = field(default="", repr=False)   # from the Secret; never log it

    log_level: str = "INFO"

    # Production steps (sections 3.3, 3.4, 4.3). All of these come from the ConfigMap.
    relevance_floor: float = 0.55    # RELEVANCE_FLOOR: below this best-passage similarity, abstain
    cache_enabled: bool = False      # CACHE_ENABLED: true on the cluster, false for local stubs
    cache_threshold: float = 0.95    # CACHE_THRESHOLD: cosine similarity needed for a cache hit
    cache_ttl_seconds: int = 604800  # CACHE_TTL_SECONDS: entries also expire after a week
    redis_host: str = "redis"        # REDIS_HOST
    redis_port: int = 6379           # REDIS_PORT
    lexical_weight: float = 0.15     # LEXICAL_WEIGHT: share of the score from exact-term overlap
    llm_timeout: float = 90.0        # LLM_TIMEOUT: seconds; a hung model must not hold a worker
    index_ttl_seconds: float = 5.0   # how long the API trusts what it read about the index


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def get_settings() -> Settings:
    return Settings(
        retriever=os.environ.get("RETRIEVER", "stub"),
        llm_provider=os.environ.get("LLM_PROVIDER", "stub"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=_int("CHROMA_PORT", 8000),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        top_k=_int("TOP_K", 5),
        temperature=_float("TEMPERATURE", 0.2),
        llm_model=os.environ.get("LLM_MODEL", "llama3.2:3b"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        hosted_base_url=os.environ.get("HOSTED_BASE_URL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        relevance_floor=_float("RELEVANCE_FLOOR", 0.55),
        cache_enabled=_bool("CACHE_ENABLED", False),
        cache_threshold=_float("CACHE_THRESHOLD", 0.95),
        cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 604800),
        redis_host=os.environ.get("REDIS_HOST", "redis"),
        redis_port=_int("REDIS_PORT", 6379),
        lexical_weight=_float("LEXICAL_WEIGHT", 0.15),
        llm_timeout=_float("LLM_TIMEOUT", 90.0),
        index_ttl_seconds=_float("INDEX_TTL_SECONDS", 5.0),
    )
