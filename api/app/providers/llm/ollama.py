"""Ollama provider: ollama-host or ollama-incluster (section 3.5).

Only OLLAMA_URL differs between the two modes: host.minikube.internal:11434 for the
laptop, the Ollama Service name for the in-cluster Deployment.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from ..llm.base import ProviderError, parse_answer
from ..llm.usage import observe_tokens
from ...services.prompt import build_messages


class OllamaProvider:
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # An explicit timeout: a hung model must not hold a worker forever.
        self.client = httpx.Client(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(settings.llm_timeout, connect=5.0),
        )

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        payload = {
            "model": self.settings.llm_model,
            "stream": False,
            # Strongest enforcement Ollama offers: decoding is constrained to this schema.
            "format": Answer.model_json_schema(),
            "options": {"temperature": self.settings.temperature, "num_predict": 700},
            "messages": build_messages(question, passages),
        }
        try:
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"ollama call failed: {exc!r}") from exc
        observe_tokens(int(data.get("prompt_eval_count", 0)), int(data.get("eval_count", 0)))
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise ProviderError("ollama returned an empty message")
        return parse_answer(content)   # pydantic ValidationError escapes on bad structure
