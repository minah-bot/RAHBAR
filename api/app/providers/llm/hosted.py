"""Hosted free-tier provider: any OpenAI-compatible endpoint (Groq, OpenRouter, ...).

The key arrives as LLM_API_KEY from a Kubernetes Secret created on the command line.
It is never logged and never part of an exception message raised here.
"""

from __future__ import annotations

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from ...services.prompt import build_messages
from ..llm.base import ProviderError, parse_answer
from ..llm.usage import observe_tokens


class HostedProvider:
    name = "hosted"

    def __init__(self, settings: Settings) -> None:
        if not settings.hosted_base_url or not settings.llm_api_key:
            raise ValueError("hosted provider needs HOSTED_BASE_URL and LLM_API_KEY")
        self.settings = settings
        self.client = httpx.Client(
            base_url=settings.hosted_base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=httpx.Timeout(settings.llm_timeout, connect=5.0),
        )

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        payload = {
            "model": self.settings.llm_model,
            "temperature": self.settings.temperature,
            "max_tokens": 700,
            # JSON mode; the schema itself travels in the system prompt (with_schema=True).
            "response_format": {"type": "json_object"},
            "messages": build_messages(question, passages, with_schema=True),
        }
        try:
            response = self.client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise ProviderError(f"hosted call failed: {type(exc).__name__}") from exc
        usage = data.get("usage") or {}
        observe_tokens(int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
        if not content:
            raise ProviderError("hosted provider returned an empty message")
        return parse_answer(content)
