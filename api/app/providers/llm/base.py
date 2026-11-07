"""The LLM provider interface (section 3.5). Swapping providers is one file."""

from __future__ import annotations

from typing import Protocol

from ...schemas import Answer, Passage


class ProviderError(RuntimeError):
    """The model backend failed (network, timeout, HTTP error, empty reply)."""


class LLMProvider(Protocol):
    name: str

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        """Answer the question from the passages, as an `Answer`.

        Raise ProviderError for transport failures and let pydantic's ValidationError
        escape for malformed output: the caller retries once, then falls back."""
        ...


def parse_answer(text: str) -> Answer:
    """Validate the model's reply against the Answer schema. Models sometimes wrap JSON
    in a code fence even when told not to; strip it, then validate strictly."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`")
        body = body[4:] if body.lower().startswith("json") else body
    return Answer.model_validate_json(body.strip())
