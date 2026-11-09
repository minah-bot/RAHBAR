"""Token accounting shared by the providers (section 4.8: tokens in and out per request)."""

from __future__ import annotations

from ...routes import metrics


def observe_tokens(tokens_in: int, tokens_out: int) -> None:
    metrics.TOKENS_IN.observe(tokens_in)
    metrics.TOKENS_OUT.observe(tokens_out)
