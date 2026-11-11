"""The contracts. Change them only with your partner's review: the UI, the
evaluation script and your tests all depend on these shapes.

`Answer` is what the model must produce (section 3.4). `AskResponse` is what
the API returns to the UI and to eval/run_eval.py. The model's `reasoning` is
never sent to the user.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    follow_up: bool = False   # follow-ups skip the cache and are never stored in it (3.3)

    @field_validator("question")
    @classmethod
    def question_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least three non-whitespace characters")
        return value


class Passage(BaseModel):
    """One retrieved chunk, as the retriever hands it to the generator."""

    chunk_id: str
    text: str
    source: str
    section: str
    page: int | None = None
    score: float = Field(description="Relevance, higher is closer: cosine similarity blended with exact-term overlap.")
    doc_version: str = ""        # lets the prompt label an older version as such
    superseded: bool = False


class Citation(BaseModel):
    """Public citation metadata, created by the server from retrieved passages."""
    source: str
    section: str
    page: int | None = None


class PassageCitation(BaseModel):
    """Model output can identify evidence, but cannot supply citation metadata."""
    passage_id: str


class Answer(BaseModel):
    """The model's output contract. Send its JSON schema to the model, then validate."""

    reasoning: str
    answer: str
    citations: list[PassageCitation]
    answered_from_context: bool
    confidence: Literal["high", "medium", "low"]


class RetrievedRef(BaseModel):
    """What the response says was retrieved. eval/run_eval.py scores against this."""

    chunk_id: str
    source: str
    section: str
    page: int | None = None
    score: float


class AskResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[Citation]
    answered_from_context: bool
    confidence: Literal["high", "medium", "low"]
    retrieved: list[RetrievedRef]
    provider: str
    cached: bool = False   # added to the starter contract: the UI shows a cache chip (ADR 0002)
