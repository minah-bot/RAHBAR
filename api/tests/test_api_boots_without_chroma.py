"""Engineering note 2: liveness must not depend on Chroma.

In CI every API pod crashed in a loop while Chroma was down, because the Chroma client connects when
it is constructed. The API must boot, answer /healthz, and report /readyz 503 naming Chroma."""

import pytest

from app.config import Settings
from app.retrieval.chroma import ChromaRetriever
from app.retrieval.index import IndexNotReady


def test_constructing_the_retriever_does_not_connect() -> None:
    ChromaRetriever(Settings(chroma_host="127.0.0.1", chroma_port=1))   # nothing listens on port 1


def test_readiness_and_queries_name_chroma_when_it_is_down() -> None:
    retriever = ChromaRetriever(Settings(chroma_host="127.0.0.1", chroma_port=1))
    with pytest.raises(IndexNotReady) as err:
        retriever.check_ready()
    assert err.value.dependency == "chroma"
    with pytest.raises(IndexNotReady):
        retriever.retrieve("What is the attendance rule?", 5)
