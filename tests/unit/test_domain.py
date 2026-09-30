"""Tests for the frozen domain models."""

from dataclasses import FrozenInstanceError

import pytest

from rag_generator.domain import Answer, Chunk, Citation, Document, ScoredChunk


def make_chunk() -> Chunk:
    return Chunk(source="leave_policy.md", index=3, text="Employees get 24 days.")


def test_document_is_frozen_when_mutated() -> None:
    document = Document(source="leave_policy.md", text="Employees get 24 days.")
    with pytest.raises(FrozenInstanceError):
        document.text = "changed"  # type: ignore[misc]


def test_chunk_is_frozen_when_mutated() -> None:
    chunk = make_chunk()
    with pytest.raises(FrozenInstanceError):
        chunk.index = 99  # type: ignore[misc]


def test_scored_chunk_is_frozen_when_mutated() -> None:
    scored = ScoredChunk(chunk=make_chunk(), score=0.82)
    with pytest.raises(FrozenInstanceError):
        scored.score = 0.1  # type: ignore[misc]


def test_citation_is_frozen_when_mutated() -> None:
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82)
    with pytest.raises(FrozenInstanceError):
        citation.score = 0.1  # type: ignore[misc]


def test_answer_is_frozen_when_mutated() -> None:
    answer = Answer(text="24 days [1].", citations=[], grounded=True)
    with pytest.raises(FrozenInstanceError):
        answer.grounded = False  # type: ignore[misc]


def test_citation_exposes_source_chunk_index_and_score() -> None:
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82)
    assert (citation.source, citation.chunk_index, citation.score) == ("leave_policy.md", 3, 0.82)


def test_answer_grounded_is_true_when_built_from_context() -> None:
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82)
    answer = Answer(text="24 days [1].", citations=[citation], grounded=True)
    assert answer.grounded is True
    assert answer.citations == [citation]


def test_answer_grounded_is_false_when_nothing_retrieved() -> None:
    answer = Answer(text="Not found.", citations=[], grounded=False)
    assert answer.grounded is False
    assert answer.citations == []


def test_scored_chunk_exposes_chunk_and_score() -> None:
    chunk = make_chunk()
    scored = ScoredChunk(chunk=chunk, score=0.82)
    assert (scored.chunk, scored.score) == (chunk, 0.82)
