"""Tests for the frozen domain models."""

from dataclasses import FrozenInstanceError

import pytest

from rag_generator.domain import Answer, Chunk, Citation, Document, LlmReply, ScoredChunk


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
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82, reference=1)
    with pytest.raises(FrozenInstanceError):
        citation.score = 0.1  # type: ignore[misc]


def test_answer_is_frozen_when_mutated() -> None:
    answer = Answer(text="24 days [1].", citations=[], grounded=True)
    with pytest.raises(FrozenInstanceError):
        answer.grounded = False  # type: ignore[misc]


def test_citation_exposes_source_chunk_index_score_and_reference() -> None:
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82, reference=1)
    fields = (citation.source, citation.chunk_index, citation.score, citation.reference)
    assert fields == ("leave_policy.md", 3, 0.82, 1)


def test_answer_grounded_is_true_when_built_from_context() -> None:
    citation = Citation(source="leave_policy.md", chunk_index=3, score=0.82, reference=1)
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


def test_answer_truncated_defaults_to_false() -> None:
    assert Answer(text="24 days [1].", citations=[], grounded=True).truncated is False


def test_answer_truncated_can_be_flagged() -> None:
    assert Answer(text="24 days", citations=[], grounded=True, truncated=True).truncated is True


def test_llm_reply_is_frozen_when_mutated() -> None:
    reply = LlmReply(text="ok")
    with pytest.raises(FrozenInstanceError):
        reply.text = "changed"  # type: ignore[misc]


def test_llm_reply_truncated_defaults_to_false() -> None:
    assert LlmReply(text="ok").truncated is False
