"""Tests for AnswerService's grounding contract."""

from pathlib import Path

import pytest

from rag_generator.domain import Answer, Chunk, Citation, ScoredChunk
from rag_generator.errors import CollectionNotFoundError
from rag_generator.llm.prompts import NOT_FOUND_MESSAGE, build_system_prompt, build_user_prompt
from rag_generator.pipeline.answer import AnswerService
from rag_generator.store.numpy_store import NumpyVectorStore
from tests.fakes import FakeEmbedder, FakeLLM, RecordedPrompt


class StubStore:
    """Returns preset hits and records every search."""

    def __init__(self, hits: list[ScoredChunk]) -> None:
        self.hits = hits
        self.searches: list[tuple[str, int]] = []

    def has_collection(self, collection: str) -> bool:
        return True

    def search(self, collection: str, query_vector: list[float], top_k: int) -> list[ScoredChunk]:
        self.searches.append((collection, top_k))
        return self.hits[:top_k]


def hit(source: str, score: float, index: int = 0) -> ScoredChunk:
    return ScoredChunk(Chunk(source=source, index=index, text=f"text of {source}"), score)


THREE_HITS = [hit("a.md", 0.9), hit("b.md", 0.8, index=2), hit("c.md", 0.7)]


def ask(hits: list[ScoredChunk], llm: FakeLLM, min_score: float = 0.3) -> Answer:
    service = AnswerService(FakeEmbedder(), StubStore(hits), llm, top_k=5, min_score=min_score)
    return service.ask("hr", "How much leave?")


def test_answer_returns_not_found_without_llm_call_when_nothing_above_min_score() -> None:
    llm = FakeLLM()
    answer = ask([hit("a.md", 0.29), hit("b.md", 0.1)], llm)
    assert answer == Answer(NOT_FOUND_MESSAGE, [], grounded=False)
    assert llm.prompts == []


def test_answer_returns_not_found_without_llm_call_when_store_returns_nothing() -> None:
    llm = FakeLLM()
    assert ask([], llm).grounded is False and llm.prompts == []


def test_answer_keeps_chunk_scoring_exactly_min_score() -> None:
    llm = FakeLLM("Leave [1].")
    assert ask([hit("a.md", 0.3)], llm).grounded is True


def test_answer_prompts_llm_with_only_chunks_above_min_score() -> None:
    llm = FakeLLM("Leave [1].")
    ask([hit("a.md", 0.9), hit("b.md", 0.1)], llm)
    expected_user = build_user_prompt("How much leave?", [hit("a.md", 0.9)])
    assert llm.prompts == [RecordedPrompt(build_system_prompt(), expected_user)]


def test_answer_searches_collection_with_configured_top_k() -> None:
    store = StubStore(THREE_HITS)
    AnswerService(FakeEmbedder(), store, FakeLLM(), top_k=2, min_score=0.3).ask("hr", "q")
    assert store.searches == [("hr", 2)]


@pytest.mark.parametrize("reply", ["", "   ", NOT_FOUND_MESSAGE, f"  {NOT_FOUND_MESSAGE}\n"])
def test_answer_is_ungrounded_not_found_when_reply_empty_or_not_found(reply: str) -> None:
    assert ask(THREE_HITS, FakeLLM(reply)) == Answer(NOT_FOUND_MESSAGE, [], grounded=False)


def test_answer_cites_only_chunks_referenced_in_reply() -> None:
    answer = ask(THREE_HITS, FakeLLM("24 days [1]; carry-over [3]."))
    assert answer.citations == [Citation("a.md", 0, 0.9, 1), Citation("c.md", 0, 0.7, 3)]
    assert answer.grounded is True


def test_answer_citation_carries_chunk_index_score_and_reference() -> None:
    answer = ask(THREE_HITS, FakeLLM("Sick leave [2]."))
    assert answer.citations == [Citation(source="b.md", chunk_index=2, score=0.8, reference=2)]


def test_answer_cites_repeated_reference_once() -> None:
    answer = ask(THREE_HITS, FakeLLM("A [2]. B [2][2]."))
    assert [c.source for c in answer.citations] == ["b.md"]


def test_answer_ignores_out_of_range_references() -> None:
    answer = ask(THREE_HITS, FakeLLM("Made up [0] and [4]."))
    assert (answer.citations, answer.grounded) == ([], False)


def test_answer_keeps_valid_references_alongside_out_of_range_ones() -> None:
    answer = ask(THREE_HITS, FakeLLM("Real [1], fake [9]."))
    assert [c.source for c in answer.citations] == ["a.md"]


def test_answer_without_references_is_ungrounded_but_keeps_text() -> None:
    answer = ask(THREE_HITS, FakeLLM("Employees get 24 days."))
    assert answer == Answer("Employees get 24 days.", [], grounded=False)


def test_answer_returns_text_and_flags_truncated_reply() -> None:
    answer = ask(THREE_HITS, FakeLLM("Employees get 24 days [1] and", truncated=True))
    assert (answer.text, answer.truncated, answer.grounded) == (
        "Employees get 24 days [1] and", True, True
    )


def test_answer_is_not_truncated_for_complete_reply() -> None:
    assert ask(THREE_HITS, FakeLLM("24 days [1].")).truncated is False


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed_query(text)


def test_answer_raises_for_missing_collection_without_embedding(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    service = AnswerService(embedder, NumpyVectorStore(tmp_path), FakeLLM(), 5, 0.3)
    with pytest.raises(CollectionNotFoundError):
        service.ask("hr", "q")
    assert embedder.calls == 0
