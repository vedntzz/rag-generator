"""Tests for LazyLLM, which defers building the real LLM until first use."""

from rag_generator.domain import LlmReply
from rag_generator.llm.lazy_llm import LazyLLM
from tests.fakes import FakeLLM


class CountingBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> FakeLLM:
        self.calls += 1
        return FakeLLM("ok [1].")


def test_lazy_llm_does_not_build_until_first_complete() -> None:
    builder = CountingBuilder()
    LazyLLM(builder)
    assert builder.calls == 0


def test_lazy_llm_builds_once_across_calls() -> None:
    builder = CountingBuilder()
    llm = LazyLLM(builder)
    llm.complete("s", "u1")
    llm.complete("s", "u2")
    assert builder.calls == 1


def test_lazy_llm_delegates_complete() -> None:
    assert LazyLLM(CountingBuilder()).complete("s", "u") == LlmReply("ok [1].")
