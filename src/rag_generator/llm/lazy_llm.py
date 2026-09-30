"""LLM that builds its real LLM on first use (so commands that never ask need no API key)."""

from collections.abc import Callable

from rag_generator.domain import LlmReply
from rag_generator.ports import LLM


class LazyLLM:
    def __init__(self, build_llm: Callable[[], LLM]) -> None:
        self._build_llm = build_llm
        self._llm: LLM | None = None

    def complete(self, system: str, user: str) -> LlmReply:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm.complete(system, user)
