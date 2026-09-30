"""Tests for the grounding prompts."""

from rag_generator.domain import Chunk, ScoredChunk
from rag_generator.llm.prompts import NOT_FOUND_MESSAGE, build_system_prompt, build_user_prompt


def scored(source: str, text: str) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(source=source, index=0, text=text), score=0.9)


def test_not_found_message_is_non_empty_sentence() -> None:
    assert NOT_FOUND_MESSAGE.strip() and NOT_FOUND_MESSAGE.endswith(".")


def test_system_prompt_restricts_answers_to_numbered_context() -> None:
    prompt = build_system_prompt().lower()
    assert "only" in prompt and "numbered context" in prompt


def test_system_prompt_requires_bracketed_citations() -> None:
    assert "[n]" in build_system_prompt()


def test_system_prompt_requires_exact_not_found_reply() -> None:
    prompt = build_system_prompt()
    assert f'"{NOT_FOUND_MESSAGE}"' in prompt and "exactly" in prompt.lower()


def test_user_prompt_numbers_chunks_with_source_labels_then_question() -> None:
    chunks = [scored("leave.md", "24 days of leave."), scored("sick.md", "10 sick days.")]
    assert build_user_prompt("How much leave?", chunks) == (
        "Context:\n\n"
        "[1] Source: leave.md\n24 days of leave.\n\n"
        "[2] Source: sick.md\n10 sick days.\n\n"
        "Question: How much leave?"
    )


def test_user_prompt_keeps_retrieval_order_in_numbering() -> None:
    prompt = build_user_prompt("q", [scored("b.md", "B"), scored("a.md", "A")])
    assert prompt.index("[1] Source: b.md") < prompt.index("[2] Source: a.md")
