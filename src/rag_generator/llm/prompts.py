"""Prompts that confine the LLM to retrieved, numbered context."""

from collections.abc import Sequence

from rag_generator.domain import ScoredChunk

NOT_FOUND_MESSAGE = "I couldn't find the answer in the provided documents."

SYSTEM_PROMPT = "\n".join(
    [
        "You answer questions using only the numbered context provided by the user.",
        "Rules:",
        "- Use only facts stated in the numbered context. Do not use prior knowledge.",
        "- Cite every fact with its context block number in square brackets, as [n],"
        " e.g. [1] or [2][3].",
        "- If the numbered context does not contain the answer, reply with exactly"
        f' "{NOT_FOUND_MESSAGE}" and nothing else.',
    ]
)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT


def build_user_prompt(question: str, scored_chunks: Sequence[ScoredChunk]) -> str:
    blocks = [
        f"[{number}] Source: {scored.chunk.source}\n{scored.chunk.text}"
        for number, scored in enumerate(scored_chunks, start=1)
    ]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
