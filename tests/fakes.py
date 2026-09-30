"""Deterministic, network-free stand-ins for the Embedder and LLM ports."""

import math
import re
import zlib
from collections.abc import Sequence
from typing import NamedTuple

WORD_PATTERN = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words embedder: each word is hashed (stably, via crc32) into a bucket."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        counts = [0.0] * self.dimension
        for word in WORD_PATTERN.findall(text.lower()):
            counts[zlib.crc32(word.encode()) % self.dimension] += 1.0
        return normalize_to_unit_length(counts)


def normalize_to_unit_length(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


class RecordedPrompt(NamedTuple):
    system: str
    user: str


class FakeLLM:
    """Records every (system, user) prompt pair and always returns the same canned answer."""

    def __init__(self, answer: str = "Canned answer [1].") -> None:
        self.answer = answer
        self.prompts: list[RecordedPrompt] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append(RecordedPrompt(system, user))
        return self.answer
