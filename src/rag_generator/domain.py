"""Immutable domain models shared by every layer."""

import re
from dataclasses import dataclass

from rag_generator.errors import InvalidCollectionNameError

COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class Document:
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    source: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    truncated: bool = False


@dataclass(frozen=True)
class LlmReply:
    text: str
    truncated: bool = False


def is_valid_collection_name(name: str) -> bool:
    return COLLECTION_NAME_PATTERN.fullmatch(name) is not None


def validate_collection_name(name: str) -> None:
    if not is_valid_collection_name(name):
        raise InvalidCollectionNameError(name)
