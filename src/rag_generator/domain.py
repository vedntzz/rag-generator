"""Immutable domain models shared by every layer."""

from dataclasses import dataclass


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
