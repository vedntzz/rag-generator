"""Protocols for every external concern; adapters implement these."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from rag_generator.domain import Chunk, Document, ScoredChunk


class DocumentLoader(Protocol):
    def load(self, path: Path) -> Document: ...


class Chunker(Protocol):
    def split(self, document: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    def add(self, collection: str, chunks: Sequence[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, collection: str, query: list[float], top_k: int) -> list[ScoredChunk]: ...

    def has_collection(self, collection: str) -> bool: ...

    def list_collections(self) -> list[str]: ...


class LLM(Protocol):
    def complete(self, prompt: str) -> str: ...
