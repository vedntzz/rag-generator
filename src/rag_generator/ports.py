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
    """Collection-scoped vector repository.

    add() replaces by source: every stored chunk whose source appears in the incoming chunks
    is deleted first, then the incoming chunks are inserted (no stale tail chunks survive).
    add() raises ValueError when len(chunks) != len(vectors).
    search() returns up to top_k chunks by descending cosine score; raises
    CollectionNotFoundError for a missing collection. has_collection() returns False instead.
    Collection names must match ^[A-Za-z0-9_-]{1,64}$; methods taking a name raise
    InvalidCollectionNameError otherwise.
    """

    def add(self, collection: str, chunks: Sequence[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[ScoredChunk]: ...

    def has_collection(self, collection: str) -> bool: ...

    def list_collections(self) -> list[str]: ...


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...
