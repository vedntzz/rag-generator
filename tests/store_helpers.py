"""Shared helpers for vector store tests."""

from rag_generator.domain import Chunk, ScoredChunk
from rag_generator.store.numpy_store import NumpyVectorStore


def chunk(source: str, index: int = 0) -> Chunk:
    return Chunk(source=source, index=index, text=f"{source}#{index}")


def add_axis_chunks(store: NumpyVectorStore, collection: str = "hr") -> None:
    chunks = [chunk("x.md"), chunk("y.md"), chunk("xy.md")]
    store.add(collection, chunks, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def sources(results: list[ScoredChunk]) -> list[str]:
    return [scored.chunk.source for scored in results]
