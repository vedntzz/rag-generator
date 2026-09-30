"""Ingest pipeline: LoaderRegistry -> Chunker -> Embedder -> VectorStore."""

from collections.abc import Sequence
from pathlib import Path

from rag_generator.domain import Chunk
from rag_generator.loaders.registry import LoaderRegistry
from rag_generator.ports import Chunker, Embedder, VectorStore


class IngestService:
    def __init__(
        self, registry: LoaderRegistry, chunker: Chunker, embedder: Embedder, store: VectorStore
    ) -> None:
        self.registry = registry
        self.chunker = chunker
        self.embedder = embedder
        self.store = store

    def ingest(self, collection: str, paths: Sequence[Path]) -> int:
        # Every path is loaded before anything is stored, so a bad file aborts the whole batch.
        chunks = self._load_and_chunk(paths)
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.store.add(collection, chunks, vectors)
        return len(chunks)

    def _load_and_chunk(self, paths: Sequence[Path]) -> list[Chunk]:
        documents = [document for path in paths for document in self.registry.load_path(path)]
        return [chunk for document in documents for chunk in self.chunker.split(document)]
