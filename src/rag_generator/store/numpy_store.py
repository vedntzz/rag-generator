"""Brute-force cosine vector store, persisted as vectors.npy + chunks.json per collection."""

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rag_generator.domain import Chunk, ScoredChunk
from rag_generator.errors import CollectionNotFoundError

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
Matrix = np.ndarray


class NumpyVectorStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def add(self, collection: str, chunks: Sequence[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        stored = self._load(collection) if self.has_collection(collection) else None
        merged = replace_sources(stored, list(chunks), np.asarray(vectors, dtype=np.float32))
        self._save(collection, *merged)

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[ScoredChunk]:
        if not self.has_collection(collection):
            raise CollectionNotFoundError(collection)
        chunks, matrix = self._load(collection)
        scores = cosine_scores(matrix, np.asarray(query_vector, dtype=np.float32))
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [ScoredChunk(chunk=chunks[i], score=float(scores[i])) for i in ranked]

    def has_collection(self, collection: str) -> bool:
        return (self.data_dir / collection / CHUNKS_FILE).is_file()

    def list_collections(self) -> list[str]:
        if not self.data_dir.is_dir():
            return []
        return sorted(p.name for p in self.data_dir.iterdir() if self.has_collection(p.name))

    def _load(self, collection: str) -> tuple[list[Chunk], Matrix]:
        directory = self.data_dir / collection
        records = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        return [Chunk(**record) for record in records], np.load(directory / VECTORS_FILE)

    def _save(self, collection: str, chunks: list[Chunk], matrix: Matrix) -> None:
        directory = self.data_dir / collection
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False)
        write_atomically(directory / VECTORS_FILE, lambda file: np.save(file, matrix))
        # chunks.json is written last: its presence is what marks the collection as existing.
        write_atomically(directory / CHUNKS_FILE, lambda file: file.write(payload.encode()))


def replace_sources(
    stored: tuple[list[Chunk], Matrix] | None, chunks: list[Chunk], matrix: Matrix
) -> tuple[list[Chunk], Matrix]:
    if stored is None:
        return chunks, matrix
    old_chunks, old_matrix = stored
    incoming = {chunk.source for chunk in chunks}
    keep = [i for i, chunk in enumerate(old_chunks) if chunk.source not in incoming]
    return [old_chunks[i] for i in keep] + chunks, np.vstack([old_matrix[keep], matrix])


def cosine_scores(matrix: Matrix, query: Matrix) -> Matrix:
    # Zero-length vectors score 0 instead of dividing by zero into NaN.
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    return np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 0)


def write_atomically(path: Path, write: Callable[[BinaryIO], object]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("wb") as file:
        write(file)
    os.replace(temp_path, path)
