"""Brute-force cosine vector store, persisted as one collection.npz per collection."""

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import numpy as np

from rag_generator.domain import Chunk, ScoredChunk
from rag_generator.errors import CollectionNotFoundError, InvalidCollectionNameError

COLLECTION_FILE = "collection.npz"
COLLECTION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
Matrix = np.ndarray


class NumpyVectorStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def add(self, collection: str, chunks: Sequence[Chunk], vectors: list[list[float]]) -> None:
        path = self._collection_path(collection)
        if len(chunks) != len(vectors):
            raise ValueError(f"Got {len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return
        stored = load_collection(path) if path.is_file() else None
        merged = replace_sources(stored, list(chunks), np.asarray(vectors, dtype=np.float32))
        save_collection(path, *merged)

    def search(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> list[ScoredChunk]:
        path = self._collection_path(collection)
        if not path.is_file():
            raise CollectionNotFoundError(collection)
        chunks, matrix = load_collection(path)
        scores = cosine_scores(matrix, np.asarray(query_vector, dtype=np.float32))
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        return [ScoredChunk(chunk=chunks[i], score=float(scores[i])) for i in ranked]

    def has_collection(self, collection: str) -> bool:
        return self._collection_path(collection).is_file()

    def list_collections(self) -> list[str]:
        if not self.data_dir.is_dir():
            return []
        names = [p.name for p in self.data_dir.iterdir() if is_valid_collection_name(p.name)]
        return sorted(name for name in names if self.has_collection(name))

    def _collection_path(self, collection: str) -> Path:
        # Validating here keeps every public method from touching paths outside data_dir.
        if not is_valid_collection_name(collection):
            raise InvalidCollectionNameError(collection)
        return self.data_dir / collection / COLLECTION_FILE


def is_valid_collection_name(name: str) -> bool:
    return COLLECTION_NAME_PATTERN.fullmatch(name) is not None


def load_collection(path: Path) -> tuple[list[Chunk], Matrix]:
    with np.load(path, allow_pickle=False) as archive:
        records: list[str] = archive["chunks"].tolist()
        return [Chunk(**json.loads(record)) for record in records], archive["vectors"]


def save_collection(path: Path, chunks: list[Chunk], matrix: Matrix) -> None:
    # One file holds vectors and chunks, so a single rename swaps both together.
    path.parent.mkdir(parents=True, exist_ok=True)
    records = np.array([json.dumps(asdict(chunk), ensure_ascii=False) for chunk in chunks])
    write_atomically(
        path, lambda file: np.savez(file, allow_pickle=False, vectors=matrix, chunks=records)
    )


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
