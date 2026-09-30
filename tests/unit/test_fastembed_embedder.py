"""Tests for FastEmbedEmbedder, using a fake in place of fastembed's TextEmbedding."""

from collections.abc import Iterable, Iterator

import numpy as np

from rag_generator.embedding.fastembed_embedder import FastEmbedEmbedder


class FakeTextEmbedding:
    """Mimics fastembed.TextEmbedding: generators of float32 arrays, calls recorded."""

    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def passage_embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        self.passage_calls.append(list(texts))
        return (np.array([1.0, float(i)], dtype=np.float32) for i, _ in enumerate(texts))

    def query_embed(self, query: str) -> Iterator[np.ndarray]:
        self.query_calls.append(query)
        return iter([np.array([0.5, 0.25], dtype=np.float32)])


def test_embedder_embed_documents_uses_passage_embedding() -> None:
    model = FakeTextEmbedding()
    FastEmbedEmbedder(model).embed_documents(["a", "b"])
    assert (model.passage_calls, model.query_calls) == ([["a", "b"]], [])


def test_embedder_embed_documents_returns_one_float_list_per_text() -> None:
    vectors = FastEmbedEmbedder(FakeTextEmbedding()).embed_documents(["a", "b"])
    assert vectors == [[1.0, 0.0], [1.0, 1.0]]
    assert all(type(value) is float for vector in vectors for value in vector)


def test_embedder_embed_documents_returns_empty_list_without_calling_model() -> None:
    model = FakeTextEmbedding()
    assert FastEmbedEmbedder(model).embed_documents([]) == []
    assert model.passage_calls == []


def test_embedder_embed_query_uses_query_embedding() -> None:
    model = FakeTextEmbedding()
    FastEmbedEmbedder(model).embed_query("how many leave days?")
    assert (model.query_calls, model.passage_calls) == (["how many leave days?"], [])


def test_embedder_embed_query_returns_float_list() -> None:
    vector = FastEmbedEmbedder(FakeTextEmbedding()).embed_query("q")
    assert vector == [0.5, 0.25]
    assert all(type(value) is float for value in vector)
