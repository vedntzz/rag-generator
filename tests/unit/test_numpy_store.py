"""Tests for NumpyVectorStore."""

import math
from pathlib import Path

import pytest

from rag_generator.domain import Chunk, ScoredChunk
from rag_generator.errors import CollectionNotFoundError
from rag_generator.store.numpy_store import NumpyVectorStore


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(data_dir: Path) -> NumpyVectorStore:
    return NumpyVectorStore(data_dir)


def chunk(source: str, index: int = 0) -> Chunk:
    return Chunk(source=source, index=index, text=f"{source}#{index}")


def add_axis_chunks(store: NumpyVectorStore, collection: str = "hr") -> None:
    chunks = [chunk("x.md"), chunk("y.md"), chunk("xy.md")]
    store.add(collection, chunks, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def sources(results: list[ScoredChunk]) -> list[str]:
    return [scored.chunk.source for scored in results]


def test_store_search_returns_top_k_by_descending_cosine(store: NumpyVectorStore) -> None:
    add_axis_chunks(store)
    assert sources(store.search("hr", [1.0, 0.0], top_k=2)) == ["x.md", "xy.md"]


def test_store_search_scores_equal_cosine_similarity(store: NumpyVectorStore) -> None:
    add_axis_chunks(store)
    scores = [scored.score for scored in store.search("hr", [1.0, 0.0], top_k=3)]
    assert scores == pytest.approx([1.0, 1 / math.sqrt(2), 0.0], abs=1e-6)


def test_store_search_returns_all_when_top_k_exceeds_count(store: NumpyVectorStore) -> None:
    add_axis_chunks(store)
    assert len(store.search("hr", [1.0, 0.0], top_k=10)) == 3


def test_store_add_replaces_all_chunks_of_readded_source(store: NumpyVectorStore) -> None:
    store.add("hr", [chunk("a.md", i) for i in range(3)], [[1.0, 0.0]] * 3)
    store.add("hr", [chunk("a.md", 0)], [[0.0, 1.0]])
    results = store.search("hr", [0.0, 1.0], top_k=10)
    assert [(r.chunk.index, r.score) for r in results] == [(0, pytest.approx(1.0))]


def test_store_add_keeps_other_sources_when_readding_one(store: NumpyVectorStore) -> None:
    store.add("hr", [chunk("a.md"), chunk("b.md")], [[1.0, 0.0], [0.0, 1.0]])
    store.add("hr", [chunk("a.md")], [[1.0, 1.0]])
    assert sorted(sources(store.search("hr", [1.0, 0.0], top_k=10))) == ["a.md", "b.md"]


def test_store_add_raises_when_chunk_and_vector_counts_differ(store: NumpyVectorStore) -> None:
    with pytest.raises(ValueError):
        store.add("hr", [chunk("a.md"), chunk("b.md")], [[1.0, 0.0]])


def test_store_persists_vectors_and_chunks_files(store: NumpyVectorStore, data_dir: Path) -> None:
    add_axis_chunks(store)
    names = sorted(path.name for path in (data_dir / "hr").iterdir())
    assert names == ["chunks.json", "vectors.npy"]


def test_store_fresh_instance_reloads_persisted_collection(
    store: NumpyVectorStore, data_dir: Path
) -> None:
    add_axis_chunks(store)
    reloaded = NumpyVectorStore(data_dir).search("hr", [1.0, 0.0], top_k=1)
    assert [r.chunk for r in reloaded] == [chunk("x.md")]


def test_store_isolates_collections(store: NumpyVectorStore) -> None:
    store.add("hr", [chunk("leave.md")], [[1.0, 0.0]])
    store.add("product", [chunk("pricing.md")], [[1.0, 0.0]])
    assert sources(store.search("hr", [1.0, 0.0], top_k=10)) == ["leave.md"]


def test_store_list_collections_is_sorted(store: NumpyVectorStore) -> None:
    store.add("product", [chunk("p.md")], [[1.0, 0.0]])
    store.add("hr", [chunk("h.md")], [[1.0, 0.0]])
    assert store.list_collections() == ["hr", "product"]


def test_store_list_collections_is_empty_before_any_add(store: NumpyVectorStore) -> None:
    assert store.list_collections() == []


def test_store_has_collection_is_false_when_missing(store: NumpyVectorStore) -> None:
    assert store.has_collection("hr") is False


def test_store_has_collection_is_true_after_add(store: NumpyVectorStore) -> None:
    add_axis_chunks(store)
    assert store.has_collection("hr") is True


def test_store_search_raises_when_collection_missing(store: NumpyVectorStore) -> None:
    with pytest.raises(CollectionNotFoundError):
        store.search("hr", [1.0, 0.0], top_k=5)


def test_store_search_scores_zero_query_vector_as_zero(store: NumpyVectorStore) -> None:
    add_axis_chunks(store)
    scores = [r.score for r in store.search("hr", [0.0, 0.0], top_k=3)]
    assert scores == [0.0, 0.0, 0.0]


def test_store_search_scores_zero_stored_vector_as_zero(store: NumpyVectorStore) -> None:
    store.add("hr", [chunk("blank.md")], [[0.0, 0.0]])
    assert store.search("hr", [1.0, 0.0], top_k=1)[0].score == 0.0
