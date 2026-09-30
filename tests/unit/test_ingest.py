"""Tests for IngestService: load -> chunk -> embed -> store."""

from pathlib import Path

import pytest

from rag_generator.chunking.recursive_chunker import RecursiveCharacterChunker
from rag_generator.errors import UnsupportedFileTypeError
from rag_generator.loaders.registry import LoaderRegistry
from rag_generator.loaders.text_loader import TextLoader
from rag_generator.pipeline.ingest import IngestService
from rag_generator.store.numpy_store import NumpyVectorStore
from tests.fakes import FakeEmbedder


@pytest.fixture
def ingest_service(store: NumpyVectorStore) -> IngestService:
    registry = LoaderRegistry()
    registry.register(".md", TextLoader())
    chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=5)
    return IngestService(registry, chunker, FakeEmbedder(), store)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def stored_sources(store: NumpyVectorStore) -> list[str]:
    results = store.search("hr", FakeEmbedder().embed_query("leave"), top_k=100)
    return sorted({scored.chunk.source for scored in results})


def test_ingest_returns_number_of_chunks_indexed(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    path = write(tmp_path / "docs" / "leave.md", "Annual leave is 24 days. " * 4)
    indexed = ingest_service.ingest("hr", [path])
    assert indexed > 1
    assert indexed == len(store.search("hr", [1.0] * 64, top_k=100))


def test_ingest_stores_chunks_under_collection(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    ingest_service.ingest("hr", [write(tmp_path / "docs" / "leave.md", "Annual leave.")])
    assert stored_sources(store) == ["leave.md"]


def test_ingest_loads_every_given_path(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    directory = tmp_path / "docs"
    write(directory / "a.md", "Leave A.")
    write(directory / "nested" / "b.md", "Leave B.")
    single = write(tmp_path / "c.md", "Leave C.")
    assert ingest_service.ingest("hr", [directory, single]) == 3
    assert stored_sources(store) == ["a.md", "c.md", "nested/b.md"]


def test_ingest_of_directory_without_documents_indexes_nothing(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    (tmp_path / "empty").mkdir()
    assert ingest_service.ingest("hr", [tmp_path / "empty"]) == 0
    assert store.has_collection("hr") is False


def test_ingest_raises_for_unsupported_file_and_stores_nothing(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    good = write(tmp_path / "a.md", "Leave.")
    bad = write(tmp_path / "budget.xlsx", "x")
    with pytest.raises(UnsupportedFileTypeError):
        ingest_service.ingest("hr", [good, bad])
    assert store.has_collection("hr") is False


def test_ingest_reingesting_shorter_document_leaves_no_stale_chunks(
    ingest_service: IngestService, store: NumpyVectorStore, tmp_path: Path
) -> None:
    path = write(tmp_path / "leave.md", "Annual leave is 24 days. " * 4)
    ingest_service.ingest("hr", [path])
    write(path, "Annual leave is 30 days.")
    ingest_service.ingest("hr", [path])
    assert len(store.search("hr", [1.0] * 64, top_k=100)) == 1
