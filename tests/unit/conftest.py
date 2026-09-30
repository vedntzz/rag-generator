"""Fixtures shared by unit tests."""

from pathlib import Path

import pytest

from rag_generator.store.numpy_store import NumpyVectorStore


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(data_dir: Path) -> NumpyVectorStore:
    return NumpyVectorStore(data_dir)
