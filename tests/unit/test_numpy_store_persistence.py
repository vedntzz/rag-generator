"""Tests for NumpyVectorStore persistence, crash safety and collection-name validation."""

import contextlib
import os
from pathlib import Path

import numpy as np
import pytest

from rag_generator.domain import Chunk
from rag_generator.errors import InvalidCollectionNameError
from rag_generator.store.numpy_store import NumpyVectorStore
from tests.store_helpers import add_axis_chunks, chunk

INVALID_NAMES = ["", "../etc", "..", "a/b", "has space", "hr\n", "hr.v2", "x" * 65]


def test_store_persists_collection_as_single_npz_file(
    store: NumpyVectorStore, data_dir: Path
) -> None:
    add_axis_chunks(store)
    assert [path.name for path in (data_dir / "hr").iterdir()] == ["collection.npz"]


def test_store_npz_holds_float32_vectors_and_json_chunks_without_pickle(
    store: NumpyVectorStore, data_dir: Path
) -> None:
    add_axis_chunks(store)
    with np.load(data_dir / "hr" / "collection.npz", allow_pickle=False) as archive:
        assert sorted(archive.files) == ["chunks", "vectors"]
        assert archive["vectors"].dtype == np.float32
        assert archive["chunks"].dtype.kind == "U"


def test_store_fresh_instance_reloads_persisted_collection(
    store: NumpyVectorStore, data_dir: Path
) -> None:
    add_axis_chunks(store)
    reloaded = NumpyVectorStore(data_dir).search("hr", [1.0, 0.0], top_k=1)
    assert [r.chunk for r in reloaded] == [chunk("x.md")]


def install_replace_that_crashes_after_first_call(monkeypatch: pytest.MonkeyPatch) -> None:
    real_replace, calls = os.replace, []

    def replace_then_crash(src: str | Path, dst: str | Path) -> None:
        calls.append(dst)
        if len(calls) > 1:
            raise OSError("simulated crash between file renames")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_then_crash)


def test_store_never_pairs_new_vectors_with_old_chunks_on_same_count_replace(
    store: NumpyVectorStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.add("hr", [Chunk("a.md", 0, "old")], [[1.0, 0.0]])
    install_replace_that_crashes_after_first_call(monkeypatch)
    with contextlib.suppress(OSError):
        store.add("hr", [Chunk("a.md", 0, "new")], [[0.0, 1.0]])
    top = NumpyVectorStore(data_dir).search("hr", [0.0, 1.0], top_k=1)[0]
    assert (top.chunk.text == "new") == (top.score == pytest.approx(1.0))


def test_store_keeps_previous_collection_when_rename_fails(
    store: NumpyVectorStore, data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.add("hr", [Chunk("a.md", 0, "old")], [[1.0, 0.0]])
    monkeypatch.setattr(os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError()))
    with contextlib.suppress(OSError):
        store.add("hr", [Chunk("a.md", 0, "new")], [[0.0, 1.0]])
    assert NumpyVectorStore(data_dir).search("hr", [1.0, 0.0], top_k=1)[0].chunk.text == "old"


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_store_add_rejects_invalid_collection_name(store: NumpyVectorStore, name: str) -> None:
    with pytest.raises(InvalidCollectionNameError):
        store.add(name, [chunk("a.md")], [[1.0, 0.0]])


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_store_search_rejects_invalid_collection_name(store: NumpyVectorStore, name: str) -> None:
    with pytest.raises(InvalidCollectionNameError):
        store.search(name, [1.0, 0.0], top_k=1)


@pytest.mark.parametrize("name", INVALID_NAMES)
def test_store_has_collection_rejects_invalid_collection_name(
    store: NumpyVectorStore, name: str
) -> None:
    with pytest.raises(InvalidCollectionNameError):
        store.has_collection(name)


def test_store_add_writes_nothing_outside_data_dir_for_traversal_name(
    store: NumpyVectorStore, tmp_path: Path
) -> None:
    with contextlib.suppress(InvalidCollectionNameError):
        store.add("../escape", [chunk("a.md")], [[1.0, 0.0]])
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("name", ["A-b_9", "x" * 64])
def test_store_accepts_valid_collection_name(store: NumpyVectorStore, name: str) -> None:
    store.add(name, [chunk("a.md")], [[1.0, 0.0]])
    assert store.has_collection(name) is True


def test_store_list_collections_ignores_stray_entries(
    store: NumpyVectorStore, data_dir: Path
) -> None:
    add_axis_chunks(store)
    (data_dir / ".DS_Store").write_text("junk")
    (data_dir / "empty-dir").mkdir()
    assert store.list_collections() == ["hr"]

