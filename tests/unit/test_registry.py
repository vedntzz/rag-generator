"""Tests for LoaderRegistry: extension dispatch and file/directory loading."""

from pathlib import Path

import pytest

from rag_generator.domain import Document
from rag_generator.errors import UnsupportedFileTypeError
from rag_generator.loaders.pdf_loader import PdfLoader
from rag_generator.loaders.registry import LoaderRegistry
from rag_generator.loaders.text_loader import TextLoader


@pytest.fixture
def registry() -> LoaderRegistry:
    registry = LoaderRegistry()
    registry.register(".txt", TextLoader())
    registry.register(".md", TextLoader())
    registry.register(".pdf", PdfLoader())
    return registry


def write(path: Path, text: str = "content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_load_path_returns_single_document_for_file(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    path = write(tmp_path / "policy.txt", "24 days")
    assert registry.load_path(path) == [Document(source="policy.txt", text="24 days")]


def test_registry_load_path_raises_for_unsupported_extension(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    path = write(tmp_path / "budget.xlsx")
    with pytest.raises(UnsupportedFileTypeError):
        registry.load_path(path)


def test_registry_load_path_matches_extension_case_insensitively(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    path = write(tmp_path / "NOTES.MD", "hello")
    assert registry.load_path(path)[0].text == "hello"


def test_registry_load_path_returns_empty_document_for_zero_byte_file(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    path = tmp_path / "empty.pdf"
    path.touch()
    assert registry.load_path(path) == [Document(source="empty.pdf", text="")]


def test_registry_load_path_recurses_into_subdirectories(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    write(tmp_path / "a.txt")
    write(tmp_path / "nested" / "deeper" / "b.md")
    assert len(registry.load_path(tmp_path)) == 2


def test_registry_load_path_skips_unsupported_files_in_directory(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    write(tmp_path / "a.txt")
    write(tmp_path / "budget.xlsx")
    write(tmp_path / ".DS_Store")
    assert [doc.source for doc in registry.load_path(tmp_path)] == ["a.txt"]


def test_registry_load_path_sorts_directory_documents_by_relative_path(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    for name in ["b.txt", "sub/c.txt", "a.md"]:
        write(tmp_path / name)
    sources = [doc.source for doc in registry.load_path(tmp_path)]
    assert sources == ["a.md", "b.txt", "sub/c.txt"]


def test_registry_load_path_returns_empty_list_for_directory_without_supported_files(
    registry: LoaderRegistry, tmp_path: Path
) -> None:
    write(tmp_path / "budget.xlsx")
    assert registry.load_path(tmp_path) == []
