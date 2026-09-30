"""Tests for the plain-text / markdown loader."""

from pathlib import Path

from rag_generator.loaders.text_loader import TextLoader


def test_text_loader_reads_utf8_text(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Leave: 24 days — café", encoding="utf-8")
    assert TextLoader().load(path).text == "Leave: 24 days — café"


def test_text_loader_reads_markdown_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nBody.", encoding="utf-8")
    assert TextLoader().load(path).text == "# Title\n\nBody."


def test_text_loader_uses_file_name_as_source(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("x", encoding="utf-8")
    assert TextLoader().load(path).source == "policy.txt"


def test_text_loader_returns_empty_text_for_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.touch()
    assert TextLoader().load(path).text == ""
