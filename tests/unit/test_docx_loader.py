"""Tests for the python-docx-backed loader."""

from pathlib import Path

from rag_generator.loaders.docx_loader import DocxLoader
from tests.document_builders import write_docx


def test_docx_loader_joins_paragraphs_with_blank_line(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "plan.docx", ["Pro plan.", "Includes SSO."])
    assert DocxLoader().load(path).text == "Pro plan.\n\nIncludes SSO."


def test_docx_loader_skips_empty_paragraphs(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "plan.docx", ["Pro plan.", "", "Includes SSO."])
    assert DocxLoader().load(path).text == "Pro plan.\n\nIncludes SSO."


def test_docx_loader_uses_file_name_as_source(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "plan.docx", ["x"])
    assert DocxLoader().load(path).source == "plan.docx"


def test_docx_loader_returns_empty_text_when_document_has_no_paragraphs(tmp_path: Path) -> None:
    path = write_docx(tmp_path / "empty.docx", [])
    assert DocxLoader().load(path).text == ""
