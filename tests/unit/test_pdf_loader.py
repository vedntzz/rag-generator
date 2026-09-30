"""Tests for the pypdf-backed loader."""

from pathlib import Path

from rag_generator.loaders.pdf_loader import PdfLoader
from tests.document_builders import write_blank_pdf, write_text_pdf


def test_pdf_loader_extracts_page_text(tmp_path: Path) -> None:
    path = write_text_pdf(tmp_path / "leave.pdf", ["Employees get 24 days of leave."])
    assert "Employees get 24 days of leave." in PdfLoader().load(path).text


def test_pdf_loader_joins_pages_in_order_with_blank_line(tmp_path: Path) -> None:
    path = write_text_pdf(tmp_path / "leave.pdf", ["Page one.", "Page two."])
    assert PdfLoader().load(path).text == "Page one.\n\nPage two."


def test_pdf_loader_uses_file_name_as_source(tmp_path: Path) -> None:
    path = write_text_pdf(tmp_path / "leave.pdf", ["x"])
    assert PdfLoader().load(path).source == "leave.pdf"


def test_pdf_loader_returns_empty_text_for_blank_page(tmp_path: Path) -> None:
    path = write_blank_pdf(tmp_path / "blank.pdf")
    assert PdfLoader().load(path).text == ""
