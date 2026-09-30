"""Loader for text-based PDFs (no OCR)."""

from pathlib import Path

from pypdf import PdfReader

from rag_generator.domain import Document

PAGE_SEPARATOR = "\n\n"


class PdfLoader:
    def load(self, path: Path) -> Document:
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        text = PAGE_SEPARATOR.join(page for page in pages if page.strip())
        return Document(source=path.name, text=text)
