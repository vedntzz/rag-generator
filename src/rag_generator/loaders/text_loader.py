"""Loader for plain-text formats (.txt, .md)."""

from pathlib import Path

from rag_generator.domain import Document


class TextLoader:
    def load(self, path: Path) -> Document:
        return Document(source=path.name, text=path.read_text(encoding="utf-8"))
