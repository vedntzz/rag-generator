"""Maps file extensions to DocumentLoaders and loads files or whole directories."""

from dataclasses import replace
from pathlib import Path

from rag_generator.domain import Document
from rag_generator.errors import UnsupportedFileTypeError
from rag_generator.ports import DocumentLoader


class LoaderRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, DocumentLoader] = {}

    def register(self, extension: str, loader: DocumentLoader) -> None:
        self._loaders[extension.lower()] = loader

    def load_path(self, path: Path) -> list[Document]:
        if path.is_dir():
            return self._load_directory(path)
        return [self._load_file(path)]

    def _loader_for(self, path: Path) -> DocumentLoader:
        extension = path.suffix.lower()
        if extension not in self._loaders:
            raise UnsupportedFileTypeError(extension)
        return self._loaders[extension]

    def _load_file(self, path: Path) -> Document:
        loader = self._loader_for(path)
        # Zero-byte files are valid input; binary parsers would reject them.
        if path.stat().st_size == 0:
            return Document(source=path.name, text="")
        return loader.load(path)

    def _load_directory(self, root: Path) -> list[Document]:
        return [
            replace(self._load_file(file), source=file.relative_to(root).as_posix())
            for file in self._supported_files_under(root)
        ]

    def _supported_files_under(self, root: Path) -> list[Path]:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in self._loaders]
        return sorted(files, key=lambda file: file.relative_to(root).as_posix())
