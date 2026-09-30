"""Domain errors raised by the RAG pipeline."""

from collections.abc import Sequence
from pathlib import Path


class RagError(Exception):
    """Base class for all RAG Generator errors."""


class UnsupportedFileTypeError(RagError):
    def __init__(self, extension: str) -> None:
        super().__init__(f"Unsupported file type: '{extension}'")
        self.extension = extension


class CollectionNotFoundError(RagError):
    def __init__(self, collection: str) -> None:
        super().__init__(f"Collection not found: '{collection}'")
        self.collection = collection


class InvalidCollectionNameError(RagError):
    def __init__(self, collection: str) -> None:
        super().__init__(f"Invalid collection name: '{collection}'")
        self.collection = collection


class LlmError(RagError):
    def __init__(self, detail: str) -> None:
        super().__init__(f"LLM request failed: {detail}")


class NoDocumentsFoundError(RagError):
    def __init__(self, paths: Sequence[Path]) -> None:
        super().__init__(f"No supported documents found in: {', '.join(map(str, paths))}")
        self.paths = list(paths)


class MissingApiKeyError(RagError):
    def __init__(self) -> None:
        super().__init__("ANTHROPIC_API_KEY is not set. Add it to .env in the repository root.")
