"""Domain errors raised by the RAG pipeline."""


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
