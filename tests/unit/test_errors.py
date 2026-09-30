"""Tests for domain error types and their messages."""

from rag_generator.errors import (
    CollectionNotFoundError,
    InvalidCollectionNameError,
    LlmError,
    RagError,
    UnsupportedFileTypeError,
)


def test_unsupported_file_type_error_message_names_extension() -> None:
    error = UnsupportedFileTypeError(".xlsx")
    assert str(error) == "Unsupported file type: '.xlsx'"


def test_unsupported_file_type_error_keeps_extension() -> None:
    assert UnsupportedFileTypeError(".xlsx").extension == ".xlsx"


def test_collection_not_found_error_message_names_collection() -> None:
    error = CollectionNotFoundError("hr")
    assert str(error) == "Collection not found: 'hr'"


def test_collection_not_found_error_keeps_collection() -> None:
    assert CollectionNotFoundError("hr").collection == "hr"


def test_errors_share_rag_error_base_class() -> None:
    assert issubclass(UnsupportedFileTypeError, RagError)
    assert issubclass(CollectionNotFoundError, RagError)
    assert issubclass(InvalidCollectionNameError, RagError)
    assert issubclass(LlmError, RagError)


def test_invalid_collection_name_error_message_names_collection() -> None:
    error = InvalidCollectionNameError("../etc")
    assert str(error) == "Invalid collection name: '../etc'"


def test_invalid_collection_name_error_keeps_collection() -> None:
    assert InvalidCollectionNameError("../etc").collection == "../etc"


def test_llm_error_message_includes_detail() -> None:
    assert str(LlmError("Connection error.")) == "LLM request failed: Connection error."
