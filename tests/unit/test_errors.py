"""Tests for domain error types and their messages."""

from rag_generator.errors import CollectionNotFoundError, RagError, UnsupportedFileTypeError


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
