"""Tests for the RagService facade."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_generator.domain import Answer
from rag_generator.errors import InvalidCollectionNameError
from rag_generator.pipeline.rag_service import RagService


def make_service() -> tuple[RagService, MagicMock, MagicMock, MagicMock]:
    ingest, answer, store = MagicMock(), MagicMock(), MagicMock()
    return RagService(ingest, answer, store), ingest, answer, store


def test_rag_service_ingest_delegates_and_returns_chunk_count() -> None:
    service, ingest, _, _ = make_service()
    ingest.ingest.return_value = 7
    assert service.ingest("hr", [Path("docs")]) == 7
    ingest.ingest.assert_called_once_with("hr", [Path("docs")])


def test_rag_service_ask_delegates_to_answer_service() -> None:
    service, _, answer, _ = make_service()
    answer.ask.return_value = Answer("24 days [1].", [], grounded=True)
    assert service.ask("hr", "How much leave?") == Answer("24 days [1].", [], grounded=True)
    answer.ask.assert_called_once_with("hr", "How much leave?")


def test_rag_service_list_collections_delegates_to_store() -> None:
    service, _, _, store = make_service()
    store.list_collections.return_value = ["hr", "product"]
    assert service.list_collections() == ["hr", "product"]


def test_rag_service_ingest_rejects_invalid_name_before_loading_anything() -> None:
    service, ingest, _, _ = make_service()
    with pytest.raises(InvalidCollectionNameError):
        service.ingest("../etc", [Path("docs")])
    ingest.ingest.assert_not_called()


def test_rag_service_ask_rejects_invalid_name_before_embedding_anything() -> None:
    service, _, answer, _ = make_service()
    with pytest.raises(InvalidCollectionNameError):
        service.ask("has space", "How much leave?")
    answer.ask.assert_not_called()
