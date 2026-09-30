"""Tests for the FastAPI app, with a fake-backed RagService injected via dependency override."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_generator.config import Settings
from rag_generator.domain import LlmReply
from rag_generator.errors import LlmError, MissingApiKeyError, NoDocumentsFoundError
from rag_generator.interfaces.api import app, get_rag_service
from rag_generator.llm.prompts import NOT_FOUND_MESSAGE
from rag_generator.pipeline.container import build_rag_service
from rag_generator.pipeline.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLLM

LEAVE_FILE = ("files", ("leave.md", b"Employees get 24 days of annual leave.", "text/markdown"))
QUESTION = {"question": "How many days of annual leave do employees get?"}


class FailingLLM:
    def complete(self, system: str, user: str) -> LlmReply:
        raise LlmError("Connection error.")


def client_for(service: RagService) -> Iterator[TestClient]:
    app.dependency_overrides[get_rag_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def service_with(tmp_path: Path, llm: object) -> RagService:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    return build_rag_service(settings, FakeEmbedder(), llm)  # type: ignore[arg-type]


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    yield from client_for(service_with(tmp_path, FakeLLM("Employees get 24 days [1].")))


@pytest.fixture
def failing_llm_client(tmp_path: Path) -> Iterator[TestClient]:
    yield from client_for(service_with(tmp_path, FailingLLM()))


def test_api_upload_returns_chunks_indexed(client: TestClient) -> None:
    response = client.post("/collections/hr/documents", files=[LEAVE_FILE])
    assert (response.status_code, response.json()) == (200, {"chunks_indexed": 1})


def test_api_ask_returns_answer_with_citations_from_uploaded_file(client: TestClient) -> None:
    client.post("/collections/hr/documents", files=[LEAVE_FILE])
    body = client.post("/collections/hr/ask", json=QUESTION).json()
    assert (body["answer"], body["grounded"], body["truncated"]) == (
        "Employees get 24 days [1].", True, False
    )
    citations = [(c["reference"], c["source"], c["chunk_index"]) for c in body["citations"]]
    assert citations == [(1, "leave.md", 0)]
    assert isinstance(body["citations"][0]["score"], float)


def test_api_ask_returns_ungrounded_answer_with_200(client: TestClient) -> None:
    client.post("/collections/hr/documents", files=[LEAVE_FILE])
    response = client.post("/collections/hr/ask", json={"question": "zzz qqq"})
    assert response.status_code == 200
    assert (response.json()["answer"], response.json()["grounded"]) == (NOT_FOUND_MESSAGE, False)


def test_api_list_collections_returns_sorted_names(client: TestClient) -> None:
    client.post("/collections/product/documents", files=[LEAVE_FILE])
    client.post("/collections/hr/documents", files=[LEAVE_FILE])
    assert client.get("/collections").json() == ["hr", "product"]


def test_api_ask_returns_404_for_missing_collection(client: TestClient) -> None:
    response = client.post("/collections/hr/ask", json=QUESTION)
    assert response.status_code == 404
    assert response.json()["detail"] == "Collection not found: 'hr'"


def test_api_upload_returns_400_for_unsupported_file_type(client: TestClient) -> None:
    bad = ("files", ("budget.xlsx", b"x", "application/octet-stream"))
    response = client.post("/collections/hr/documents", files=[bad])
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type: '.xlsx'"


@pytest.mark.parametrize("name", ["bad.name", "has%20space", "x" * 65])
def test_api_upload_returns_422_for_invalid_collection_name(client: TestClient, name: str) -> None:
    response = client.post(f"/collections/{name}/documents", files=[LEAVE_FILE])
    assert response.status_code == 422


def test_api_ask_returns_422_for_invalid_collection_name(client: TestClient) -> None:
    assert client.post("/collections/bad.name/ask", json=QUESTION).status_code == 422


def test_api_ask_returns_502_when_llm_fails(failing_llm_client: TestClient) -> None:
    failing_llm_client.post("/collections/hr/documents", files=[LEAVE_FILE])
    response = failing_llm_client.post("/collections/hr/ask", json=QUESTION)
    assert response.status_code == 502
    assert response.json()["detail"] == "LLM request failed: Connection error."


def test_api_upload_stores_file_name_only_as_source(client: TestClient) -> None:
    sneaky = ("files", ("../../etc/leave.md", LEAVE_FILE[1][1], "text/markdown"))
    client.post("/collections/hr/documents", files=[sneaky])
    citations = client.post("/collections/hr/ask", json=QUESTION).json()["citations"]
    assert [c["source"] for c in citations] == ["leave.md"]


def record_ingested_paths(service: RagService) -> list[Path]:
    recorded: list[Path] = []
    original_ingest = service.ingest

    def recording_ingest(collection: str, paths: list[Path]) -> int:
        recorded.extend(paths)
        assert all(path.is_file() for path in paths)
        return original_ingest(collection, paths)

    service.ingest = recording_ingest  # type: ignore[method-assign]
    return recorded


def test_api_upload_removes_temporary_directory_after_ingest(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeLLM())
    recorded = record_ingested_paths(service)
    for test_client in client_for(service):
        test_client.post("/collections/hr/documents", files=[LEAVE_FILE])
    assert recorded and not any(path.parent.exists() for path in recorded)


def test_api_upload_removes_temporary_directory_when_ingest_fails(tmp_path: Path) -> None:
    service = service_with(tmp_path, FakeLLM())
    recorded = record_ingested_paths(service)
    bad = ("files", ("budget.xlsx", b"x", "application/octet-stream"))
    for test_client in client_for(service):
        assert test_client.post("/collections/hr/documents", files=[bad]).status_code == 400
    assert recorded and not any(path.parent.exists() for path in recorded)


def test_api_upload_returns_400_when_no_documents_found() -> None:
    service = MagicMock()
    service.ingest.side_effect = NoDocumentsFoundError([Path("upload")])
    for test_client in client_for(service):
        response = test_client.post("/collections/hr/documents", files=[LEAVE_FILE])
        assert response.status_code == 400
        assert response.json()["detail"] == "No supported documents found in: upload"


def test_api_ask_returns_503_when_api_key_missing() -> None:
    service = MagicMock()
    service.ask.side_effect = MissingApiKeyError()
    for test_client in client_for(service):
        response = test_client.post("/collections/hr/ask", json=QUESTION)
        assert response.status_code == 503
        assert response.json()["detail"] == str(MissingApiKeyError())
