"""Tests for the Typer CLI, with a fake-backed RagService injected."""

from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from rag_generator.config import Settings
from rag_generator.interfaces import cli
from rag_generator.llm.prompts import NOT_FOUND_MESSAGE
from rag_generator.pipeline import container
from rag_generator.pipeline.container import build_rag_service
from rag_generator.pipeline.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLLM

QUESTION = "How many days of annual leave do employees get?"


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RagService:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    service = build_rag_service(settings, FakeEmbedder(), FakeLLM("Employees get 24 days [1]."))
    monkeypatch.setattr(cli, "get_rag_service", lambda: service)
    return service


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    directory = tmp_path / "docs"
    directory.mkdir()
    (directory / "leave.md").write_text("Employees get 24 days of annual leave.")
    return directory


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, list(args))


def test_cli_ingest_reports_chunks_indexed(service: RagService, docs: Path) -> None:
    result = run("ingest", "--collection", "hr", str(docs))
    assert result.exit_code == 0
    assert "Indexed 1 chunk(s) into 'hr'" in result.stdout


def test_cli_ingest_accepts_multiple_paths(service: RagService, docs: Path, tmp_path: Path) -> None:
    extra = tmp_path / "sick.md"
    extra.write_text("Sick leave is 10 days.")
    result = run("ingest", "--collection", "hr", str(docs), str(extra))
    assert "Indexed 2 chunk(s) into 'hr'" in result.stdout


def test_cli_ask_prints_answer_and_citations(service: RagService, docs: Path) -> None:
    service.ingest("hr", [docs])
    result = run("ask", "--collection", "hr", QUESTION)
    assert result.exit_code == 0
    assert "Employees get 24 days [1]." in result.stdout
    assert "[1] leave.md (chunk 0, score " in result.stdout


def test_cli_ask_labels_citations_with_reply_reference_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, docs: Path
) -> None:
    (docs / "leave2.md").write_text("Employees get 24 days of annual leave, rising to 30.")
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")  # type: ignore[call-arg]
    service = build_rag_service(settings, FakeEmbedder(), FakeLLM("Rises to 30 [2]."))
    monkeypatch.setattr(cli, "get_rag_service", lambda: service)
    service.ingest("hr", [docs])
    lines = run("ask", "--collection", "hr", QUESTION).stdout.splitlines()
    assert [line.split(" ")[0] for line in lines if line.startswith("[")] == ["[2]"]


def test_cli_ask_exits_zero_when_answer_not_grounded(service: RagService, docs: Path) -> None:
    service.ingest("hr", [docs])
    result = run("ask", "--collection", "hr", "zzz qqq")
    assert result.exit_code == 0
    assert NOT_FOUND_MESSAGE in result.stdout


def test_cli_ask_reports_missing_collection_on_stderr(service: RagService) -> None:
    result = run("ask", "--collection", "hr", QUESTION)
    assert result.exit_code == 1
    assert "Collection not found: 'hr'" in result.stderr


def test_cli_ingest_reports_unsupported_file_on_stderr(service: RagService, tmp_path: Path) -> None:
    bad = tmp_path / "budget.xlsx"
    bad.write_text("x")
    result = run("ingest", "--collection", "hr", str(bad))
    assert result.exit_code == 1
    assert "Unsupported file type: '.xlsx'" in result.stderr


def test_cli_ingest_reports_invalid_collection_name_on_stderr(
    service: RagService, docs: Path
) -> None:
    result = run("ingest", "--collection", "bad name", str(docs))
    assert result.exit_code == 1
    assert "Invalid collection name: 'bad name'" in result.stderr


def test_cli_list_prints_collections_one_per_line(service: RagService, docs: Path) -> None:
    service.ingest("product", [docs])
    service.ingest("hr", [docs])
    result = run("list")
    assert (result.exit_code, result.stdout.splitlines()) == (0, ["hr", "product"])


class ExplodingTextEmbedding:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("TextEmbedding must not be constructed by `rag list`")


def test_cli_list_with_real_container_never_constructs_text_embedding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container, "TextEmbedding", ExplodingTextEmbedding)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    result = run("list")
    assert result.exit_code == 0, result.output
