"""Tests for the composition root; fakes are injected so no model is downloaded."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from rag_generator.config import Settings
from rag_generator.errors import MissingApiKeyError
from rag_generator.llm.anthropic_llm import AnthropicLLM
from rag_generator.pipeline import container
from rag_generator.pipeline.container import (
    build_default_embedder,
    build_default_llm,
    build_rag_service,
)
from tests.document_builders import write_docx, write_text_pdf
from tests.fakes import FakeEmbedder, FakeLLM

LEAVE_TEXT = "Employees get 24 days of annual leave."
QUESTION = "How many days of annual leave do employees get?"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {"data_dir": tmp_path / "data", "top_k": 5, "min_score": 0.3}
    return Settings(_env_file=None, **(values | overrides))  # type: ignore[arg-type]


def write_leave_doc(tmp_path: Path) -> Path:
    path = tmp_path / "docs" / "leave.md"
    path.parent.mkdir(parents=True)
    path.write_text(LEAVE_TEXT, encoding="utf-8")
    return path


def test_container_wires_ingest_and_grounded_answer_with_fakes(tmp_path: Path) -> None:
    service = build_rag_service(make_settings(tmp_path), FakeEmbedder(), FakeLLM("24 days [1]."))
    service.ingest("hr", [write_leave_doc(tmp_path)])
    answer = service.ask("hr", QUESTION)
    assert answer.grounded is True and [c.source for c in answer.citations] == ["leave.md"]


def test_container_applies_min_score_from_settings(tmp_path: Path) -> None:
    llm = FakeLLM("24 days [1].")
    service = build_rag_service(make_settings(tmp_path, min_score=0.99), FakeEmbedder(), llm)
    service.ingest("hr", [write_leave_doc(tmp_path)])
    assert service.ask("hr", QUESTION).grounded is False and llm.prompts == []


def test_container_persists_collections_under_settings_data_dir(tmp_path: Path) -> None:
    service = build_rag_service(make_settings(tmp_path), FakeEmbedder(), FakeLLM())
    service.ingest("hr", [write_leave_doc(tmp_path)])
    assert (tmp_path / "data" / "hr" / "collection.npz").is_file()
    assert service.list_collections() == ["hr"]


def test_container_registers_txt_md_pdf_and_docx_loaders(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Text file.")
    (docs / "b.md").write_text("Markdown file.")
    write_text_pdf(docs / "c.pdf", ["PDF file."])
    write_docx(docs / "d.docx", ["Word file."])
    service = build_rag_service(make_settings(tmp_path), FakeEmbedder(), FakeLLM())
    assert service.ingest("hr", [docs]) == 4


class RecordingTextEmbedding:
    """Stands in for fastembed.TextEmbedding so nothing is downloaded."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name


def test_default_embedder_uses_configured_model_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(container, "TextEmbedding", RecordingTextEmbedding)
    embedder = build_default_embedder(make_settings(tmp_path, embed_model="some/model"))
    assert embedder.model.model_name == "some/model"  # type: ignore[attr-defined]


def test_default_llm_uses_configured_model_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    llm = build_default_llm(make_settings(tmp_path, llm_model="m"))
    assert isinstance(llm, AnthropicLLM)
    assert (llm.model, llm.client.api_key) == ("m", "sk-test")


class CountingTextEmbedding:
    """Stands in for fastembed.TextEmbedding and counts how often it is constructed."""

    instances = 0

    def __init__(self, model_name: str) -> None:
        CountingTextEmbedding.instances += 1

    def passage_embed(self, texts: list[str]) -> Iterator[np.ndarray]:
        return (np.array(FakeEmbedder().embed_query(text)) for text in texts)

    def query_embed(self, query: str) -> Iterator[np.ndarray]:
        return iter([np.array(FakeEmbedder().embed_query(query))])


@pytest.fixture
def counting_text_embedding(monkeypatch: pytest.MonkeyPatch) -> type[CountingTextEmbedding]:
    CountingTextEmbedding.instances = 0
    monkeypatch.setattr(container, "TextEmbedding", CountingTextEmbedding)
    return CountingTextEmbedding


def test_container_list_collections_never_constructs_text_embedding(
    tmp_path: Path, counting_text_embedding: type[CountingTextEmbedding]
) -> None:
    build_rag_service(make_settings(tmp_path), llm=FakeLLM()).list_collections()
    assert counting_text_embedding.instances == 0


def test_container_constructs_text_embedding_once_on_first_embed(
    tmp_path: Path, counting_text_embedding: type[CountingTextEmbedding]
) -> None:
    service = build_rag_service(make_settings(tmp_path), llm=FakeLLM("24 days [1]."))
    service.ingest("hr", [write_leave_doc(tmp_path)])
    service.ask("hr", QUESTION)
    assert counting_text_embedding.instances == 1


@pytest.mark.parametrize("key", [None, ""])
def test_default_llm_raises_missing_api_key_without_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if key is not None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    with pytest.raises(MissingApiKeyError):
        build_default_llm(make_settings(tmp_path))


def test_container_without_key_still_lists_and_ingests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = build_rag_service(make_settings(tmp_path), FakeEmbedder())
    service.ingest("hr", [write_leave_doc(tmp_path)])
    assert service.list_collections() == ["hr"]


def test_container_without_key_raises_missing_api_key_when_llm_needed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    service = build_rag_service(make_settings(tmp_path), FakeEmbedder())
    service.ingest("hr", [write_leave_doc(tmp_path)])
    with pytest.raises(MissingApiKeyError):
        service.ask("hr", QUESTION)
