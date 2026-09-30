"""End-to-end proof: two document sets, two collections, answers only from the queried one."""

from pathlib import Path

import pytest

from rag_generator.config import Settings
from rag_generator.pipeline.container import build_rag_service
from rag_generator.pipeline.rag_service import RagService
from tests.fakes import FakeEmbedder, FakeLLM

SAMPLE_DOCS = Path(__file__).resolve().parents[2] / "sample_docs"
HR_FILES = {"leave_policy.md", "remote_work_policy.md", "parental_leave.txt"}
PRODUCT_FILES = {"pricing.md", "features.md", "support.txt"}
HR_QUESTION = "How many paid leave days do employees get per year?"
PRODUCT_QUESTION = "How much does the Pro plan cost per month?"
# 1024 hash buckets keep bag-of-words collisions rare; with these docs and questions,
# own-collection top scores are >= 0.32 and cross-collection scores <= 0.14.
FAKE_DIMENSION, MIN_SCORE = 1024, 0.2


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM("Answer citing everything [1][2][3][4][5].")


@pytest.fixture
def service(tmp_path: Path, llm: FakeLLM) -> RagService:
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", min_score=MIN_SCORE)  # type: ignore[call-arg]
    service = build_rag_service(settings, FakeEmbedder(dimension=FAKE_DIMENSION), llm)
    service.ingest("hr", [SAMPLE_DOCS / "hr"])
    service.ingest("product", [SAMPLE_DOCS / "product"])
    return service


def test_e2e_both_collections_are_listed(service: RagService) -> None:
    assert service.list_collections() == ["hr", "product"]


def test_e2e_hr_question_on_hr_cites_only_hr_files(service: RagService) -> None:
    answer = service.ask("hr", HR_QUESTION)
    sources = {citation.source for citation in answer.citations}
    assert answer.grounded is True
    assert "leave_policy.md" in sources and sources <= HR_FILES


def test_e2e_hr_question_on_product_is_not_grounded_and_skips_llm(
    service: RagService, llm: FakeLLM
) -> None:
    calls_before = len(llm.prompts)
    answer = service.ask("product", HR_QUESTION)
    assert (answer.grounded, answer.citations) == (False, [])
    assert len(llm.prompts) == calls_before


def test_e2e_product_question_on_product_cites_only_product_files(service: RagService) -> None:
    answer = service.ask("product", PRODUCT_QUESTION)
    sources = {citation.source for citation in answer.citations}
    assert answer.grounded is True
    assert "pricing.md" in sources and sources <= PRODUCT_FILES


def test_e2e_product_question_on_hr_is_not_grounded(service: RagService) -> None:
    assert service.ask("hr", PRODUCT_QUESTION).grounded is False


def test_e2e_hr_prompt_contains_no_product_text(service: RagService, llm: FakeLLM) -> None:
    service.ask("hr", HR_QUESTION)
    assert "$49" not in llm.prompts[-1].user and "24 paid leave days" in llm.prompts[-1].user
