"""Tests proving the test fakes are deterministic."""

from tests.fakes import FakeEmbedder, FakeLLM, RecordedPrompt


def test_fake_embedder_returns_same_vector_for_same_text() -> None:
    first = FakeEmbedder().embed_query("annual leave days")
    second = FakeEmbedder().embed_query("annual leave days")
    assert first == second


def test_fake_embedder_returns_different_vectors_for_different_text() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed_query("annual leave") != embedder.embed_query("pro plan pricing")


def test_fake_embedder_embeds_documents_like_queries() -> None:
    embedder = FakeEmbedder()
    assert embedder.embed_documents(["annual leave"]) == [embedder.embed_query("annual leave")]


def test_fake_embedder_vectors_have_configured_dimension() -> None:
    assert len(FakeEmbedder(dimension=16).embed_query("annual leave")) == 16


def test_fake_llm_returns_canned_answer() -> None:
    llm = FakeLLM(answer="24 days [1].")
    assert llm.complete("system rules", "user question") == "24 days [1]."


def test_fake_llm_records_system_and_user_prompts_in_order() -> None:
    llm = FakeLLM()
    llm.complete("rules", "first")
    llm.complete("rules", "second")
    assert llm.prompts == [RecordedPrompt("rules", "first"), RecordedPrompt("rules", "second")]
