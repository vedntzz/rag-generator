"""Shared pytest fixtures."""

import pytest

from tests.fakes import FakeEmbedder, FakeLLM


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
