"""Tests for LazyEmbedder, which defers building the real embedder until first use."""

from rag_generator.embedding.lazy_embedder import LazyEmbedder
from tests.fakes import FakeEmbedder


class CountingBuilder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> FakeEmbedder:
        self.calls += 1
        return FakeEmbedder()


def test_lazy_embedder_does_not_build_until_first_embed() -> None:
    builder = CountingBuilder()
    LazyEmbedder(builder)
    assert builder.calls == 0


def test_lazy_embedder_builds_once_across_calls() -> None:
    builder = CountingBuilder()
    embedder = LazyEmbedder(builder)
    embedder.embed_query("a")
    embedder.embed_documents(["b", "c"])
    assert builder.calls == 1


def test_lazy_embedder_delegates_embed_query() -> None:
    assert LazyEmbedder(FakeEmbedder).embed_query("leave") == FakeEmbedder().embed_query("leave")


def test_lazy_embedder_delegates_embed_documents() -> None:
    expected = FakeEmbedder().embed_documents(["a", "b"])
    assert LazyEmbedder(FakeEmbedder).embed_documents(["a", "b"]) == expected
