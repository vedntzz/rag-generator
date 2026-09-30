"""Embedder that builds its real embedder on first use (keeps model loading off cold paths)."""

from collections.abc import Callable, Sequence

from rag_generator.ports import Embedder


class LazyEmbedder:
    def __init__(self, build_embedder: Callable[[], Embedder]) -> None:
        self._build_embedder = build_embedder
        self._embedder: Embedder | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._get_embedder().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedder().embed_query(text)

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self._build_embedder()
        return self._embedder
