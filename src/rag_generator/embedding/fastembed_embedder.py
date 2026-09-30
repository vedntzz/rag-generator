"""Embedder adapter over fastembed's local ONNX TextEmbedding (no API key)."""

from collections.abc import Iterable, Sequence
from typing import Any, Protocol


class TextEmbeddingModel(Protocol):
    """The slice of fastembed.TextEmbedding this adapter relies on."""

    def passage_embed(self, texts: Iterable[str]) -> Iterable[Any]: ...

    def query_embed(self, query: str) -> Iterable[Any]: ...


class FastEmbedEmbedder:
    def __init__(self, model: TextEmbeddingModel) -> None:
        self.model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self.model.passage_embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        # bge models embed queries differently from passages, hence query_embed here.
        return next(iter(self.model.query_embed(text))).tolist()
