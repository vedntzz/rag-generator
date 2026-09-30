"""Composition root: the only module that constructs concrete adapters."""

import anthropic
from fastembed import TextEmbedding

from rag_generator.chunking.recursive_chunker import RecursiveCharacterChunker
from rag_generator.config import Settings
from rag_generator.embedding.fastembed_embedder import FastEmbedEmbedder
from rag_generator.embedding.lazy_embedder import LazyEmbedder
from rag_generator.llm.anthropic_llm import AnthropicLLM
from rag_generator.loaders.docx_loader import DocxLoader
from rag_generator.loaders.pdf_loader import PdfLoader
from rag_generator.loaders.registry import LoaderRegistry
from rag_generator.loaders.text_loader import TextLoader
from rag_generator.pipeline.answer import AnswerService
from rag_generator.pipeline.ingest import IngestService
from rag_generator.pipeline.rag_service import RagService
from rag_generator.ports import LLM, Embedder
from rag_generator.store.numpy_store import NumpyVectorStore


def build_rag_service(
    settings: Settings, embedder: Embedder | None = None, llm: LLM | None = None
) -> RagService:
    """Wire every adapter; embedder/llm overrides let tests inject fakes (no download, no API)."""
    embedder = build_lazy_default_embedder(settings) if embedder is None else embedder
    llm = build_default_llm(settings) if llm is None else llm
    store = NumpyVectorStore(settings.data_dir)
    chunker = RecursiveCharacterChunker(settings.chunk_size, settings.chunk_overlap)
    ingest = IngestService(build_loader_registry(), chunker, embedder, store)
    answer = AnswerService(embedder, store, llm, settings.top_k, settings.min_score)
    return RagService(ingest, answer, store)


def build_loader_registry() -> LoaderRegistry:
    registry = LoaderRegistry()
    registry.register(".txt", TextLoader())
    registry.register(".md", TextLoader())
    registry.register(".pdf", PdfLoader())
    registry.register(".docx", DocxLoader())
    return registry


def build_lazy_default_embedder(settings: Settings) -> LazyEmbedder:
    # Model download/loading waits for the first embed call, so `rag list` stays cheap.
    return LazyEmbedder(lambda: build_default_embedder(settings))


def build_default_embedder(settings: Settings) -> FastEmbedEmbedder:
    return FastEmbedEmbedder(TextEmbedding(model_name=settings.embed_model))


def build_default_llm(settings: Settings) -> AnthropicLLM:
    key = settings.anthropic_api_key
    client = anthropic.Anthropic(api_key=key.get_secret_value() if key else None)
    return AnthropicLLM(settings.llm_model, client, settings.llm_temperature)
