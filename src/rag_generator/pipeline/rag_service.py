"""RagService: the facade the CLI and API talk to."""

from collections.abc import Sequence
from pathlib import Path

from rag_generator.domain import Answer, validate_collection_name
from rag_generator.pipeline.answer import AnswerService
from rag_generator.pipeline.ingest import IngestService
from rag_generator.ports import VectorStore


class RagService:
    def __init__(
        self, ingest_service: IngestService, answer_service: AnswerService, store: VectorStore
    ) -> None:
        self.ingest_service = ingest_service
        self.answer_service = answer_service
        self.store = store

    def ingest(self, collection: str, paths: Sequence[Path]) -> int:
        validate_collection_name(collection)
        return self.ingest_service.ingest(collection, paths)

    def ask(self, collection: str, question: str) -> Answer:
        validate_collection_name(collection)
        return self.answer_service.ask(collection, question)

    def list_collections(self) -> list[str]:
        return self.store.list_collections()
