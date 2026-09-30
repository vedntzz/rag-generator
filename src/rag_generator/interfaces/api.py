"""HTTP API: upload documents, ask questions, list collections."""

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_generator.config import Settings
from rag_generator.domain import Answer
from rag_generator.errors import (
    CollectionNotFoundError,
    InvalidCollectionNameError,
    LlmError,
    RagError,
    UnsupportedFileTypeError,
)
from rag_generator.pipeline.container import build_rag_service
from rag_generator.pipeline.rag_service import RagService

app = FastAPI(title="RAG Generator")
ERROR_STATUS_CODES: dict[type[RagError], int] = {
    CollectionNotFoundError: 404,
    UnsupportedFileTypeError: 400,
    InvalidCollectionNameError: 422,
    LlmError: 502,
}


class AskRequest(BaseModel):
    question: str


class CitationResponse(BaseModel):
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    truncated: bool
    citations: list[CitationResponse]


class IngestResponse(BaseModel):
    chunks_indexed: int


@lru_cache
def get_rag_service() -> RagService:
    return build_rag_service(Settings())


RagServiceDependency = Annotated[RagService, Depends(get_rag_service)]


@app.exception_handler(RagError)
def rag_error_to_response(request: Request, error: RagError) -> JSONResponse:
    status_code = ERROR_STATUS_CODES.get(type(error), 500)
    return JSONResponse(status_code=status_code, content={"detail": str(error)})


@app.post("/collections/{name}/documents")
def upload_documents(
    name: str, files: Annotated[list[UploadFile], File()], service: RagServiceDependency
) -> IngestResponse:
    # Uploads live only for the duration of the ingest; the directory is always removed.
    with TemporaryDirectory() as directory:
        paths = [save_upload(upload, Path(directory)) for upload in files]
        return IngestResponse(chunks_indexed=service.ingest(name, paths))


@app.post("/collections/{name}/ask")
def ask(name: str, request: AskRequest, service: RagServiceDependency) -> AskResponse:
    return to_ask_response(service.ask(name, request.question))


@app.get("/collections")
def list_collections(service: RagServiceDependency) -> list[str]:
    return service.list_collections()


def save_upload(upload: UploadFile, directory: Path) -> Path:
    # Keep only the base name, so "../../x.md" cannot escape the upload directory.
    path = directory / (Path(upload.filename or "").name or "upload")
    path.write_bytes(upload.file.read())
    return path


def to_ask_response(answer: Answer) -> AskResponse:
    citations = [CitationResponse(**asdict(citation)) for citation in answer.citations]
    return AskResponse(
        answer=answer.text,
        grounded=answer.grounded,
        truncated=answer.truncated,
        citations=citations,
    )
