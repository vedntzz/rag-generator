"""Command-line interface: rag ingest | ask | list."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from rag_generator.config import Settings
from rag_generator.domain import Answer, Citation
from rag_generator.errors import RagError
from rag_generator.pipeline.container import build_rag_service
from rag_generator.pipeline.rag_service import RagService

app = typer.Typer(no_args_is_help=True, help="Ingest documents and ask grounded questions.")
CollectionOption = Annotated[str, typer.Option("--collection", "-c", help="Collection name.")]


def get_rag_service() -> RagService:
    return build_rag_service(Settings())


@contextmanager
def exit_on_rag_error() -> Iterator[None]:
    # Expected failures print a one-line message instead of a traceback.
    try:
        yield
    except RagError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command()
def ingest(
    collection: CollectionOption,
    paths: Annotated[list[Path], typer.Argument(exists=True, help="Files or directories.")],
) -> None:
    """Index files or directories into a collection."""
    with exit_on_rag_error():
        count = get_rag_service().ingest(collection, paths)
    typer.echo(f"Indexed {count} chunk(s) into '{collection}'.")


@app.command()
def ask(collection: CollectionOption, question: str) -> None:
    """Answer a question from a collection, with citations."""
    with exit_on_rag_error():
        answer = get_rag_service().ask(collection, question)
    typer.echo(format_answer(answer))


@app.command(name="list")
def list_collections() -> None:
    """List collection names."""
    for name in get_rag_service().list_collections():
        typer.echo(name)


def format_answer(answer: Answer) -> str:
    sources = [format_citation(citation) for citation in answer.citations]
    return "\n".join([answer.text, *(["", "Sources:", *sources] if sources else [])])


def format_citation(citation: Citation) -> str:
    return f"- {citation.source} (chunk {citation.chunk_index}, score {citation.score:.2f})"
