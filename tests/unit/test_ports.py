"""Contract tests pinning the parameter names adapters must implement."""

import inspect
from collections.abc import Callable

from rag_generator.ports import LLM, VectorStore


def parameter_names_after_self(method: Callable[..., object]) -> list[str]:
    return list(inspect.signature(method).parameters)[1:]


def test_vector_store_search_takes_collection_query_vector_and_top_k() -> None:
    names = parameter_names_after_self(VectorStore.search)
    assert names == ["collection", "query_vector", "top_k"]


def test_llm_complete_takes_system_and_user_prompts() -> None:
    assert parameter_names_after_self(LLM.complete) == ["system", "user"]
