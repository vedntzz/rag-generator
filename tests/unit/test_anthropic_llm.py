"""Tests for AnthropicLLM with a mocked Anthropic client (no network)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from rag_generator.errors import LlmError
from rag_generator.llm.anthropic_llm import AnthropicLLM

MODEL = "claude-haiku-4-5-20251001"


def client_returning(*blocks: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(content=list(blocks))
    return client


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def client_raising(error: Exception) -> MagicMock:
    client = MagicMock()
    client.messages.create.side_effect = error
    return client


def test_llm_sends_system_user_and_deterministic_settings() -> None:
    client = client_returning(text_block("ok"))
    AnthropicLLM(MODEL, client).complete("rules", "question")
    client.messages.create.assert_called_once_with(
        model=MODEL,
        system="rules",
        messages=[{"role": "user", "content": "question"}],
        temperature=0,
        max_tokens=1024,
    )


def test_llm_joins_text_blocks_and_skips_other_blocks() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="hidden")
    client = client_returning(text_block("24 days "), thinking, text_block("[1]."))
    assert AnthropicLLM(MODEL, client).complete("s", "u") == "24 days [1]."


def test_llm_returns_empty_string_when_no_text_blocks() -> None:
    assert AnthropicLLM(MODEL, client_returning()).complete("s", "u") == ""


def test_llm_wraps_connection_error_in_llm_error() -> None:
    error = anthropic.APIConnectionError(request=MagicMock())
    with pytest.raises(LlmError) as raised:
        AnthropicLLM(MODEL, client_raising(error)).complete("s", "u")
    assert raised.value.__cause__ is error


def test_llm_wraps_status_error_in_llm_error() -> None:
    response = MagicMock(status_code=429)
    error = anthropic.RateLimitError("rate limited", response=response, body=None)
    with pytest.raises(LlmError):
        AnthropicLLM(MODEL, client_raising(error)).complete("s", "u")


def test_llm_lets_non_api_errors_propagate() -> None:
    with pytest.raises(TypeError):
        AnthropicLLM(MODEL, client_raising(TypeError("bug"))).complete("s", "u")
