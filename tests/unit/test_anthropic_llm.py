"""Tests for AnthropicLLM with a mocked Anthropic client (no network)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import pytest

from rag_generator.domain import LlmReply
from rag_generator.errors import LlmError
from rag_generator.llm.anthropic_llm import AnthropicLLM

MODEL = "claude-haiku-4-5-20251001"


def client_returning(*blocks: SimpleNamespace, stop_reason: str = "end_turn") -> MagicMock:
    client = MagicMock()
    response = SimpleNamespace(content=list(blocks), stop_reason=stop_reason)
    client.messages.create.return_value = response
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


def test_llm_sends_configured_temperature() -> None:
    client = client_returning(text_block("ok"))
    AnthropicLLM(MODEL, client, temperature=0.7).complete("s", "u")
    assert client.messages.create.call_args.kwargs["temperature"] == 0.7


def test_llm_omits_temperature_when_none() -> None:
    client = client_returning(text_block("ok"))
    AnthropicLLM(MODEL, client, temperature=None).complete("s", "u")
    assert "temperature" not in client.messages.create.call_args.kwargs


def test_llm_joins_text_blocks_and_skips_other_blocks() -> None:
    thinking = SimpleNamespace(type="thinking", thinking="hidden")
    client = client_returning(text_block("24 days "), thinking, text_block("[1]."))
    assert AnthropicLLM(MODEL, client).complete("s", "u").text == "24 days [1]."


def test_llm_returns_empty_text_when_no_text_blocks() -> None:
    assert AnthropicLLM(MODEL, client_returning()).complete("s", "u").text == ""


def test_llm_reply_is_not_truncated_on_end_turn() -> None:
    reply = AnthropicLLM(MODEL, client_returning(text_block("ok"))).complete("s", "u")
    assert reply == LlmReply(text="ok", truncated=False)


def test_llm_reply_is_truncated_on_max_tokens_stop() -> None:
    client = client_returning(text_block("partial"), stop_reason="max_tokens")
    assert AnthropicLLM(MODEL, client).complete("s", "u") == LlmReply("partial", truncated=True)


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
