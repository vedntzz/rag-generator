"""LLM adapter over the Anthropic Messages API."""

import anthropic

from rag_generator.domain import LlmReply
from rag_generator.errors import LlmError

MAX_TOKENS = 1024


class AnthropicLLM:
    def __init__(self, model: str, client: anthropic.Anthropic) -> None:
        self.model = model
        self.client = client

    def complete(self, system: str, user: str) -> LlmReply:
        try:
            response = self._create_message(system, user)
        except anthropic.APIError as error:
            raise LlmError(str(error)) from error
        return reply_from_message(response)

    def _create_message(self, system: str, user: str) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
        )


def reply_from_message(response: anthropic.types.Message) -> LlmReply:
    text = "".join(block.text for block in response.content if block.type == "text")
    return LlmReply(text=text, truncated=response.stop_reason == "max_tokens")
