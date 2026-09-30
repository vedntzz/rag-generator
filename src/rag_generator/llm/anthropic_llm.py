"""LLM adapter over the Anthropic Messages API."""

import anthropic

from rag_generator.errors import LlmError

TEMPERATURE = 0
MAX_TOKENS = 1024


class AnthropicLLM:
    def __init__(self, model: str, client: anthropic.Anthropic) -> None:
        self.model = model
        self.client = client

    def complete(self, system: str, user: str) -> str:
        try:
            response = self._create_message(system, user)
        except anthropic.APIError as error:
            raise LlmError(str(error)) from error
        return join_text_blocks(response)

    def _create_message(self, system: str, user: str) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )


def join_text_blocks(response: anthropic.types.Message) -> str:
    return "".join(block.text for block in response.content if block.type == "text")
