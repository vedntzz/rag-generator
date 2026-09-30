"""LLM adapter over the Anthropic Messages API."""

import anthropic

from rag_generator.domain import LlmReply
from rag_generator.errors import LlmError

DEFAULT_TEMPERATURE = 0.0
MAX_TOKENS = 1024


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        client: anthropic.Anthropic,
        temperature: float | None = DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model
        self.client = client
        self.temperature = temperature

    def complete(self, system: str, user: str) -> LlmReply:
        try:
            response = self._create_message(system, user)
        except anthropic.APIError as error:
            raise LlmError(str(error)) from error
        return reply_from_message(response)

    def _create_message(self, system: str, user: str) -> anthropic.types.Message:
        # Some models reject any temperature, so None leaves it out of the request entirely.
        sampling = {} if self.temperature is None else {"temperature": self.temperature}
        return self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=MAX_TOKENS,
            **sampling,
        )


def reply_from_message(response: anthropic.types.Message) -> LlmReply:
    text = "".join(block.text for block in response.content if block.type == "text")
    return LlmReply(text=text, truncated=response.stop_reason == "max_tokens")
