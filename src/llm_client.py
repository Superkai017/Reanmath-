"""Thin wrapper around the Anthropic API call."""
from anthropic import Anthropic

from reanmath.config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from reanmath.schemas import ChatMessage

_client = Anthropic(api_key=ANTHROPIC_API_KEY)


def call_claude(
    system: str,
    message: str,
    history: list[ChatMessage] | None = None,
    max_tokens: int = 1024,
) -> str:
    messages = []
    for turn in history or []:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": message})

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )

    return "".join(block.text for block in response.content if block.type == "text")
