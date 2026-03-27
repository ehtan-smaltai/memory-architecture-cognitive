"""
LLM Client Adapter — Supports Anthropic and OpenAI-compatible providers (DeepSeek, etc).

Exposes a unified interface matching the Anthropic SDK's messages.create() shape,
so existing code needs minimal changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class _TextBlock:
    text: str


@dataclass
class _Response:
    content: list[_TextBlock]


class _OpenAIMessagesAdapter:
    """Wraps an OpenAI-compatible client to mimic anthropic.Anthropic().messages.create()."""

    def __init__(self, client):
        self._client = client

    def create(self, *, model: str, max_tokens: int, system: str,
               messages: list[dict], **kwargs) -> _Response:
        oai_messages = [{"role": "system", "content": system}] + messages
        resp = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return _Response(content=[_TextBlock(text=resp.choices[0].message.content)])


class _OpenAIAdapter:
    """Mimics anthropic.Anthropic() but backed by an OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.messages = _OpenAIMessagesAdapter(self._client)


class _AnthropicAdapter:
    """Thin pass-through to the real Anthropic client."""

    def __init__(self):
        import anthropic
        self._client = anthropic.Anthropic()
        self.messages = self._client.messages


# ── Provider registry ────────────────────────────────────────────────────────

_PROVIDERS = {
    "anthropic": {
        "builder": lambda: _AnthropicAdapter(),
        "default_model": "claude-sonnet-4-20250514",
    },
    "deepseek": {
        "builder": lambda: _OpenAIAdapter(
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        ),
        "default_model": "deepseek-chat",
    },
}


def get_client(provider: str | None = None):
    """Return an LLM client for the given provider.

    Provider is auto-detected from env var LLM_PROVIDER, defaulting to 'anthropic'.
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    spec = _PROVIDERS.get(provider)
    if not spec:
        raise ValueError(f"Unknown LLM provider: {provider}. Options: {list(_PROVIDERS.keys())}")
    return spec["builder"]()


def get_default_model(provider: str | None = None) -> str:
    """Return the default model name for the given provider."""
    provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
    return _PROVIDERS[provider]["default_model"]
