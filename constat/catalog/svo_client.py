# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Concrete LLMClient adapters for SVOExtractor."""

from __future__ import annotations

from chonk import LLMClient  # noqa: F401 — re-exported for callers


class AnthropicSVOClient:
    """LLMClient adapter backed by the Anthropic Messages API.

    Args:
        model: Anthropic model ID. Defaults to claude-haiku-4-5-20251001.
        api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        max_tokens: Max tokens for the completion response.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
