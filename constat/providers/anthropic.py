# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Anthropic Claude provider with tool support."""

import logging
from typing import Callable, Optional

import anthropic

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider with tool calling support."""

    # Model output token limits
    MODEL_OUTPUT_LIMITS = {
        "claude-3-opus": 4096,
        "claude-3-sonnet": 4096,
        "claude-3-haiku": 4096,
        "claude-3-5-sonnet": 8192,
        "claude-3-5-haiku": 8192,
        "claude-sonnet-4": 16384,
        "claude-opus-4": 16384,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @property
    def max_output_tokens(self) -> int:
        """Get max output tokens for the current model."""
        for prefix, limit in self.MODEL_OUTPUT_LIMITS.items():
            if self.model.startswith(prefix):
                return limit
        return 16384  # Default for newer models

    def generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate a response, automatically handling tool calls.

        Args:
            system: System prompt
            user_message: User's question/request
            tools: Tool definitions in Anthropic format
            tool_handlers: Dict mapping tool names to handler functions
            max_tokens: Maximum tokens to generate
            model: Override model for this call (for tiered model selection)

        Returns:
            Final text response after all tool calls are resolved
        """
        messages = [{"role": "user", "content": user_message}]
        tool_handlers = tool_handlers or {}
        use_model = model or self.model

        while True:
            # Make API call
            kwargs = {
                "model": use_model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

            # Check for truncation due to max_tokens limit
            if response.stop_reason == "max_tokens":
                logger.warning(
                    f"[ANTHROPIC] Response truncated at max_tokens={max_tokens}. "
                    f"Consider increasing limit or simplifying request."
                )

            # Check if we need to handle tool calls
            if response.stop_reason == "tool_use":
                # Extract tool uses and text from response
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({
                            "type": "text",
                            "text": block.text
                        })
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })

                        # Call the handler
                        handler = tool_handlers.get(block.name)
                        if handler:
                            try:
                                result = handler(**block.input)
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(result)
                                })
                            except Exception as e:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": f"Error: {e}",
                                    "is_error": True
                                })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Unknown tool: {block.name}",
                                "is_error": True
                            })

                # Add assistant message with tool uses (only if non-empty)
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                # Add tool results (only if non-empty to avoid API error)
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No tool results - edge case (stop_reason=tool_use but no tool blocks)
                    # Extract any text from the response and return it
                    text_parts = []
                    for block in response.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                    return "\n".join(text_parts) if text_parts else ""

            else:
                # No more tool calls, extract final text
                text_parts = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                return "\n".join(text_parts)
