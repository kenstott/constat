# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""OpenAI GPT provider with tool support."""

import json
import logging
from typing import Callable, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider with tool calling support.

    Supports GPT-4, GPT-4 Turbo, GPT-4o, and other OpenAI models.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: Model to use (e.g., "gpt-4o", "gpt-4-turbo", "gpt-4")
            base_url: Optional custom base URL for OpenAI-compatible APIs
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install openai"
            )

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model

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

        Converts Anthropic-style tool definitions to OpenAI format.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        tool_handlers = tool_handlers or {}
        use_model = model or self.model

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = self.convert_tools_to_openai_format(tools)

        while True:
            kwargs = {
                "model": use_model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            # Check for truncation due to max_tokens limit
            if choice.finish_reason == "length":
                logger.warning(
                    f"[OPENAI] Response truncated at max_tokens={max_tokens}. "
                    f"Consider increasing limit or simplifying request."
                )

            # Check if we need to handle tool calls
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                # Add assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in message.tool_calls
                    ]
                })

                # Execute each tool call and add results
                for tool_call in message.tool_calls:
                    handler = tool_handlers.get(tool_call.function.name)
                    if handler:
                        try:
                            args = json.loads(tool_call.function.arguments)
                            result = handler(**args)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": str(result),
                            })
                        except Exception as e:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {e}",
                            })
                    else:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Unknown tool: {tool_call.function.name}",
                        })
            else:
                # No more tool calls, return final text
                return message.content or ""
