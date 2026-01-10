"""Anthropic Claude provider with tool support."""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import anthropic


@dataclass
class ToolResult:
    """Result from a tool call."""
    tool_use_id: str
    result: Any


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = ""


class AnthropicProvider:
    """Anthropic Claude provider with tool calling support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a response, automatically handling tool calls.

        Args:
            system: System prompt
            user_message: User's question/request
            tools: Tool definitions in Anthropic format
            tool_handlers: Dict mapping tool names to handler functions
            max_tokens: Maximum tokens to generate

        Returns:
            Final text response after all tool calls are resolved
        """
        messages = [{"role": "user", "content": user_message}]
        tool_handlers = tool_handlers or {}

        while True:
            # Make API call
            kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            }
            if tools:
                kwargs["tools"] = tools

            response = self.client.messages.create(**kwargs)

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

                # Add assistant message with tool uses
                messages.append({"role": "assistant", "content": assistant_content})
                # Add tool results
                messages.append({"role": "user", "content": tool_results})

            else:
                # No more tool calls, extract final text
                text_parts = []
                for block in response.content:
                    if block.type == "text":
                        text_parts.append(block.text)
                return "\n".join(text_parts)

    def generate_code(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate code, extracting from markdown code blocks if present.

        Returns just the code string, stripped of markdown fencing.
        """
        response = self.generate(
            system=system,
            user_message=user_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_tokens=max_tokens,
        )

        return self._extract_code(response)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
        import re

        # Try to find ```python ... ``` block
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic ``` ... ``` block
        pattern = r"```\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # No code block found, return as-is
        return text.strip()
