"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import re


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


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM providers must implement:
    - generate(): Generate text response, handling tool calls automatically
    - generate_code(): Generate and extract code from response

    Providers should handle their own tool calling conventions and convert
    to/from the standard tool format used by Constat.
    """

    @property
    def supports_tools(self) -> bool:
        """Whether this provider/model supports tool calling.

        Override in subclasses if tool support is conditional (e.g., model-dependent).
        Returns True by default for providers that implement tool handling.
        """
        return True

    @abstractmethod
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
            tools: Tool definitions (provider converts to its format)
            tool_handlers: Dict mapping tool names to handler functions
            max_tokens: Maximum tokens to generate
            model: Override model for this call

        Returns:
            Final text response after all tool calls are resolved
        """
        pass

    def generate_code(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
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
            model=model,
        )
        return self._extract_code(response)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks."""
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

    @staticmethod
    def convert_tools_to_openai_format(tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tools to OpenAI function format.

        Anthropic format:
            {"name": "x", "description": "y", "input_schema": {...}}

        OpenAI format:
            {"type": "function", "function": {"name": "x", "description": "y", "parameters": {...}}}
        """
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                }
            })
        return openai_tools
