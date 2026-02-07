# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Base LLM provider interface."""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Shared thread pool for running sync operations in async context
_DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=10)


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

    @property
    def max_output_tokens(self) -> int:
        """Maximum output tokens supported by the model.

        Override in subclasses based on model capabilities.
        Default is 16384 which covers most modern models.
        """
        return 16384

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
        max_tokens: int = 16384,
        model: Optional[str] = None,
    ) -> str:
        """
        Generate code, extracting from markdown code blocks if present.

        Returns just the code string, stripped of markdown fencing.
        Uses high token limit (16k) since there's no cost penalty for unused headroom.
        """
        response = self.generate(
            system=system,
            user_message=user_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_tokens=max_tokens,
            model=model,
        )
        code, was_truncated = self._extract_code_with_truncation_check(response)
        if was_truncated:
            logger.warning(f"[TRUNCATION] Code response appears truncated (max_tokens={max_tokens})")
        return code

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks.

        Handles various cases:
        - Complete markdown blocks: ```python ... ```
        - Incomplete blocks (no closing fence from truncated responses)
        - Generic ``` ... ``` blocks
        """
        code, _ = self._extract_code_with_truncation_check(text)
        return code

    def _extract_code_with_truncation_check(self, text: str) -> tuple[str, bool]:
        """Extract Python code and detect if response was truncated.

        Returns:
            Tuple of (code, was_truncated)
        """
        text = text.strip()
        was_truncated = False

        # Try to find ```python ... ``` block
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip(), False

        # Try generic ``` ... ``` block
        pattern = r"```\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip(), False

        # Handle incomplete code blocks (truncated response without closing fence)
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                # Skip the first line (```python or ```)
                code = lines[1]
                # Remove trailing ``` if present at the end
                if code.rstrip().endswith("```"):
                    code = code.rstrip()[:-3]
                else:
                    # No closing fence - likely truncated
                    was_truncated = True
                code = code.strip()

                # Additional truncation indicators
                if self._looks_truncated(code):
                    was_truncated = True

                return code, was_truncated

        # No code block found, check for truncation indicators
        if self._looks_truncated(text):
            was_truncated = True

        return text, was_truncated

    def _looks_truncated(self, code: str) -> bool:
        """Detect if code appears truncated based on common patterns."""
        if not code:
            return False

        # Check for unterminated strings
        # Count quotes - odd number suggests truncation
        single_quotes = code.count("'") - code.count("\\'")
        double_quotes = code.count('"') - code.count('\\"')
        triple_single = code.count("'''")
        triple_double = code.count('"""')

        # Adjust for triple quotes (each triple quote is 3 single/double)
        single_quotes -= triple_single * 3
        double_quotes -= triple_double * 3

        if single_quotes % 2 != 0 or double_quotes % 2 != 0:
            return True
        if triple_single % 2 != 0 or triple_double % 2 != 0:
            return True

        # Check for unclosed brackets/parens
        if code.count('(') != code.count(')'):
            return True
        if code.count('[') != code.count(']'):
            return True
        if code.count('{') != code.count('}'):
            return True

        # Check if ends mid-statement (common truncation patterns)
        stripped = code.rstrip()
        truncation_endings = [
            ',', ':', '=', '+', '-', '*', '/', '(', '[', '{',
            'and', 'or', 'not', 'in', 'is', 'if', 'else', 'elif',
            'for', 'while', 'with', 'try', 'except', 'finally',
            'def', 'class', 'return', 'yield', 'import', 'from',
        ]
        last_line = stripped.split('\n')[-1].strip()
        for ending in truncation_endings:
            if last_line.endswith(ending):
                return True

        return False

    async def async_generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> str:
        """
        Async version of generate() for use in parallel resolution.

        By default, runs the sync generate() in a thread pool executor.
        Providers can override this with native async implementations.

        Args:
            system: System prompt
            user_message: User's question/request
            tools: Tool definitions (provider converts to its format)
            tool_handlers: Dict mapping tool names to handler functions
            max_tokens: Maximum tokens to generate
            model: Override model for this call
            executor: Optional custom thread pool executor

        Returns:
            Final text response after all tool calls are resolved
        """
        loop = asyncio.get_event_loop()
        exec_pool = executor or _DEFAULT_EXECUTOR
        return await loop.run_in_executor(
            exec_pool,
            lambda: self.generate(
                system=system,
                user_message=user_message,
                tools=tools,
                tool_handlers=tool_handlers,
                max_tokens=max_tokens,
                model=model,
            )
        )

    async def async_generate_code(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ) -> str:
        """
        Async version of generate_code().

        Returns just the code string, stripped of markdown fencing.
        """
        response = await self.async_generate(
            system=system,
            user_message=user_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_tokens=max_tokens,
            model=model,
            executor=executor,
        )
        return self._extract_code(response)

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
