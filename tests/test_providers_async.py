# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for async_generate and async_generate_code methods in BaseLLMProvider.

These methods wrap synchronous generate() calls using ThreadPoolExecutor
for use in async contexts (e.g., AsyncFactResolver).
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Optional, Callable

import pytest

from constat.providers.base import BaseLLMProvider, _DEFAULT_EXECUTOR


class ConcreteTestProvider(BaseLLMProvider):
    """Concrete implementation for testing base class async methods."""

    def __init__(self):
        self.generate_mock = MagicMock(return_value="Test response")

    def generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
    ) -> str:
        return self.generate_mock(
            system=system,
            user_message=user_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_tokens=max_tokens,
            model=model,
        )


class TestAsyncGenerateExceptionPropagation:
    """P0: Test that exceptions from sync generate() propagate correctly."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.mark.asyncio
    async def test_async_generate_propagates_value_error(self, provider):
        """ValueError from generate() is raised in async context."""
        provider.generate_mock.side_effect = ValueError("Invalid input")

        with pytest.raises(ValueError, match="Invalid input"):
            await provider.async_generate(
                system="test",
                user_message="test",
            )

    @pytest.mark.asyncio
    async def test_async_generate_propagates_runtime_error(self, provider):
        """RuntimeError from generate() propagates correctly."""
        provider.generate_mock.side_effect = RuntimeError("Connection failed")

        with pytest.raises(RuntimeError, match="Connection failed"):
            await provider.async_generate(
                system="test",
                user_message="test",
            )

    @pytest.mark.asyncio
    async def test_async_generate_propagates_type_error(self, provider):
        """TypeError from generate() propagates correctly."""
        provider.generate_mock.side_effect = TypeError("Wrong type")

        with pytest.raises(TypeError, match="Wrong type"):
            await provider.async_generate(
                system="test",
                user_message="test",
            )

    @pytest.mark.asyncio
    async def test_async_generate_propagates_custom_exception(self, provider):
        """Custom exceptions from generate() propagate correctly."""

        class CustomAPIError(Exception):
            pass

        provider.generate_mock.side_effect = CustomAPIError("API rate limited")

        with pytest.raises(CustomAPIError, match="API rate limited"):
            await provider.async_generate(
                system="test",
                user_message="test",
            )


class TestAsyncGenerateBasicFunctionality:
    """P0: Test basic async_generate functionality."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.mark.asyncio
    async def test_async_generate_returns_response(self, provider):
        """async_generate returns the response from sync generate."""
        provider.generate_mock.return_value = "Generated text"

        result = await provider.async_generate(
            system="System prompt",
            user_message="User message",
        )

        assert result == "Generated text"

    @pytest.mark.asyncio
    async def test_async_generate_forwards_all_parameters(self, provider):
        """All parameters are forwarded to sync generate."""
        tools = [{"name": "test_tool", "description": "A test tool"}]
        handlers = {"test_tool": lambda x: f"result: {x}"}

        await provider.async_generate(
            system="System prompt",
            user_message="User message",
            tools=tools,
            tool_handlers=handlers,
            max_tokens=1000,
            model="custom-model",
        )

        provider.generate_mock.assert_called_once_with(
            system="System prompt",
            user_message="User message",
            tools=tools,
            tool_handlers=handlers,
            max_tokens=1000,
            model="custom-model",
        )

    @pytest.mark.asyncio
    async def test_async_generate_default_parameters(self, provider):
        """Default parameters are used when not specified."""
        await provider.async_generate(
            system="test",
            user_message="test",
        )

        call_kwargs = provider.generate_mock.call_args.kwargs
        assert call_kwargs["tools"] is None
        assert call_kwargs["tool_handlers"] is None
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["model"] is None


class TestAsyncGenerateExecutorBehavior:
    """P1: Test ThreadPoolExecutor behavior."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.fixture
    def custom_executor(self):
        executor = ThreadPoolExecutor(max_workers=2)
        yield executor
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_async_generate_uses_default_executor_when_none_provided(self, provider):
        """Default executor is used when no executor parameter is provided."""
        # The default executor should be used - we verify by checking the call works
        result = await provider.async_generate(
            system="test",
            user_message="test",
        )
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_async_generate_uses_custom_executor(self, provider, custom_executor):
        """Custom executor is used when provided."""
        # Track which executor was used
        used_executor = None
        original_run_in_executor = asyncio.get_event_loop().run_in_executor

        async def tracking_run_in_executor(executor, func):
            nonlocal used_executor
            used_executor = executor
            return await original_run_in_executor(executor, func)

        with patch.object(
            asyncio.get_event_loop(),
            'run_in_executor',
            side_effect=tracking_run_in_executor
        ):
            await provider.async_generate(
                system="test",
                user_message="test",
                executor=custom_executor,
            )

        # Custom executor should have been passed
        assert used_executor is custom_executor

    @pytest.mark.asyncio
    async def test_async_generate_concurrent_calls(self, provider):
        """Multiple concurrent calls complete successfully."""
        provider.generate_mock.return_value = "response"

        tasks = [
            provider.async_generate(system="test", user_message=f"msg{i}")
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(r == "response" for r in results)
        assert provider.generate_mock.call_count == 5

    @pytest.mark.asyncio
    async def test_async_generate_concurrent_calls_with_different_params(self, provider):
        """Concurrent calls with different parameters don't interfere."""
        call_log = []

        def tracking_generate(**kwargs):
            call_log.append(kwargs["user_message"])
            return f"response to {kwargs['user_message']}"

        provider.generate_mock.side_effect = tracking_generate

        tasks = [
            provider.async_generate(system="test", user_message=f"unique_msg_{i}")
            for i in range(3)
        ]

        results = await asyncio.gather(*tasks)

        # Each call should have received its unique message
        assert len(call_log) == 3
        for i in range(3):
            assert f"unique_msg_{i}" in call_log
            assert f"response to unique_msg_{i}" in results


class TestAsyncGenerateCode:
    """P1: Tests for async_generate_code method."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.mark.asyncio
    async def test_async_generate_code_extracts_python_block(self, provider):
        """Python code block is extracted correctly."""
        provider.generate_mock.return_value = """
Here is the code:

```python
def add(a, b):
    return a + b
```

That's the function.
"""

        result = await provider.async_generate_code(
            system="test",
            user_message="test",
        )

        assert result == "def add(a, b):\n    return a + b"
        assert "```" not in result

    @pytest.mark.asyncio
    async def test_async_generate_code_extracts_generic_block(self, provider):
        """Generic code block (no language) is extracted correctly."""
        provider.generate_mock.return_value = """
```
x = 1
y = 2
```
"""

        result = await provider.async_generate_code(
            system="test",
            user_message="test",
        )

        assert result == "x = 1\ny = 2"

    @pytest.mark.asyncio
    async def test_async_generate_code_returns_raw_when_no_block(self, provider):
        """Response without code blocks is returned stripped."""
        provider.generate_mock.return_value = "  x = 1  "

        result = await provider.async_generate_code(
            system="test",
            user_message="test",
        )

        assert result == "x = 1"

    @pytest.mark.asyncio
    async def test_async_generate_code_propagates_exception(self, provider):
        """Exception from async_generate propagates through async_generate_code."""
        provider.generate_mock.side_effect = ValueError("API error")

        with pytest.raises(ValueError, match="API error"):
            await provider.async_generate_code(
                system="test",
                user_message="test",
            )

    @pytest.mark.asyncio
    async def test_async_generate_code_forwards_parameters(self, provider):
        """All parameters are forwarded through async_generate_code."""
        provider.generate_mock.return_value = "```python\ncode\n```"
        tools = [{"name": "tool"}]

        await provider.async_generate_code(
            system="sys",
            user_message="user",
            tools=tools,
            max_tokens=500,
            model="model",
        )

        provider.generate_mock.assert_called_once_with(
            system="sys",
            user_message="user",
            tools=tools,
            tool_handlers=None,
            max_tokens=500,
            model="model",
        )

    @pytest.mark.asyncio
    async def test_async_generate_code_extracts_first_python_block(self, provider):
        """When multiple code blocks exist, first python block is extracted."""
        provider.generate_mock.return_value = """
First:
```python
first_code = 1
```

Second:
```python
second_code = 2
```
"""

        result = await provider.async_generate_code(
            system="test",
            user_message="test",
        )

        assert "first_code = 1" in result
        # The regex finds the first match


class TestAsyncGenerateEdgeCases:
    """P2: Edge case tests."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.mark.asyncio
    async def test_async_generate_with_empty_response(self, provider):
        """Empty string response is handled."""
        provider.generate_mock.return_value = ""

        result = await provider.async_generate(
            system="test",
            user_message="test",
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_async_generate_with_whitespace_response(self, provider):
        """Whitespace-only response is returned as-is."""
        provider.generate_mock.return_value = "   \n\t  "

        result = await provider.async_generate(
            system="test",
            user_message="test",
        )

        assert result == "   \n\t  "

    @pytest.mark.asyncio
    async def test_async_generate_with_unicode_response(self, provider):
        """Unicode characters in response are handled correctly."""
        provider.generate_mock.return_value = "Hello 世界 🌍 émojis"

        result = await provider.async_generate(
            system="test",
            user_message="test",
        )

        assert result == "Hello 世界 🌍 émojis"

    @pytest.mark.asyncio
    async def test_async_generate_with_very_long_response(self, provider):
        """Very long responses are handled correctly."""
        long_response = "x" * 100000
        provider.generate_mock.return_value = long_response

        result = await provider.async_generate(
            system="test",
            user_message="test",
        )

        assert result == long_response
        assert len(result) == 100000

    @pytest.mark.asyncio
    async def test_async_generate_with_empty_system_prompt(self, provider):
        """Empty system prompt is handled."""
        await provider.async_generate(
            system="",
            user_message="test",
        )

        call_kwargs = provider.generate_mock.call_args.kwargs
        assert call_kwargs["system"] == ""

    @pytest.mark.asyncio
    async def test_async_generate_code_with_empty_code_block(self, provider):
        """Empty code block returns empty string."""
        provider.generate_mock.return_value = "```python\n```"

        result = await provider.async_generate_code(
            system="test",
            user_message="test",
        )

        assert result == ""


class TestAsyncGenerateWithToolHandlers:
    """P1: Tests for tool handlers in async context."""

    @pytest.fixture
    def provider(self):
        return ConcreteTestProvider()

    @pytest.mark.asyncio
    async def test_async_generate_with_tool_handlers(self, provider):
        """Tool handlers are passed correctly to sync generate."""
        call_count = 0

        def tool_handler(arg):
            nonlocal call_count
            call_count += 1
            return f"handled: {arg}"

        handlers = {"my_tool": tool_handler}

        await provider.async_generate(
            system="test",
            user_message="test",
            tool_handlers=handlers,
        )

        call_kwargs = provider.generate_mock.call_args.kwargs
        assert call_kwargs["tool_handlers"] is handlers
        assert "my_tool" in call_kwargs["tool_handlers"]

    @pytest.mark.asyncio
    async def test_async_generate_with_multiple_tool_handlers(self, provider):
        """Multiple tool handlers are all passed correctly."""
        handlers = {
            "tool1": lambda x: f"t1: {x}",
            "tool2": lambda x: f"t2: {x}",
            "tool3": lambda x: f"t3: {x}",
        }

        await provider.async_generate(
            system="test",
            user_message="test",
            tool_handlers=handlers,
        )

        call_kwargs = provider.generate_mock.call_args.kwargs
        assert len(call_kwargs["tool_handlers"]) == 3


class TestDefaultExecutor:
    """Tests for the module-level default executor."""

    def test_default_executor_exists(self):
        """Module-level _DEFAULT_EXECUTOR should exist."""
        assert _DEFAULT_EXECUTOR is not None
        assert isinstance(_DEFAULT_EXECUTOR, ThreadPoolExecutor)

    def test_default_executor_has_workers(self):
        """Default executor should have max_workers configured."""
        # ThreadPoolExecutor has _max_workers attribute
        assert _DEFAULT_EXECUTOR._max_workers == 10


class TestAsyncGenerateIntegrationWithFactResolver:
    """Integration tests verifying async_generate works with AsyncFactResolver."""

    @pytest.mark.asyncio
    async def test_provider_async_generate_called_by_resolver(self):
        """AsyncFactResolver successfully calls provider's async_generate."""
        from constat.execution.fact_resolver import AsyncFactResolver

        # Create a mock LLM with async_generate
        mock_llm = MagicMock()
        mock_llm.async_generate = AsyncMock(
            return_value="VALUE: 42\nCONFIDENCE: 0.9\nTYPE: knowledge\nREASONING: Test"
        )

        resolver = AsyncFactResolver(llm=mock_llm)

        # Resolve a fact - should use async_generate
        result = await resolver._resolve_from_llm_async("test_fact", {})

        assert result is not None
        assert result.value == 42.0
        mock_llm.async_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_provider_fallback_when_no_async_generate(self):
        """Resolver falls back to sync generate in executor when no async_generate."""
        from constat.execution.fact_resolver import AsyncFactResolver

        # Create a mock LLM without async_generate
        mock_llm = MagicMock(spec=['generate', 'max_output_tokens'])  # Only has generate, not async_generate
        mock_llm.generate = MagicMock(
            return_value="VALUE: 99\nCONFIDENCE: 0.8\nTYPE: heuristic\nREASONING: Fallback"
        )
        mock_llm.max_output_tokens = 500

        resolver = AsyncFactResolver(llm=mock_llm)

        result = await resolver._resolve_from_llm_async("test_fact", {})

        assert result is not None
        assert result.value == 99.0
        # Should have called sync generate (run in executor)
        mock_llm.generate.assert_called()
