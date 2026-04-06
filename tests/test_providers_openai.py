from __future__ import annotations

# Copyright (c) 2025 Kenneth Stott
# Canary: f0e226dd-3488-41ae-80c7-40d2fbe56fa8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for OpenAI provider."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import OpenAIProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


@pytest.fixture
def require_openai_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY not set — required for this test")


# =============================================================================
# OpenAI Provider Tests (Skipped by default)
# =============================================================================

class TestOpenAIProvider:
    """Tests for OpenAI GPT provider.

    These tests are skipped by default. Set OPENAI_API_KEY to run them.
    """

    def test_instantiation(self, require_openai_key):
        """Provider can be instantiated."""
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o"
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_openai_key):
        """Provider accepts custom model."""
        provider = OpenAIProvider(model="gpt-4-turbo")
        assert provider.model == "gpt-4-turbo"

    def test_generate_simple(self, require_openai_key):
        """Basic generation without tools."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You are a helpful assistant. Be concise.",
                    user_message="What is 2 + 2? Reply with just the number.",
                    max_tokens=50,
                )
                assert "4" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_code(self, require_openai_key):
        """Code generation extracts from markdown blocks."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate_code(
                    system="You are a Python expert. Return only code in markdown blocks.",
                    user_message="Write a function that adds two numbers. Just the function.",
                    max_tokens=200,
                )
                assert "def" in response
                assert "```" not in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_tools(self, require_openai_key):
        """Generation with tool calling."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them to answer questions.",
                    user_message="What's the weather in Tokyo?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Tokyo" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_calculation_tool(self, require_openai_key):
        """Tool calling with calculation."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="Use the calculate tool for math. Report the result.",
                    user_message="What is 23 * 4?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "92" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_tool_format_conversion(self, require_openai_key):
        """Tools are converted to OpenAI format correctly."""
        openai_tools = OpenAIProvider.convert_tools_to_openai_format(SAMPLE_TOOLS)
        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "get_weather"
        assert "parameters" in openai_tools[0]["function"]
