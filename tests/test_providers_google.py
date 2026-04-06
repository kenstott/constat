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

"""Tests for Google Gemini provider."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import GeminiProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


@pytest.fixture
def require_google_key():
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.fail("GOOGLE_API_KEY not set — required for this test")


# =============================================================================
# Gemini Provider Tests (Skipped by default)
# =============================================================================

class TestGeminiProvider:
    """Tests for Google Gemini provider.

    These tests are skipped by default. Set GOOGLE_API_KEY to run them.
    """

    def test_instantiation(self, require_google_key):
        """Provider can be instantiated."""
        provider = GeminiProvider()
        assert provider.model_name == "gemini-1.5-pro"
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_google_key):
        """Provider accepts custom model."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        assert provider.model_name == "gemini-1.5-flash"

    def test_generate_simple(self, require_google_key):
        """Basic generation without tools."""
        provider = GeminiProvider(model="gemini-1.5-flash")
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

    def test_generate_code(self, require_google_key):
        """Code generation extracts from markdown blocks."""
        provider = GeminiProvider(model="gemini-1.5-flash")
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

    def test_generate_with_tools(self, require_google_key):
        """Generation with tool calling."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them to answer questions.",
                    user_message="What's the weather in Berlin?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Berlin" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_calculation_tool(self, require_google_key):
        """Tool calling with calculation."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="Use the calculate tool for math. Report the result.",
                    user_message="What is 12 * 8?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "96" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc
