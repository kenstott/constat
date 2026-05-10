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

"""Tests for xAI Grok, Together AI, and Groq providers."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import GrokProvider, TogetherProvider, GroqProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


@pytest.fixture
def require_xai_key():
    if not os.environ.get("XAI_API_KEY"):
        pytest.fail("XAI_API_KEY not set — required for this test")


@pytest.fixture
def require_together_key():
    if not os.environ.get("TOGETHER_API_KEY"):
        pytest.fail("TOGETHER_API_KEY not set — required for this test")


@pytest.fixture
def require_groq_key():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.fail("GROQ_API_KEY not set — required for this test")


# =============================================================================
# Grok Provider Tests (Skipped by default)
# =============================================================================

class TestGrokProvider:
    """Tests for xAI Grok provider.

    These tests are skipped by default. Set XAI_API_KEY to run them.
    """

    def test_instantiation(self, require_xai_key):
        """Provider can be instantiated."""
        provider = GrokProvider()
        assert provider.model == "grok-2-latest"
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_xai_key):
        """Provider accepts custom model."""
        provider = GrokProvider(model="grok-2")
        assert provider.model == "grok-2"

    def test_generate_simple(self, require_xai_key):
        """Basic generation without tools."""
        provider = GrokProvider()
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

    def test_generate_code(self, require_xai_key):
        """Code generation extracts from markdown blocks."""
        provider = GrokProvider()
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

    def test_generate_with_tools(self, require_xai_key):
        """Generation with tool calling."""
        provider = GrokProvider()
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them to answer questions.",
                    user_message="What's the weather in Sydney?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Sydney" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_calculation_tool(self, require_xai_key):
        """Tool calling with calculation."""
        provider = GrokProvider()
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="Use the calculate tool for math. Report the result.",
                    user_message="What is 9 * 9?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "81" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


# =============================================================================
# Together Provider Tests (Skipped by default)
# =============================================================================

class TestTogetherProvider:
    """Tests for Together AI provider.

    These tests are skipped by default. Set TOGETHER_API_KEY to run them.
    """

    def test_instantiation(self, require_together_key):
        """Provider can be instantiated."""
        provider = TogetherProvider()
        assert "llama" in provider.model.lower()
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_together_key):
        """Provider accepts custom model."""
        provider = TogetherProvider(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        assert "8B" in provider.model

    def test_generate_simple(self, require_together_key):
        """Basic generation without tools."""
        provider = TogetherProvider()
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

    def test_generate_code(self, require_together_key):
        """Code generation extracts from markdown blocks."""
        provider = TogetherProvider()
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


# =============================================================================
# Groq Provider Tests (Skipped by default)
# =============================================================================

class TestGroqProvider:
    """Tests for Groq provider.

    These tests are skipped by default. Set GROQ_API_KEY to run them.
    """

    def test_instantiation(self, require_groq_key):
        """Provider can be instantiated."""
        provider = GroqProvider()
        assert "llama" in provider.model.lower()
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_groq_key):
        """Provider accepts custom model."""
        provider = GroqProvider(model="llama-3.1-8b-instant")
        assert "8b" in provider.model.lower()

    def test_generate_simple(self, require_groq_key):
        """Basic generation without tools."""
        provider = GroqProvider()
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

    def test_generate_code(self, require_groq_key):
        """Code generation extracts from markdown blocks."""
        provider = GroqProvider()
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
