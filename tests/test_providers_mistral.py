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

"""Tests for Mistral and Codestral providers."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import MistralProvider, CodestralProvider, OllamaProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


@pytest.fixture
def require_mistral_key():
    if not os.environ.get("MISTRAL_API_KEY"):
        pytest.fail("MISTRAL_API_KEY not set — required for this test")


# =============================================================================
# Mistral Provider Unit Tests (no API required)
# =============================================================================

class TestMistralProviderUnit:
    """Unit tests for MistralProvider - no API required."""

    def test_instantiation_requires_api_key(self):
        """Provider requires API key."""
        # Clear any existing env var for this test
        old_key = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="API key required"):
                MistralProvider()
        finally:
            if old_key:
                os.environ["MISTRAL_API_KEY"] = old_key

    def test_list_models_returns_dict(self):
        """list_models() returns available models."""
        models = MistralProvider.list_models()
        assert isinstance(models, dict)
        assert "mistral-large-latest" in models
        assert "codestral-latest" in models

    def test_codestral_provider_default_model(self):
        """CodestralProvider defaults to codestral model."""
        # This will fail without API key, but we can check the class
        assert CodestralProvider.__bases__[0] == MistralProvider


# =============================================================================
# Mistral Provider Tests (Skipped by default)
# =============================================================================

class TestMistralProvider:
    """Tests for Mistral AI provider.

    These tests are skipped by default. Set MISTRAL_API_KEY to run them.
    """

    def test_instantiation(self, require_mistral_key):
        """Provider can be instantiated."""
        provider = MistralProvider()
        assert provider.model == "mistral-large-latest"
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_mistral_key):
        """Provider accepts custom model."""
        provider = MistralProvider(model="mistral-small-latest")
        assert provider.model == "mistral-small-latest"

    def test_generate_simple(self, require_mistral_key):
        """Basic generation without tools."""
        provider = MistralProvider(model="mistral-small-latest")
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

    def test_generate_code(self, require_mistral_key):
        """Code generation extracts from markdown blocks."""
        provider = MistralProvider(model="mistral-small-latest")
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

    def test_generate_with_tools(self, require_mistral_key):
        """Generation with tool calling."""
        provider = MistralProvider(model="mistral-small-latest")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them to answer questions.",
                    user_message="What's the weather in Madrid?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Madrid" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_calculation_tool(self, require_mistral_key):
        """Tool calling with calculation."""
        provider = MistralProvider(model="mistral-small-latest")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="Use the calculate tool for math. Report the result.",
                    user_message="What is 17 * 6?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "102" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_mistral_large_generation(self, require_mistral_key):
        """Test with Mistral Large model."""
        provider = MistralProvider(model="mistral-large-latest")
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You are a helpful assistant. Be concise.",
                    user_message="What is the capital of France? Reply with just the city name.",
                    max_tokens=50,
                )
                assert "Paris" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


# =============================================================================
# Codestral Provider Tests
# =============================================================================

class TestCodestralProvider:
    """Tests for Codestral (code-specialized Mistral model).

    These tests are skipped by default. Set MISTRAL_API_KEY to run them.
    """

    def test_instantiation(self, require_mistral_key):
        """Provider can be instantiated."""
        provider = CodestralProvider()
        assert "codestral" in provider.model.lower()
        assert provider.supports_tools is True

    def test_generate_code(self, require_mistral_key):
        """Code generation with Codestral."""
        provider = CodestralProvider()
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate_code(
                    system="You are a code expert. Return only code in markdown blocks.",
                    user_message="Write a Python function to check if a number is prime.",
                    max_tokens=300,
                )
                assert "def" in response
                assert "prime" in response.lower() or "%" in response
                assert "```" not in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_sql(self, require_mistral_key):
        """SQL generation with Codestral."""
        provider = CodestralProvider()
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate_code(
                    system="You are a SQL expert. Return only SQL in markdown blocks.",
                    user_message="Write a SQL query to get the top 5 customers by total orders.",
                    max_tokens=200,
                )
                assert "SELECT" in response.upper()
                assert "```" not in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


# =============================================================================
# Mistral Nemo Integration Tests (via Ollama - local or Docker)
# =============================================================================

class TestMistralNemoIntegration:
    """Integration tests for Mistral Nemo via Ollama.

    These tests use the mistral_container fixture which:
    1. Prefers local Ollama if running
    2. Falls back to Docker if local not available

    Mistral Nemo is a 12B open-weight model.
    """

    def test_generate_simple(self, mistral_container):
        """Basic generation with Mistral Nemo."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You are a helpful assistant. Be very brief.",
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

    def test_generate_code(self, mistral_container):
        """Code generation with Mistral Nemo."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate_code(
                    system="You are a Python expert. Return only code in markdown blocks.",
                    user_message="Write a one-line function that returns the sum of a and b.",
                    max_tokens=200,
                )
                code_tokens = ("def", "lambda", "+", "import", "return", "(", "=")
                assert any(tok in response for tok in code_tokens), (
                    f"Response does not look like Python code: {response!r}"
                )
                assert "```" not in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_tools(self, mistral_container):
        """Tool calling with Mistral Nemo."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        if not provider.supports_tools:
            pytest.fail("Model does not support tools")

        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them when needed.",
                    user_message="What's the weather in Paris?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Paris" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_multiple_generations(self, mistral_container):
        """Multiple generations work without issues."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        last_exc = None
        for attempt in range(3):
            try:
                # First generation
                response1 = provider.generate(
                    system="You are a helpful assistant. Be brief.",
                    user_message="What is 1 + 1? Reply with just the number.",
                    max_tokens=20,
                )
                assert "2" in response1

                # Second generation
                response2 = provider.generate(
                    system="You are a helpful assistant. Be brief.",
                    user_message="What is 3 + 3? Reply with just the number.",
                    max_tokens=20,
                )
                assert "6" in response2
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_supports_tools_property(self, mistral_container):
        """Mistral Nemo should support tool calling."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        # mistral-nemo should be recognized as tool-capable
        assert provider.supports_tools is True
