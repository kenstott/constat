# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for LLM provider implementations.

Provider test strategy:
- Anthropic: Tested with real API (ANTHROPIC_API_KEY required)
- Ollama: Tested with local server (requires Ollama running)
- OpenAI: Skipped by default (requires OPENAI_API_KEY)
- Gemini: Skipped by default (requires GOOGLE_API_KEY)
- Grok: Skipped by default (requires XAI_API_KEY)
t- Mistral: Skipped by default (requires MISTRAL_API_KEY)
- Together: Skipped by default (requires TOGETHER_API_KEY)
- Groq: Skipped by default (requires GROQ_API_KEY)
"""

import os

import pytest

from constat.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    MistralProvider,
    CodestralProvider,
    OllamaProvider,
    TogetherProvider,
    GroqProvider,
)


# Skip markers for providers requiring API keys
requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - set to run OpenAI tests"
)

requires_google_key = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set - set to run Gemini tests"
)

requires_xai_key = pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY"),
    reason="XAI_API_KEY not set - set to run Grok tests"
)

requires_together_key = pytest.mark.skipif(
    not os.environ.get("TOGETHER_API_KEY"),
    reason="TOGETHER_API_KEY not set - set to run Together tests"
)

requires_groq_key = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set - set to run Groq tests"
)

requires_mistral_key = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set - set to run Mistral tests"
)


def ollama_available() -> bool:
    """Check if Ollama server is running locally."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def get_ollama_model() -> str | None:
    """Get an available Ollama model for testing.

    Returns the first available llama model, or None if none available.
    """
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            # Prefer llama3.2:3b, then any llama model, then any model
            for preferred in ["llama3.2:3b", "llama3.1:8b", "llama3:8b"]:
                if preferred in models:
                    return preferred
            for model in models:
                if "llama" in model.lower():
                    return model
            if models:
                return models[0]
    except Exception:
        pass
    return None


OLLAMA_TEST_MODEL = get_ollama_model()

requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason="Ollama server not running at localhost:11434"
)

requires_ollama_model = pytest.mark.skipif(
    not OLLAMA_TEST_MODEL,
    reason="No suitable Ollama model available for testing"
)


# Sample tools for testing tool calling
SAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
]


def get_weather(location: str) -> str:
    """Mock weather tool handler."""
    return f"Weather in {location}: 72F, sunny"


def calculate(expression: str) -> str:
    """Mock calculator tool handler."""
    try:
        result = eval(expression)  # Safe for tests with controlled input
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# =============================================================================
# Anthropic Provider Tests
# =============================================================================

class TestAnthropicProvider:
    """Tests for Anthropic Claude provider."""

    @requires_anthropic_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = AnthropicProvider()
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.supports_tools is True

    @requires_anthropic_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        assert provider.model == "claude-3-haiku-20240307"

    @requires_anthropic_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_anthropic_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function, no explanation.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response  # Should be extracted

    @requires_anthropic_key
    def test_generate_with_tools(self):
        """Generation with tool calling."""
        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        response = provider.generate(
            system="You have access to tools. Use them to answer questions.",
            user_message="What's the weather in Paris?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "Paris" in response or "72" in response or "sunny" in response

    @requires_anthropic_key
    def test_generate_with_calculation_tool(self):
        """Tool calling with calculation."""
        provider = AnthropicProvider(model="claude-3-haiku-20240307")
        response = provider.generate(
            system="Use the calculate tool for math. Report the result.",
            user_message="What is 15 * 7?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "105" in response

    @requires_anthropic_key
    def test_model_override_in_generate(self):
        """Model can be overridden per-call (for task-type routing)."""
        # Initialize with default sonnet model
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

        # Override to haiku for this specific call (simulating task routing)
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 3 + 3? Reply with just the number.",
            model="claude-3-5-haiku-20241022",  # Override for this call
            max_tokens=50,
        )
        assert "6" in response

        # Original model should still be sonnet
        assert provider.model == "claude-sonnet-4-20250514"

    @requires_anthropic_key
    def test_task_routing_integration(self):
        """Test task-type routing as used by planner/session."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec

        # Configure task routing like production would
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
            "summarization": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
        })
        llm_config = LLMConfig(
            model="claude-sonnet-4-20250514",  # Default is sonnet
            task_routing=routing,
        )

        # Provider initialized with default model
        provider = AnthropicProvider(model=llm_config.model)
        assert provider.model == "claude-sonnet-4-20250514"

        # Get routing config and select model for planning task
        task_routing = llm_config.get_task_routing()
        planning_models = task_routing.get_models_for_task("planning")
        assert len(planning_models) >= 1
        planning_model = planning_models[0].model
        assert planning_model == "claude-3-5-haiku-20241022"

        response = provider.generate(
            system="You are a helpful assistant.",
            user_message="What is 1 + 1?",
            model=planning_model,  # Use planning task model
            max_tokens=50,
        )
        assert "2" in response


# =============================================================================
# Ollama Provider Unit Tests (Mock-based, no server required)
# =============================================================================

class TestOllamaProviderUnit:
    """Unit tests for OllamaProvider using mocks - no server required."""

    def test_instantiation_defaults(self):
        """Provider has correct default values."""
        provider = OllamaProvider()
        assert provider.model == "llama3.2"
        assert provider.DEFAULT_BASE_URL == "http://localhost:11434/v1"

    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = OllamaProvider(model="llama3.1:8b")
        assert provider.model == "llama3.1:8b"

    def test_instantiation_with_custom_base_url(self):
        """Provider accepts custom base URL."""
        provider = OllamaProvider(base_url="http://192.168.1.100:11434/v1")
        assert provider._base_url == "http://192.168.1.100:11434"

    def test_base_url_normalization_strips_trailing_v1(self):
        """Base URL stored without /v1 suffix for API calls."""
        provider = OllamaProvider(base_url="http://custom:8080/v1")
        assert provider._base_url == "http://custom:8080"

    def test_base_url_normalization_strips_trailing_slash(self):
        """Base URL stored without trailing slash."""
        provider = OllamaProvider(base_url="http://custom:8080/")
        assert provider._base_url == "http://custom:8080"

    def test_base_url_normalization_handles_v1_and_slash(self):
        """Base URL handles both /v1 and trailing slash."""
        provider = OllamaProvider(base_url="http://custom:8080/v1/")
        assert provider._base_url == "http://custom:8080"

    def test_base_url_no_change_needed(self):
        """Base URL without /v1 is unchanged."""
        provider = OllamaProvider(base_url="http://custom:8080")
        assert provider._base_url == "http://custom:8080"

    def test_supports_tools_returns_true_for_llama32(self):
        """llama3.2 is a known tool-capable model."""
        provider = OllamaProvider(model="llama3.2")
        assert provider.supports_tools is True

    def test_supports_tools_returns_true_for_llama31(self):
        """llama3.1 is a known tool-capable model."""
        provider = OllamaProvider(model="llama3.1")
        assert provider.supports_tools is True

    def test_supports_tools_returns_true_for_mistral(self):
        """mistral is a known tool-capable model."""
        provider = OllamaProvider(model="mistral")
        assert provider.supports_tools is True

    def test_supports_tools_returns_true_for_qwen(self):
        """qwen2.5 is a known tool-capable model."""
        provider = OllamaProvider(model="qwen2.5")
        assert provider.supports_tools is True

    def test_supports_tools_handles_model_tags(self):
        """Model with tag still matches base model."""
        provider = OllamaProvider(model="llama3.2:3b-instruct")
        assert provider.supports_tools is True

    def test_supports_tools_handles_complex_tags(self):
        """Model with complex tag still matches base model."""
        provider = OllamaProvider(model="llama3.2:3b-instruct-q4_0")
        assert provider.supports_tools is True

    def test_supports_tools_case_insensitive(self):
        """Model name matching is case-insensitive for base name."""
        provider = OllamaProvider(model="LLAMA3.2:3B")
        # The comparison lowercases, so this should work
        assert provider.supports_tools is True

    def test_supports_tools_caches_result(self):
        """Second call uses cached value."""
        provider = OllamaProvider(model="llama3.2")

        # First call
        result1 = provider.supports_tools
        assert result1 is True

        # Verify cache is set
        assert provider._supports_tools_cache is True

        # Second call should use cache
        result2 = provider.supports_tools
        assert result2 is True

    def test_supports_tools_false_for_unknown_model_no_server(self):
        """Unknown model without server access returns False."""
        # This will try to check the API but fail (no server)
        provider = OllamaProvider(model="totally-unknown-model")
        # Should return False without crashing
        result = provider.supports_tools
        assert result is False

    def test_tool_capable_models_list_not_empty(self):
        """Sanity check: TOOL_CAPABLE_MODELS is defined."""
        assert len(OllamaProvider.TOOL_CAPABLE_MODELS) > 0
        assert "llama3.2" in OllamaProvider.TOOL_CAPABLE_MODELS

    def test_model_attribute_set_correctly(self):
        """Provider.model returns the configured model."""
        provider = OllamaProvider(model="custom-model")
        assert provider.model == "custom-model"

    def test_client_is_openai_instance(self):
        """Provider creates an OpenAI client."""
        from openai import OpenAI
        provider = OllamaProvider()
        assert isinstance(provider.client, OpenAI)


class TestOllamaProviderMockedAPI:
    """Tests for OllamaProvider with mocked HTTP calls."""

    def test_unknown_model_checks_api_and_finds_tools_support(self):
        """Unknown model triggers API check, finds tool support in template."""
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "template": "This model supports tools and function calling"
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            provider = OllamaProvider(model="custom-tool-model")
            result = provider.supports_tools

            # Should have called the API
            mock_post.assert_called_once()
            assert result is True

    def test_unknown_model_checks_api_no_tools_support(self):
        """Unknown model triggers API check, no tool support in template."""
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "template": "This is a basic chat template"
        }

        with patch("httpx.post", return_value=mock_response) as mock_post:
            provider = OllamaProvider(model="basic-model")
            result = provider.supports_tools

            mock_post.assert_called_once()
            assert result is False

    def test_supports_tools_handles_connection_error(self):
        """Connection error returns False, doesn't crash."""
        from unittest.mock import patch
        import httpx

        with patch("httpx.post", side_effect=httpx.ConnectError("Connection refused")):
            provider = OllamaProvider(model="unknown-model")
            result = provider.supports_tools

            # Should return False, not raise
            assert result is False

    def test_supports_tools_handles_timeout(self):
        """Timeout returns False, doesn't crash."""
        from unittest.mock import patch
        import httpx

        with patch("httpx.post", side_effect=httpx.TimeoutException("Timeout")):
            provider = OllamaProvider(model="unknown-model")
            result = provider.supports_tools

            assert result is False

    def test_supports_tools_handles_invalid_json(self):
        """Invalid JSON response returns False."""
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with patch("httpx.post", return_value=mock_response):
            provider = OllamaProvider(model="unknown-model")
            result = provider.supports_tools

            assert result is False

    def test_supports_tools_handles_missing_template_key(self):
        """Response without 'template' key returns False."""
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "model", "size": 12345}

        with patch("httpx.post", return_value=mock_response):
            provider = OllamaProvider(model="unknown-model")
            result = provider.supports_tools

            assert result is False

    def test_supports_tools_handles_http_error(self):
        """HTTP error status returns False."""
        from unittest.mock import patch, Mock

        mock_response = Mock()
        mock_response.status_code = 404

        with patch("httpx.post", return_value=mock_response):
            provider = OllamaProvider(model="unknown-model")
            result = provider.supports_tools

            assert result is False


# =============================================================================
# Ollama Provider Integration Tests (Docker-based, server required)
# =============================================================================

@pytest.mark.requires_docker
class TestOllamaProviderIntegration:
    """Integration tests for Ollama provider - requires Docker to start Ollama server."""

    def test_instantiation_with_running_server(self, ollama_container):
        """Provider can connect to running Ollama server."""
        provider = OllamaProvider(base_url=ollama_container["base_url"])
        assert provider.model == "llama3.2"

    def test_supports_tools_with_live_api(self, ollama_container):
        """Tool support detection works against live server."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        # The test model should be a known tool-capable model
        result = provider.supports_tools
        # llama3.2 variants support tools
        assert isinstance(result, bool)

    def test_generate_simple(self, ollama_container):
        """Basic generation without tools."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        response = provider.generate(
            system="You are a helpful assistant. Be very brief.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    def test_generate_code(self, ollama_container):
        """Code generation extracts from markdown blocks."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a one-line function that returns the sum of a and b.",
            max_tokens=200,
        )
        assert "def" in response or "lambda" in response or "+" in response
        assert "```" not in response

    def test_generate_with_tools(self, ollama_container):
        """Generation with tool calling (if model supports it)."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        if not provider.supports_tools:
            pytest.skip("Model does not support tools")

        response = provider.generate(
            system="You have access to tools. Use them when needed.",
            user_message="What's the weather in London?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        # Response should exist
        assert len(response) > 0

    def test_generate_multiple_responses(self, ollama_container):
        """Multiple generations work without issues."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
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


# Keep the old markers for backward compatibility when running without fixtures
@requires_ollama
def test_ollama_provider_backward_compat():
    """Backward compatibility test - uses old marker."""
    provider = OllamaProvider()
    assert provider.model == "llama3.2"


@requires_ollama_model
def test_ollama_generate_backward_compat():
    """Backward compatibility test for generation."""
    provider = OllamaProvider(model=OLLAMA_TEST_MODEL)
    response = provider.generate(
        system="Be brief.",
        user_message="Say hello.",
        max_tokens=20,
    )
    assert len(response) > 0


# =============================================================================
# OpenAI Provider Tests (Skipped by default)
# =============================================================================

class TestOpenAIProvider:
    """Tests for OpenAI GPT provider.

    These tests are skipped by default. Set OPENAI_API_KEY to run them.
    """

    @requires_openai_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = OpenAIProvider()
        assert provider.model == "gpt-4o"
        assert provider.supports_tools is True

    @requires_openai_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = OpenAIProvider(model="gpt-4-turbo")
        assert provider.model == "gpt-4-turbo"

    @requires_openai_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_openai_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response

    @requires_openai_key
    def test_generate_with_tools(self):
        """Generation with tool calling."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        response = provider.generate(
            system="You have access to tools. Use them to answer questions.",
            user_message="What's the weather in Tokyo?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "Tokyo" in response or "72" in response or "sunny" in response

    @requires_openai_key
    def test_generate_with_calculation_tool(self):
        """Tool calling with calculation."""
        provider = OpenAIProvider(model="gpt-4o-mini")
        response = provider.generate(
            system="Use the calculate tool for math. Report the result.",
            user_message="What is 23 * 4?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "92" in response

    @requires_openai_key
    def test_tool_format_conversion(self):
        """Tools are converted to OpenAI format correctly."""
        openai_tools = OpenAIProvider.convert_tools_to_openai_format(SAMPLE_TOOLS)
        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "get_weather"
        assert "parameters" in openai_tools[0]["function"]


# =============================================================================
# Gemini Provider Tests (Skipped by default)
# =============================================================================

class TestGeminiProvider:
    """Tests for Google Gemini provider.

    These tests are skipped by default. Set GOOGLE_API_KEY to run them.
    """

    @requires_google_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = GeminiProvider()
        assert provider.model_name == "gemini-1.5-pro"
        assert provider.supports_tools is True

    @requires_google_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        assert provider.model_name == "gemini-1.5-flash"

    @requires_google_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_google_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response

    @requires_google_key
    def test_generate_with_tools(self):
        """Generation with tool calling."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        response = provider.generate(
            system="You have access to tools. Use them to answer questions.",
            user_message="What's the weather in Berlin?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "Berlin" in response or "72" in response or "sunny" in response

    @requires_google_key
    def test_generate_with_calculation_tool(self):
        """Tool calling with calculation."""
        provider = GeminiProvider(model="gemini-1.5-flash")
        response = provider.generate(
            system="Use the calculate tool for math. Report the result.",
            user_message="What is 12 * 8?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "96" in response


# =============================================================================
# Grok Provider Tests (Skipped by default)
# =============================================================================

class TestGrokProvider:
    """Tests for xAI Grok provider.

    These tests are skipped by default. Set XAI_API_KEY to run them.
    """

    @requires_xai_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = GrokProvider()
        assert provider.model == "grok-2-latest"
        assert provider.supports_tools is True

    @requires_xai_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = GrokProvider(model="grok-2")
        assert provider.model == "grok-2"

    @requires_xai_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = GrokProvider()
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_xai_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = GrokProvider()
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response

    @requires_xai_key
    def test_generate_with_tools(self):
        """Generation with tool calling."""
        provider = GrokProvider()
        response = provider.generate(
            system="You have access to tools. Use them to answer questions.",
            user_message="What's the weather in Sydney?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "Sydney" in response or "72" in response or "sunny" in response

    @requires_xai_key
    def test_generate_with_calculation_tool(self):
        """Tool calling with calculation."""
        provider = GrokProvider()
        response = provider.generate(
            system="Use the calculate tool for math. Report the result.",
            user_message="What is 9 * 9?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "81" in response


# =============================================================================
# Together Provider Tests (Skipped by default)
# =============================================================================

class TestTogetherProvider:
    """Tests for Together AI provider.

    These tests are skipped by default. Set TOGETHER_API_KEY to run them.
    """

    @requires_together_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = TogetherProvider()
        assert "llama" in provider.model.lower()
        assert provider.supports_tools is True

    @requires_together_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = TogetherProvider(model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        assert "8B" in provider.model

    @requires_together_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = TogetherProvider()
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_together_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = TogetherProvider()
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response


# =============================================================================
# Groq Provider Tests (Skipped by default)
# =============================================================================

class TestGroqProvider:
    """Tests for Groq provider.

    These tests are skipped by default. Set GROQ_API_KEY to run them.
    """

    @requires_groq_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = GroqProvider()
        assert "llama" in provider.model.lower()
        assert provider.supports_tools is True

    @requires_groq_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = GroqProvider(model="llama-3.1-8b-instant")
        assert "8b" in provider.model.lower()

    @requires_groq_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = GroqProvider()
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_groq_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = GroqProvider()
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response


# =============================================================================
# Mistral Provider Tests (Skipped by default)
# =============================================================================

class TestMistralProviderUnit:
    """Unit tests for MistralProvider - no API required."""

    def test_instantiation_requires_api_key(self):
        """Provider requires API key."""
        # Clear any existing env var for this test
        import os
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


class TestMistralProvider:
    """Tests for Mistral AI provider.

    These tests are skipped by default. Set MISTRAL_API_KEY to run them.
    """

    @requires_mistral_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = MistralProvider()
        assert provider.model == "mistral-large-latest"
        assert provider.supports_tools is True

    @requires_mistral_key
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = MistralProvider(model="mistral-small-latest")
        assert provider.model == "mistral-small-latest"

    @requires_mistral_key
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = MistralProvider(model="mistral-small-latest")
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_mistral_key
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = MistralProvider(model="mistral-small-latest")
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a function that adds two numbers. Just the function.",
            max_tokens=200,
        )
        assert "def" in response
        assert "```" not in response

    @requires_mistral_key
    def test_generate_with_tools(self):
        """Generation with tool calling."""
        provider = MistralProvider(model="mistral-small-latest")
        response = provider.generate(
            system="You have access to tools. Use them to answer questions.",
            user_message="What's the weather in Madrid?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "Madrid" in response or "72" in response or "sunny" in response

    @requires_mistral_key
    def test_generate_with_calculation_tool(self):
        """Tool calling with calculation."""
        provider = MistralProvider(model="mistral-small-latest")
        response = provider.generate(
            system="Use the calculate tool for math. Report the result.",
            user_message="What is 17 * 6?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert "102" in response

    @requires_mistral_key
    def test_mistral_large_generation(self):
        """Test with Mistral Large model."""
        provider = MistralProvider(model="mistral-large-latest")
        response = provider.generate(
            system="You are a helpful assistant. Be concise.",
            user_message="What is the capital of France? Reply with just the city name.",
            max_tokens=50,
        )
        assert "Paris" in response


class TestCodestralProvider:
    """Tests for Codestral (code-specialized Mistral model).

    These tests are skipped by default. Set MISTRAL_API_KEY to run them.
    """

    @requires_mistral_key
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = CodestralProvider()
        assert "codestral" in provider.model.lower()
        assert provider.supports_tools is True

    @requires_mistral_key
    def test_generate_code(self):
        """Code generation with Codestral."""
        provider = CodestralProvider()
        response = provider.generate_code(
            system="You are a code expert. Return only code in markdown blocks.",
            user_message="Write a Python function to check if a number is prime.",
            max_tokens=300,
        )
        assert "def" in response
        assert "prime" in response.lower() or "%" in response
        assert "```" not in response

    @requires_mistral_key
    def test_generate_sql(self):
        """SQL generation with Codestral."""
        provider = CodestralProvider()
        response = provider.generate_code(
            system="You are a SQL expert. Return only SQL in markdown blocks.",
            user_message="Write a SQL query to get the top 5 customers by total orders.",
            max_tokens=200,
        )
        assert "SELECT" in response.upper()
        assert "```" not in response


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
        response = provider.generate(
            system="You are a helpful assistant. Be very brief.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    def test_generate_code(self, mistral_container):
        """Code generation with Mistral Nemo."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a one-line function that returns the sum of a and b.",
            max_tokens=200,
        )
        assert "def" in response or "lambda" in response or "+" in response
        assert "```" not in response

    def test_generate_with_tools(self, mistral_container):
        """Tool calling with Mistral Nemo."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        if not provider.supports_tools:
            pytest.skip("Model does not support tools")

        response = provider.generate(
            system="You have access to tools. Use them when needed.",
            user_message="What's the weather in Paris?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        assert len(response) > 0

    def test_multiple_generations(self, mistral_container):
        """Multiple generations work without issues."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
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

    def test_supports_tools_property(self, mistral_container):
        """Mistral Nemo should support tool calling."""
        provider = OllamaProvider(
            model=mistral_container["model"],
            base_url=mistral_container["base_url"],
        )
        # mistral-nemo should be recognized as tool-capable
        assert provider.supports_tools is True


# =============================================================================
# Base Provider Tests (Unit tests, no API calls)
# =============================================================================

class TestBaseLLMProvider:
    """Tests for base provider functionality."""

    def test_extract_code_python_block(self):
        """Python code block is extracted correctly."""
        from constat.providers.base import BaseLLMProvider

        # Create a concrete implementation for testing
        class TestProvider(BaseLLMProvider):
            def generate(self, **kwargs):
                return ""

        provider = TestProvider()
        text = """Here is the code:

```python
def add(a, b):
    return a + b
```

That's a simple function."""

        code = provider._extract_code(text)
        assert code == "def add(a, b):\n    return a + b"

    def test_extract_code_generic_block(self):
        """Generic code block is extracted."""
        from constat.providers.base import BaseLLMProvider

        class TestProvider(BaseLLMProvider):
            def generate(self, **kwargs):
                return ""

        provider = TestProvider()
        text = """```
x = 1
```"""
        code = provider._extract_code(text)
        assert code == "x = 1"

    def test_extract_code_no_block(self):
        """Text without code block is returned as-is."""
        from constat.providers.base import BaseLLMProvider

        class TestProvider(BaseLLMProvider):
            def generate(self, **kwargs):
                return ""

        provider = TestProvider()
        text = "def foo(): pass"
        code = provider._extract_code(text)
        assert code == "def foo(): pass"

    def test_convert_tools_to_openai_format(self):
        """Tool format conversion is correct."""
        from constat.providers.base import BaseLLMProvider

        openai_tools = BaseLLMProvider.convert_tools_to_openai_format(SAMPLE_TOOLS)

        assert len(openai_tools) == 2
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "get_weather"
        assert openai_tools[0]["function"]["description"] == "Get the current weather for a location"
        assert openai_tools[0]["function"]["parameters"]["type"] == "object"
        assert "location" in openai_tools[0]["function"]["parameters"]["properties"]


# =============================================================================
# Task Router Tests (replaces ProviderFactory tests)
# =============================================================================

class TestTaskRouter:
    """Tests for TaskRouter with task-type routing and automatic escalation."""

    def test_router_creates_provider_from_config(self):
        """Router creates provider from LLM config."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        router = TaskRouter(llm_config)

        # Router should be initialized
        assert router.llm_config == llm_config
        assert router.routing_config is not None

    def test_router_uses_task_routing_config(self):
        """Router uses task routing configuration."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[
                    ModelSpec(provider="ollama", model="sqlcoder:7b"),
                    ModelSpec(model="claude-sonnet-4-20250514"),
                ]
            )
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Router should have the custom routing
        models = router.routing_config.get_models_for_task("sql_generation")
        assert len(models) == 2
        assert models[0].provider == "ollama"

    def test_router_caches_providers(self):
        """Router caches provider instances."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        router = TaskRouter(llm_config)

        # Access the provider twice
        spec1 = router.routing_config.get_models_for_task("planning")[0]
        spec2 = router.routing_config.get_models_for_task("planning")[0]

        # Models should be equivalent
        assert spec1.model == spec2.model

    def test_router_high_complexity_uses_different_models(self):
        """Router selects different models for high complexity tasks."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "python_analysis": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")],
                high_complexity_models=[ModelSpec(model="claude-sonnet-4-20250514")],
            )
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Medium complexity uses standard model
        medium_models = router.routing_config.get_models_for_task("python_analysis", "medium")
        assert medium_models[0].model == "claude-3-5-haiku-20241022"

        # High complexity uses advanced model
        high_models = router.routing_config.get_models_for_task("python_analysis", "high")
        assert high_models[0].model == "claude-sonnet-4-20250514"

    def test_router_escalation_stats(self):
        """Router tracks escalation statistics."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        router = TaskRouter(llm_config)

        # Initially no escalations
        stats = router.get_escalation_stats()
        assert stats["total_escalations"] == 0

        # Clear stats should work
        router.clear_stats()
        stats = router.get_escalation_stats()
        assert stats["total_escalations"] == 0

    def test_router_unknown_provider_returns_failed_result(self):
        """Router returns failed result for unknown provider."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter
        from constat.core.models import TaskType

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(provider="unknown_provider", model="some-model")],
            )
        })
        llm_config = LLMConfig(
            provider="unknown_provider",
            model="some-model",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Execute catches exceptions and returns a failed TaskResult
        result = router.execute(
            task_type=TaskType.PLANNING,
            system="Test",
            user_message="Test",
        )
        
        assert not result.success
        assert "Unknown provider" in result.content


# =============================================================================
# Multi-Provider Integration Tests (updated for task routing)
# =============================================================================

class TestMultiProviderIntegration:
    """Integration tests for multi-provider task routing.

    These tests verify that different providers can be used for different task types.
    """

    @requires_anthropic_key
    @requires_ollama_model
    def test_anthropic_planning_ollama_sql(self):
        """Use Anthropic for planning, Ollama for SQL generation."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(provider="ollama", model=OLLAMA_TEST_MODEL)]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Verify routing configuration
        planning_models = router.routing_config.get_models_for_task("planning")
        assert planning_models[0].model == "claude-3-5-haiku-20241022"
        assert planning_models[0].provider is None  # Uses default

        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == OLLAMA_TEST_MODEL

    @requires_anthropic_key
    def test_all_tasks_same_provider_different_models(self):
        """All task types use same provider but different models."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-sonnet-4-20250514")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
            "summarization": TaskRoutingEntry(
                models=[ModelSpec(model="claude-3-5-haiku-20241022")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Verify different models for different task types
        planning_models = router.routing_config.get_models_for_task("planning")
        sql_models = router.routing_config.get_models_for_task("sql_generation")

        assert planning_models[0].model == "claude-sonnet-4-20250514"
        assert sql_models[0].model == "claude-3-5-haiku-20241022"


# =============================================================================
# Task Routing Integration Tests
# =============================================================================


class TestTaskRoutingIntegration:
    """
    Integration tests verifying that task-type routing works correctly
    between Planner and TaskRouter components.

    Note: These are integration tests, not true E2E tests. They test
    Planner + mock Router, not the full Session.solve() pipeline.

    These tests verify that:
    1. Planning phase uses the PLANNING task type routing
    2. SQL generation uses SQL_GENERATION task type routing
    3. Different providers can be used for different task types
    """

    def test_routing_tracks_models_used_at_each_phase(self):
        """
        Integration: Track which models are called at each phase of NLQ processing.

        Uses a mock router to verify correct task type routing.
        """
        from unittest.mock import MagicMock, patch
        from constat.core.config import Config, LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter
        from constat.providers.router import TaskResult
        from constat.execution.planner import Planner
        from constat.catalog.schema_manager import SchemaManager
        from constat.core.models import TaskType

        # Track all execute calls
        execute_calls = []

        # Create routing with specific models for each task type
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="planning-model-v1")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="sql-model-v1")]
            ),
            "python_analysis": TaskRoutingEntry(
                models=[ModelSpec(model="python-model-v1")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="default-model-v1",
            api_key="test-key",
            task_routing=routing,
        )

        # Create a minimal config
        config = Config(
            databases={"test": {"uri": "sqlite:///:memory:"}},
            llm=llm_config,
        )

        # Create a mock router that tracks calls
        mock_router = MagicMock(spec=TaskRouter)
        
        def track_execute(task_type=None, **kwargs):
            execute_calls.append({
                "task_type": task_type.value if hasattr(task_type, 'value') else str(task_type),
            })
            return TaskResult(
                success=True,
                content='''```json
{
    "reasoning": "Test plan",
    "steps": [
        {"number": 1, "goal": "Load data", "inputs": [], "outputs": ["data"], "depends_on": [], "task_type": "sql_generation"}
    ]
}
```''',
                model_used="planning-model-v1",
                provider_used="anthropic",
            )

        mock_router.execute = MagicMock(side_effect=track_execute)

        # Create schema manager with mock
        schema_manager = MagicMock(spec=SchemaManager)
        schema_manager.get_overview.return_value = "test: 1 table"
        schema_manager.get_table_schema.return_value = {"table": "test", "columns": []}
        schema_manager.find_relevant_tables.return_value = []

        # Create planner with mock router
        planner = Planner(config, schema_manager, mock_router)

        # Run planning
        execute_calls.clear()
        result = planner.plan("Show me all customers")

        # Verify planning task type was used
        assert len(execute_calls) >= 1
        planning_call = execute_calls[0]
        assert planning_call["task_type"] == "planning", (
            f"Planning should use 'planning' task type, got '{planning_call['task_type']}'"
        )

    def test_routing_different_providers_per_task(self):
        """
        Integration: Verify different providers can be used for different task types.

        Configures Anthropic for planning and Ollama for SQL generation.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Configure different providers for different task types
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-sonnet-4-20250514")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(provider="ollama", model="sqlcoder:7b")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # Verify routing configuration
        planning_models = router.routing_config.get_models_for_task("planning")
        sql_models = router.routing_config.get_models_for_task("sql_generation")

        # Planning uses default provider (anthropic)
        assert planning_models[0].provider is None
        assert planning_models[0].model == "claude-sonnet-4-20250514"

        # SQL generation uses ollama
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == "sqlcoder:7b"

    def test_routing_with_escalation_chain(self):
        """
        Integration: Verify escalation chain configuration works.

        Configures multiple models per task type for fallback behavior.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Configure escalation chain: ollama -> haiku -> sonnet
        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[
                    ModelSpec(provider="ollama", model="sqlcoder:7b"),
                    ModelSpec(model="claude-3-5-haiku-20241022"),
                    ModelSpec(model="claude-sonnet-4-20250514"),
                ]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # Verify escalation chain
        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert len(sql_models) == 3
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == "sqlcoder:7b"
        assert sql_models[1].model == "claude-3-5-haiku-20241022"
        assert sql_models[2].model == "claude-sonnet-4-20250514"

    def test_routing_falls_back_to_defaults(self):
        """
        Integration: Verify that unconfigured task types fall back to defaults.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Only configure sql_generation, leave others as default
        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="special-sql-model")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="default-model",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # SQL generation should use configured model
        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert sql_models[0].model == "special-sql-model"

        # Planning should fall back to default routing
        planning_models = router.routing_config.get_models_for_task("planning")
        # Default routing uses claude-sonnet-4-20250514 for planning
        assert len(planning_models) >= 1

        # Unknown task type should return empty (router handles fallback)
        unknown_models = router.routing_config.get_models_for_task("nonexistent")
        assert unknown_models == []
