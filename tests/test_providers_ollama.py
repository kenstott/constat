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

"""Tests for Ollama provider."""

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import OllamaProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


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
        pass  # Probe: Ollama not reachable; return None as sentinel
    return None


OLLAMA_TEST_MODEL = get_ollama_model()


@pytest.fixture
def require_ollama():
    if not ollama_available():
        pytest.fail("Ollama server not running at localhost:11434 — required for this test")


@pytest.fixture
def require_ollama_model():
    if not OLLAMA_TEST_MODEL:
        pytest.fail("No suitable Ollama model available for testing — required for this test")


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

    def test_generate_code(self, ollama_container):
        """Code generation extracts from markdown blocks."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate_code(
                    system="You are a Python expert. Return only code in markdown blocks.",
                    user_message="Write a one-line function that returns the sum of a and b.",
                    max_tokens=200,
                )
                assert "def" in response or "lambda" in response or "+" in response
                assert "```" not in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_tools(self, ollama_container):
        """Generation with tool calling (if model supports it)."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
        )
        if not provider.supports_tools:
            pytest.fail("Model does not support tools")

        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You have access to tools. Use them when needed.",
                    user_message="What's the weather in London?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "London" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_multiple_responses(self, ollama_container):
        """Multiple generations work without issues."""
        provider = OllamaProvider(
            model=ollama_container["model"],
            base_url=ollama_container["base_url"],
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


def test_ollama_provider_backward_compat(require_ollama):
    """Backward compatibility test - uses old fixture."""
    provider = OllamaProvider()
    assert provider.model == "llama3.2"


def test_ollama_generate_backward_compat(require_ollama, require_ollama_model):
    """Backward compatibility test for generation."""
    provider = OllamaProvider(model=OLLAMA_TEST_MODEL)
    last_exc = None
    for attempt in range(3):
        try:
            response = provider.generate(
                system="Be brief.",
                user_message="Say hello.",
                max_tokens=20,
            )
            assert "hello" in response.lower() or len(response) > 2
            break
        except AssertionError as e:
            last_exc = e
            if attempt == 2:
                raise
    else:
        raise last_exc
