"""Tests for LLM provider implementations.

Provider test strategy:
- Anthropic: Tested with real API (ANTHROPIC_API_KEY required)
- Ollama: Tested with local server (requires Ollama running)
- OpenAI: Skipped by default (requires OPENAI_API_KEY)
- Gemini: Skipped by default (requires GOOGLE_API_KEY)
- Grok: Skipped by default (requires XAI_API_KEY)
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
        """Model can be overridden per-call (for tiered model selection)."""
        # Initialize with default sonnet model
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

        # Override to haiku for this specific call (simulating tier selection)
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
    def test_tiered_model_selection_integration(self):
        """Test tiered model selection as used by planner/session."""
        from constat.core.config import LLMConfig, LLMTiersConfig

        # Configure tiers like production would
        tiers = LLMTiersConfig(
            planning="claude-3-5-haiku-20241022",  # Use haiku for planning
            codegen="claude-3-5-haiku-20241022",   # Use haiku for codegen
            simple="claude-3-5-haiku-20241022",    # Use haiku for simple
        )
        llm_config = LLMConfig(
            model="claude-sonnet-4-20250514",  # Default is sonnet
            tiers=tiers,
        )

        # Provider initialized with default model
        provider = AnthropicProvider(model=llm_config.model)
        assert provider.model == "claude-sonnet-4-20250514"

        # But can use tiered models for specific operations
        planning_model = llm_config.get_model("planning")
        assert planning_model == "claude-3-5-haiku-20241022"

        response = provider.generate(
            system="You are a helpful assistant.",
            user_message="What is 1 + 1?",
            model=planning_model,  # Use planning tier
            max_tokens=50,
        )
        assert "2" in response


# =============================================================================
# Ollama Provider Tests
# =============================================================================

class TestOllamaProvider:
    """Tests for Ollama local provider."""

    @requires_ollama
    def test_instantiation(self):
        """Provider can be instantiated."""
        provider = OllamaProvider()
        assert provider.model == "llama3.2"
        assert provider.DEFAULT_BASE_URL == "http://localhost:11434/v1"

    @requires_ollama
    def test_instantiation_with_custom_model(self):
        """Provider accepts custom model."""
        provider = OllamaProvider(model="llama3.1")
        assert provider.model == "llama3.1"

    @requires_ollama
    def test_supports_tools_detection(self):
        """Tool support is detected for known models."""
        provider = OllamaProvider(model="llama3.2")
        # llama3.2 is in the TOOL_CAPABLE_MODELS list
        assert provider.supports_tools is True

        # Older/unknown models may not support tools
        provider_old = OllamaProvider(model="codellama")
        # codellama is not in the list, supports_tools should be False
        assert provider_old.supports_tools is False

    @requires_ollama_model
    def test_generate_simple(self):
        """Basic generation without tools."""
        provider = OllamaProvider(model=OLLAMA_TEST_MODEL)
        response = provider.generate(
            system="You are a helpful assistant. Be very brief.",
            user_message="What is 2 + 2? Reply with just the number.",
            max_tokens=50,
        )
        assert "4" in response

    @requires_ollama_model
    def test_generate_code(self):
        """Code generation extracts from markdown blocks."""
        provider = OllamaProvider(model=OLLAMA_TEST_MODEL)
        response = provider.generate_code(
            system="You are a Python expert. Return only code in markdown blocks.",
            user_message="Write a one-line function that returns the sum of a and b.",
            max_tokens=200,
        )
        assert "def" in response or "lambda" in response
        assert "```" not in response

    @requires_ollama_model
    def test_generate_with_tools(self):
        """Generation with tool calling (if model supports it)."""
        provider = OllamaProvider(model=OLLAMA_TEST_MODEL)
        if not provider.supports_tools:
            pytest.skip("Model does not support tools")

        response = provider.generate(
            system="You have access to tools. Use them when needed.",
            user_message="What's the weather in London?",
            tools=SAMPLE_TOOLS,
            tool_handlers=TOOL_HANDLERS,
            max_tokens=500,
        )
        # Response should mention the location or weather info
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
# Provider Factory Tests
# =============================================================================

class TestProviderFactory:
    """Tests for ProviderFactory and multi-provider tiering."""

    def test_factory_creates_default_provider(self):
        """Factory creates the default provider."""
        from constat.core.config import LLMConfig
        from constat.providers import ProviderFactory

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        factory = ProviderFactory(llm_config)
        provider = factory.get_default_provider()

        assert provider is not None
        assert provider.model == "claude-3-5-haiku-20241022"

    def test_factory_caches_providers(self):
        """Factory caches provider instances."""
        from constat.core.config import LLMConfig
        from constat.providers import ProviderFactory

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
        )
        factory = ProviderFactory(llm_config)

        provider1 = factory.get_default_provider()
        provider2 = factory.get_default_provider()

        assert provider1 is provider2  # Same instance

    def test_factory_tier_without_override_uses_default_provider(self):
        """Tier without provider override uses default provider."""
        from constat.core.config import LLMConfig, LLMTiersConfig
        from constat.providers import ProviderFactory

        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",  # Just model, no provider override
            codegen="claude-sonnet-4-20250514",
            simple="claude-3-5-haiku-20241022",
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )
        factory = ProviderFactory(llm_config)

        provider, model = factory.get_provider_for_tier("planning")
        default_provider = factory.get_default_provider()

        # Should use same provider (anthropic), different model
        assert provider is default_provider
        assert model == "claude-opus-4-20250514"

    def test_factory_tier_with_provider_override(self):
        """Tier with provider override creates different provider."""
        from constat.core.config import LLMConfig, LLMTiersConfig, TierConfig
        from constat.providers import ProviderFactory

        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple=TierConfig(provider="ollama", model="llama3.2:3b"),
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )
        factory = ProviderFactory(llm_config)

        simple_provider, simple_model = factory.get_provider_for_tier("simple")
        default_provider = factory.get_default_provider()

        # Simple tier should use different provider
        assert simple_provider is not default_provider
        assert simple_model == "llama3.2:3b"
        assert simple_provider.__class__.__name__ == "OllamaProvider"

    def test_tier_config_model_only(self):
        """TierConfig with just model uses default provider."""
        from constat.core.config import LLMConfig, LLMTiersConfig
        from constat.providers import ProviderFactory

        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple="claude-3-5-haiku-20241022",
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )

        tier_config = llm_config.get_tier_config("simple")
        assert tier_config.provider is None  # Uses default
        assert tier_config.model == "claude-3-5-haiku-20241022"

    def test_tier_config_with_provider(self):
        """TierConfig with provider override."""
        from constat.core.config import LLMConfig, LLMTiersConfig, TierConfig

        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple=TierConfig(provider="ollama", model="llama3.2:3b"),
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )

        tier_config = llm_config.get_tier_config("simple")
        assert tier_config.provider == "ollama"
        assert tier_config.model == "llama3.2:3b"

    def test_unknown_provider_raises_error(self):
        """Unknown provider name raises ValueError."""
        from constat.core.config import LLMConfig
        from constat.providers import ProviderFactory

        llm_config = LLMConfig(
            provider="unknown_provider",
            model="some-model",
        )
        factory = ProviderFactory(llm_config)

        with pytest.raises(ValueError, match="Unknown provider"):
            factory.get_default_provider()


# =============================================================================
# Multi-Provider Integration Tests
# =============================================================================

class TestMultiProviderIntegration:
    """Integration tests for multi-provider tiering.

    These tests verify that different providers can be used for different tiers.
    """

    @requires_anthropic_key
    @requires_ollama_model
    def test_anthropic_planning_ollama_simple(self):
        """Use Anthropic for planning, Ollama for simple tasks."""
        from constat.core.config import LLMConfig, LLMTiersConfig, TierConfig
        from constat.providers import ProviderFactory

        tiers = LLMTiersConfig(
            planning="claude-3-5-haiku-20241022",  # Anthropic for planning
            codegen="claude-3-5-haiku-20241022",   # Anthropic for codegen
            simple=TierConfig(provider="ollama", model=OLLAMA_TEST_MODEL),  # Ollama for simple
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-3-5-haiku-20241022",
            tiers=tiers,
        )
        factory = ProviderFactory(llm_config)

        # Test planning tier (Anthropic)
        planning_provider, planning_model = factory.get_provider_for_tier("planning")
        planning_response = planning_provider.generate(
            system="Be concise.",
            user_message="What is 5 + 5?",
            model=planning_model,
            max_tokens=50,
        )
        assert "10" in planning_response

        # Test simple tier (Ollama)
        simple_provider, simple_model = factory.get_provider_for_tier("simple")
        simple_response = simple_provider.generate(
            system="Be concise.",
            user_message="What is 3 + 3?",
            model=simple_model,
            max_tokens=50,
        )
        assert "6" in simple_response

        # Verify they're different providers
        assert planning_provider.__class__.__name__ == "AnthropicProvider"
        assert simple_provider.__class__.__name__ == "OllamaProvider"

    @requires_anthropic_key
    def test_all_tiers_same_provider_different_models(self):
        """All tiers use same provider but different models."""
        from constat.core.config import LLMConfig, LLMTiersConfig
        from constat.providers import ProviderFactory

        tiers = LLMTiersConfig(
            planning="claude-3-5-haiku-20241022",
            codegen="claude-3-5-haiku-20241022",
            simple="claude-3-5-haiku-20241022",
        )
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",  # Default is different
            tiers=tiers,
        )
        factory = ProviderFactory(llm_config)

        # All tiers should use the same provider instance
        planning_provider, _ = factory.get_provider_for_tier("planning")
        codegen_provider, _ = factory.get_provider_for_tier("codegen")
        simple_provider, _ = factory.get_provider_for_tier("simple")

        assert planning_provider is codegen_provider
        assert codegen_provider is simple_provider

        # But models are different from default
        _, planning_model = factory.get_provider_for_tier("planning")
        _, default_model = factory.get_provider_for_tier("default")

        assert planning_model == "claude-3-5-haiku-20241022"
        assert default_model == "claude-sonnet-4-20250514"