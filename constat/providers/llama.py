# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Llama model providers with tool support.

Supports multiple Llama deployment options:
- Ollama (local deployment)
- Together AI (hosted Llama models)
- Groq (fast inference)

All use OpenAI-compatible APIs.
"""

from typing import Optional

from .openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """Ollama provider for running Llama models locally.

    Ollama provides an OpenAI-compatible API at localhost:11434.
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    # Models known to support tool calling
    TOOL_CAPABLE_MODELS = {
        "llama3.2", "llama3.1", "llama3", "llama3.3",
        "mistral", "mixtral", "mistral-nemo",
        "qwen2.5", "qwen2",
        "command-r",
    }

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Ollama provider.

        Args:
            model: Model to use (e.g., "llama3.2", "llama3.1", "codellama")
            base_url: Custom Ollama server URL (default: http://localhost:11434/v1)
        """
        url = base_url or self.DEFAULT_BASE_URL
        self._base_url = url.rstrip("/v1").rstrip("/")  # Store base for API calls

        # Ollama doesn't require an API key
        super().__init__(api_key="ollama", model=model, base_url=url)
        self._supports_tools_cache: Optional[bool] = None

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tool calling.

        Uses a known list of tool-capable model families, with runtime
        verification via the Ollama API when possible.
        """
        if self._supports_tools_cache is not None:
            return self._supports_tools_cache

        # Check against known tool-capable model families
        model_base = self.model.split(":")[0].lower()
        for capable in self.TOOL_CAPABLE_MODELS:
            if model_base.startswith(capable):
                self._supports_tools_cache = True
                return True

        # Try to verify via Ollama API
        try:
            import httpx
            response = httpx.post(
                f"{self._base_url}/api/show",
                json={"name": self.model},
                timeout=5.0,
            )
            if response.status_code == 200:
                data = response.json()
                # Check model info for tool support indicators
                template = data.get("template", "")
                if "tools" in template.lower() or "function" in template.lower():
                    self._supports_tools_cache = True
                    return True
        except Exception:
            pass  # Fall through to default

        self._supports_tools_cache = False
        return False


class TogetherProvider(OpenAIProvider):
    """Together AI provider for hosted Llama models.

    Together AI provides fast inference for open-source models including
    Llama 3, Code Llama, and others.
    """

    TOGETHER_BASE_URL = "https://api.together.xyz/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    ):
        """
        Initialize Together AI provider.

        Args:
            api_key: Together AI API key (or uses TOGETHER_API_KEY env var)
            model: Model to use. Popular options:
                - meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
                - meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
                - codellama/CodeLlama-70b-Instruct-hf
        """
        import os

        resolved_key = api_key or os.environ.get("TOGETHER_API_KEY")
        super().__init__(api_key=resolved_key, model=model, base_url=self.TOGETHER_BASE_URL)


class GroqProvider(OpenAIProvider):
    """Groq provider for fast Llama inference.

    Groq provides extremely fast inference using custom LPU hardware.
    Note: Different from xAI's Grok model.
    """

    GROQ_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize Groq provider.

        Args:
            api_key: Groq API key (or uses GROQ_API_KEY env var)
            model: Model to use. Popular options:
                - llama-3.3-70b-versatile
                - llama-3.1-70b-versatile
                - llama-3.1-8b-instant
                - llama3-70b-8192
        """
        import os

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        super().__init__(api_key=resolved_key, model=model, base_url=self.GROQ_BASE_URL)


# Convenience alias - default to Ollama for local use
LlamaProvider = OllamaProvider
