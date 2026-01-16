"""Mistral AI provider for La Plateforme.

Provides access to Mistral's hosted models including:
- Mistral Large (flagship model)
- Mistral Small (efficient, low-latency)
- Codestral (code-specialized)
- Mistral Nemo (open-weight, self-deployable)
- Ministral (edge-optimized)

Uses OpenAI-compatible API.
"""

import os
from typing import Optional

from .openai import OpenAIProvider


class MistralProvider(OpenAIProvider):
    """Mistral AI provider for La Plateforme.

    Mistral provides high-quality European-hosted models with
    an OpenAI-compatible API.
    """

    MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

    # Available models (as of 2025)
    MODELS = {
        # Flagship
        "mistral-large-latest": "Most capable model, strong reasoning",
        "mistral-large-2411": "November 2024 release",
        # Efficient
        "mistral-small-latest": "Fast, cost-effective for simple tasks",
        "mistral-small-2503": "Latest small model",
        # Code
        "codestral-latest": "Optimized for code generation",
        "codestral-2501": "January 2025 code model",
        # Open weight
        "mistral-nemo": "12B open-weight model",
        "ministral-8b-latest": "8B edge-optimized model",
        "ministral-3b-latest": "3B ultra-light model",
        # Legacy
        "mistral-medium": "Deprecated, use mistral-small",
        "mistral-tiny": "Deprecated, use ministral",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-large-latest",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Mistral provider.

        Args:
            api_key: Mistral API key (or uses MISTRAL_API_KEY env var)
            model: Model to use. Options:
                - mistral-large-latest (flagship, best quality)
                - mistral-small-latest (fast, cost-effective)
                - codestral-latest (code generation)
                - mistral-nemo (open-weight 12B)
                - ministral-8b-latest (edge 8B)
                - ministral-3b-latest (edge 3B)
            base_url: Custom base URL (default: https://api.mistral.ai/v1)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Mistral provider requires the openai package. "
                "Install with: pip install openai"
            )

        resolved_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )

        url = base_url or self.MISTRAL_BASE_URL

        self.client = OpenAI(base_url=url, api_key=resolved_key)
        self.model = model

    @classmethod
    def list_models(cls) -> dict[str, str]:
        """List available Mistral models with descriptions."""
        return cls.MODELS.copy()


class CodestralProvider(MistralProvider):
    """Convenience provider for Codestral (code-specialized model).

    Codestral is optimized for code generation, completion, and explanation.
    Supports 250+ programming languages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "codestral-latest",
    ):
        """
        Initialize Codestral provider.

        Args:
            api_key: Mistral API key (or uses MISTRAL_API_KEY env var)
            model: Codestral model variant (default: codestral-latest)
        """
        super().__init__(api_key=api_key, model=model)