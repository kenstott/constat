"""Provider factory for instantiating LLM providers by name.

This module provides a factory pattern for creating LLM providers,
supporting multi-provider configurations where different tiers can
use different providers.
"""

from typing import Optional

from constat.core.config import LLMConfig, TierConfig
from .base import BaseLLMProvider


class ProviderFactory:
    """Factory for creating and caching LLM provider instances.

    The factory:
    1. Creates providers by name (anthropic, openai, ollama, etc.)
    2. Caches instances to avoid re-creating them
    3. Handles provider-specific configuration options

    Usage:
        factory = ProviderFactory(llm_config)

        # Get provider for a specific tier
        provider, model = factory.get_provider_for_tier("planning")
        response = provider.generate(..., model=model)

        # Or get the default provider
        provider = factory.get_default_provider()
    """

    # Map of provider names to their classes
    PROVIDER_CLASSES = {
        "anthropic": "constat.providers.anthropic.AnthropicProvider",
        "openai": "constat.providers.openai.OpenAIProvider",
        "gemini": "constat.providers.gemini.GeminiProvider",
        "grok": "constat.providers.grok.GrokProvider",
        "ollama": "constat.providers.llama.OllamaProvider",
        "llama": "constat.providers.llama.LlamaProvider",
        "together": "constat.providers.llama.TogetherProvider",
        "groq": "constat.providers.llama.GroqProvider",
    }

    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the factory with LLM configuration.

        Args:
            llm_config: LLM configuration from the main config
        """
        self.llm_config = llm_config
        self._provider_cache: dict[str, BaseLLMProvider] = {}

    def _get_provider_class(self, provider_name: str) -> type:
        """Get the provider class by name."""
        class_path = self.PROVIDER_CLASSES.get(provider_name.lower())
        if not class_path:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {list(self.PROVIDER_CLASSES.keys())}"
            )

        # Import and return the class
        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _create_provider(
        self,
        provider_name: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> BaseLLMProvider:
        """
        Create a provider instance.

        Args:
            provider_name: Name of the provider (anthropic, openai, etc.)
            model: Default model for the provider
            api_key: API key (optional, providers may use env vars)
            base_url: Custom base URL (for Ollama or custom endpoints)

        Returns:
            Configured provider instance
        """
        provider_class = self._get_provider_class(provider_name)

        # Build kwargs based on what the provider accepts
        kwargs = {"model": model}

        # Add API key if provided (and provider likely needs it)
        if api_key and provider_name not in ("ollama", "llama"):
            kwargs["api_key"] = api_key

        # Add base_url if provided (mainly for Ollama)
        if base_url:
            kwargs["base_url"] = base_url

        return provider_class(**kwargs)

    def _get_cache_key(
        self,
        provider_name: str,
        base_url: Optional[str] = None,
    ) -> str:
        """Generate cache key for a provider configuration."""
        key = provider_name.lower()
        if base_url:
            key += f":{base_url}"
        return key

    def get_default_provider(self) -> BaseLLMProvider:
        """
        Get the default provider (as configured in llm_config).

        Returns:
            The default provider instance
        """
        cache_key = self._get_cache_key(
            self.llm_config.provider,
            self.llm_config.base_url,
        )

        if cache_key not in self._provider_cache:
            self._provider_cache[cache_key] = self._create_provider(
                provider_name=self.llm_config.provider,
                model=self.llm_config.model,
                api_key=self.llm_config.api_key,
                base_url=self.llm_config.base_url,
            )

        return self._provider_cache[cache_key]

    def get_provider_for_tier(
        self,
        tier: str,
    ) -> tuple[BaseLLMProvider, str]:
        """
        Get the provider and model for a specific tier.

        If the tier specifies a provider override, returns that provider.
        Otherwise returns the default provider.

        Args:
            tier: Tier name ("planning", "codegen", "simple", or "default")

        Returns:
            Tuple of (provider, model_name)
        """
        tier_config = self.llm_config.get_tier_config(tier)

        # Determine which provider to use
        if tier_config.provider:
            # Tier overrides the provider
            provider_name = tier_config.provider
            base_url = tier_config.base_url
            api_key = self.llm_config.api_key  # Use default API key
        else:
            # Use default provider
            provider_name = self.llm_config.provider
            base_url = tier_config.base_url or self.llm_config.base_url
            api_key = self.llm_config.api_key

        cache_key = self._get_cache_key(provider_name, base_url)

        if cache_key not in self._provider_cache:
            self._provider_cache[cache_key] = self._create_provider(
                provider_name=provider_name,
                model=tier_config.model,
                api_key=api_key,
                base_url=base_url,
            )

        return self._provider_cache[cache_key], tier_config.model

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._provider_cache.clear()
