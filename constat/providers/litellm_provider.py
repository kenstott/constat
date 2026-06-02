# Copyright (c) 2025 Kenneth Stott
# Canary: 4c8c5451-700c-4342-ae17-ba388e7c15b3
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Universal LLM provider backed by LiteLLM.

Supports all providers (Anthropic, OpenAI, Gemini, Ollama, Groq, Mistral,
Together, xAI/Grok, Codestral) through a single unified interface.

Tools are accepted in Anthropic format (input_schema) and converted internally
to the OpenAI format that LiteLLM uses as its common wire format.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Callable, Optional

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Map constat provider names → LiteLLM model-string prefixes
_PROVIDER_PREFIX: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "grok": "xai",          # xAI / Grok
    "mistral": "mistral",
    "codestral": "codestral",
    "ollama": "ollama",
    "llama": "ollama",      # local Ollama alias
    "together": "together_ai",
    "groq": "groq",
}

# Env-var → (constat provider name, default model)
_ENV_PROVIDERS: list[tuple[str, str, str]] = [
    ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-6"),
    ("OPENAI_API_KEY", "openai", "gpt-4o"),
    ("GOOGLE_API_KEY", "gemini", "gemini-2.0-flash"),
    ("XAI_API_KEY", "grok", "grok-3-mini-fast"),
    ("MISTRAL_API_KEY", "mistral", "mistral-large-latest"),
    ("TOGETHER_API_KEY", "together", "meta-llama/Llama-3-70b-chat-hf"),
    ("GROQ_API_KEY", "groq", "llama3-70b-8192"),
    ("OLLAMA_HOST", "ollama", "llama3"),
]


def _litellm_model(provider: str, model: str) -> str:
    """Compose a LiteLLM model string from provider + model name."""
    prefix = _PROVIDER_PREFIX.get(provider.lower(), provider.lower())
    # Avoid double-prefix if model already contains the prefix
    if model.startswith(f"{prefix}/"):
        return model
    return f"{prefix}/{model}"


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Convert tools from Anthropic format (input_schema) to OpenAI format (parameters)."""
    out = []
    for t in tools:
        if "function" in t:
            # Already in OpenAI format
            out.append(t)
        else:
            out.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            })
    return out


class LiteLLMProvider(BaseLLMProvider):
    """Universal LLM provider backed by LiteLLM.

    Accepts the same interface as other constat providers but routes calls
    through LiteLLM, giving access to 100+ models without per-SDK boilerplate.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError(
                "LiteLLMProvider requires litellm. Install with: pip install litellm"
            )
        self._provider = provider.lower()
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout or 120.0

        # Set API key in env if provided (LiteLLM reads from env)
        if api_key:
            env_key = _provider_env_key(self._provider)
            if env_key and not os.environ.get(env_key):
                os.environ[env_key] = api_key

    @property
    def max_output_tokens(self) -> int:
        return 16384

    def _call_kwargs(
        self,
        model_override: Optional[str] = None,
        timeout_override: Optional[float] = None,
    ) -> dict:
        model_str = _litellm_model(
            self._provider,
            model_override or self.model,
        )
        kwargs: dict = {
            "model": model_str,
            "timeout": timeout_override or self._timeout,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return kwargs

    def generate(
        self,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> str:
        import litellm

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        tool_handlers = tool_handlers or {}
        openai_tools = _to_openai_tools(tools) if tools else None
        kwargs = self._call_kwargs(model, timeout)
        kwargs["max_tokens"] = max_tokens
        if openai_tools:
            kwargs["tools"] = openai_tools

        while True:
            response = litellm.completion(messages=messages, **kwargs)
            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls" and choice.message.tool_calls:
                # Append assistant message with tool calls
                messages.append(choice.message.model_dump())

                tool_results = []
                for tc in choice.message.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    handler = tool_handlers.get(name)
                    if handler:
                        try:
                            result = handler(**args)
                            content = str(result)
                        except Exception as exc:
                            content = f"Error: {exc}"
                    else:
                        content = f"Unknown tool: {name}"

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    })

                messages.extend(tool_results)
            else:
                return choice.message.content or ""

    def generate_vision(
        self,
        system: str,
        image_bytes: bytes,
        mime_type: str,
        text_prompt: str,
        max_tokens: int = 1024,
        model: Optional[str] = None,
    ) -> str:
        import litellm

        image_b64 = base64.standard_b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime_type};base64,{image_b64}"

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": text_prompt},
            ],
        })

        kwargs = self._call_kwargs(model)
        kwargs["max_tokens"] = max_tokens
        response = litellm.completion(messages=messages, **kwargs)
        return response.choices[0].message.content or ""


def _provider_env_key(provider: str) -> Optional[str]:
    """Return the canonical env-var name for a provider's API key."""
    _map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "grok": "XAI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "codestral": "CODESTRAL_API_KEY",
        "together": "TOGETHER_API_KEY",
        "together_ai": "TOGETHER_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    return _map.get(provider)


def auto_detect_provider() -> LiteLLMProvider:
    """Create a LiteLLMProvider from the first matching API key env var."""
    for env_var, provider, default_model in _ENV_PROVIDERS:
        val = os.environ.get(env_var)
        if val:
            return LiteLLMProvider(
                provider=provider,
                model=default_model,
                api_key=val if provider != "ollama" else None,
            )
    raise RuntimeError(
        "No LLM API key found. Set one of: "
        + ", ".join(e for e, _, _ in _ENV_PROVIDERS)
    )
