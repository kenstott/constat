# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Adapts constat LLM providers into chonk's LLMClient protocol.

Usage:
    llm = build_chonk_llm(chonk_model_spec, llm_config)
    # llm.complete(prompt) -> str

If chonk_model_spec is None, falls back to llm_config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.core.config import LLMConfig
    from constat.core.source_config import ChonkModelSpec


class _ConstatLLMClient:
    """Wraps a constat BaseLLMProvider as a chonk LLMClient."""

    def __init__(self, provider) -> None:
        self._provider = provider

    def complete(self, prompt: str) -> str:
        return self._provider.generate(system="", user_message=prompt, max_tokens=2048)


def build_chonk_llm(
    spec: "ChonkModelSpec | None",
    llm_config: "LLMConfig",
) -> "_ConstatLLMClient | None":
    """Build a chonk LLMClient from a ChonkModelSpec + constat LLMConfig.

    Returns None if no provider can be constructed (missing model/provider).
    """
    from constat.core.config import ModelSpec
    from constat.providers.router import TaskRouter

    try:
        if spec is not None:
            model_spec = ModelSpec(
                model=spec.model,
                provider=spec.provider,
                api_key=spec.api_key,
                base_url=spec.base_url,
                max_tokens=spec.max_tokens,
            )
        else:
            # Fall back to main llm_config provider/model
            if not llm_config.model:
                return None
            model_spec = ModelSpec(
                model=llm_config.model,
                provider=llm_config.provider if llm_config.provider != "anthropic" else None,
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
            )

        router = TaskRouter(llm_config)
        provider = router._get_provider(model_spec)
        return _ConstatLLMClient(provider)
    except Exception:
        return None
