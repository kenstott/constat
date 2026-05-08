# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""LLM provider implementations.

Available providers:
- AnthropicProvider: Claude models (Claude 3, Claude 3.5, etc.)
- OpenAIProvider: GPT models (GPT-4, GPT-4o, etc.)
- GeminiProvider: Google Gemini models
- GrokProvider: xAI Grok models
- MistralProvider: Mistral AI models (Mistral Large, Small, Nemo)
- CodestralProvider: Mistral's code-specialized model
- OllamaProvider: Local Llama/Mistral models via Ollama
- TogetherProvider: Hosted Llama models via Together AI
- GroqProvider: Fast Llama inference via Groq
- LlamaProvider: Alias for OllamaProvider (default local option)

Routing:
- TaskRouter: Routes tasks to models with automatic escalation
"""

from .anthropic import AnthropicProvider
from .base import BaseLLMProvider, GenerationResult, ToolResult
from .gemini import GeminiProvider
from .grok import GrokProvider
from .llama import LlamaProvider, OllamaProvider, TogetherProvider, GroqProvider
from .mistral import MistralProvider, CodestralProvider
from .openai import OpenAIProvider
from .router import TaskRouter

__all__ = [
    # Base
    "BaseLLMProvider",
    "GenerationResult",
    "ToolResult",
    # Router
    "TaskRouter",
    # Providers
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "GrokProvider",
    "MistralProvider",
    "CodestralProvider",
    "LlamaProvider",
    "OllamaProvider",
    "TogetherProvider",
    "GroqProvider",
]
