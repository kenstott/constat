"""LLM provider implementations.

Available providers:
- AnthropicProvider: Claude models (Claude 3, Claude 3.5, etc.)
- OpenAIProvider: GPT models (GPT-4, GPT-4o, etc.)
- GeminiProvider: Google Gemini models
- GrokProvider: xAI Grok models
- OllamaProvider: Local Llama models via Ollama
- TogetherProvider: Hosted Llama models via Together AI
- GroqProvider: Fast Llama inference via Groq
- LlamaProvider: Alias for OllamaProvider (default local option)

Routing:
- TaskRouter: Routes tasks to models with automatic escalation
"""

from .base import BaseLLMProvider, GenerationResult, ToolResult
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .grok import GrokProvider
from .llama import LlamaProvider, OllamaProvider, TogetherProvider, GroqProvider
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
    "LlamaProvider",
    "OllamaProvider",
    "TogetherProvider",
    "GroqProvider",
]
