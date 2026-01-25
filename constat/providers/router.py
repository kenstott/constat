# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Task-based routing with automatic escalation.

Routes tasks to appropriate models with automatic fallback to more capable
models on failure. Supports local-first with cloud fallback pattern.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from constat.core.config import LLMConfig, ModelSpec, TaskRoutingConfig
from constat.core.models import TaskType
from constat.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class EscalationEvent:
    """Record of an escalation from one model to another."""
    task_type: str
    from_model: str
    to_model: str
    reason: str  # Error message or timeout
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskResult:
    """Result of a routed task execution."""
    success: bool
    content: str
    model_used: str
    provider_used: str
    escalations: list[EscalationEvent] = field(default_factory=list)
    attempts: int = 1
    total_time_ms: int = 0


class TaskRouter:
    """Routes tasks to appropriate models with automatic escalation.

    The router:
    1. Looks up the model chain for the task type
    2. Tries each model in order
    3. On failure, escalates to next model in chain
    4. Records escalation metrics

    Usage:
        router = TaskRouter(llm_config)

        result = router.execute(
            task_type=TaskType.SQL_GENERATION,
            system="Generate SQL...",
            user_message="Get top 5 customers",
        )

        if result.escalations:
            logger.warning(f"Task escalated: {result.escalations}")
    """

    # Map of provider names to their classes
    PROVIDER_CLASSES = {
        "anthropic": "constat.providers.anthropic.AnthropicProvider",
        "openai": "constat.providers.openai.OpenAIProvider",
        "gemini": "constat.providers.gemini.GeminiProvider",
        "grok": "constat.providers.grok.GrokProvider",
        "mistral": "constat.providers.mistral.MistralProvider",
        "codestral": "constat.providers.mistral.CodestralProvider",
        "ollama": "constat.providers.llama.OllamaProvider",
        "llama": "constat.providers.llama.LlamaProvider",
        "together": "constat.providers.llama.TogetherProvider",
        "groq": "constat.providers.llama.GroqProvider",
    }

    def __init__(self, llm_config: LLMConfig):
        """
        Initialize the router with LLM configuration.

        Args:
            llm_config: LLM configuration with task routing
        """
        self.llm_config = llm_config
        self.routing_config = llm_config.get_task_routing()
        self._provider_cache: dict[str, BaseLLMProvider] = {}

        # Escalation history for observability
        self._escalation_history: list[EscalationEvent] = []

        # Escalation callback
        self._on_escalation: Optional[Callable[[EscalationEvent], None]] = None

    def on_escalation(self, callback: Callable[[EscalationEvent], None]) -> None:
        """Register callback for escalation events."""
        self._on_escalation = callback

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

    def _get_provider(self, spec: ModelSpec) -> BaseLLMProvider:
        """Get or create a provider for a model spec."""
        # Determine provider name
        provider_name = spec.provider or self.llm_config.provider
        base_url = spec.base_url or self.llm_config.base_url

        cache_key = self._get_cache_key(provider_name, base_url)

        if cache_key not in self._provider_cache:
            provider_class = self._get_provider_class(provider_name)

            # Build kwargs
            kwargs = {"model": spec.model}

            # Add API key if provider likely needs it
            if self.llm_config.api_key and provider_name not in ("ollama", "llama"):
                kwargs["api_key"] = self.llm_config.api_key

            # Add base_url if provided
            if base_url:
                kwargs["base_url"] = base_url

            self._provider_cache[cache_key] = provider_class(**kwargs)

        return self._provider_cache[cache_key]

    def execute(
        self,
        task_type: TaskType,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 4096,
        complexity: str = "medium",
    ) -> TaskResult:
        """
        Execute a task with automatic model escalation.

        Args:
            task_type: The type of task for routing
            system: System prompt
            user_message: User's request
            tools: Optional tool definitions
            tool_handlers: Optional tool handler functions
            max_tokens: Max tokens to generate
            complexity: Complexity hint (low, medium, high)

        Returns:
            TaskResult with content and escalation info
        """
        start_time = time.time()
        models = self.routing_config.get_models_for_task(
            task_type.value,
            complexity
        )

        # If no models configured for this task type, use general fallback
        if not models:
            models = self.routing_config.get_models_for_task("general", complexity)

        # If still no models, use default from llm_config
        if not models:
            models = [ModelSpec(model=self.llm_config.model)]

        escalations = []
        last_error = None

        for i, spec in enumerate(models):
            provider_name = spec.provider or self.llm_config.provider
            try:
                provider = self._get_provider(spec)

                content = provider.generate(
                    system=system,
                    user_message=user_message,
                    tools=tools,
                    tool_handlers=tool_handlers,
                    max_tokens=max_tokens,
                    model=spec.model,
                )

                elapsed_ms = int((time.time() - start_time) * 1000)

                # Log prompt for analysis (fire-and-forget, don't block on errors)
                try:
                    from constat.providers.prompt_logger import log_prompt
                    log_prompt(
                        task_type=task_type.value,
                        model=spec.model,
                        provider=provider_name,
                        system_prompt=system,
                        user_message=user_message,
                        response_time_ms=elapsed_ms,
                        success=True,
                    )
                except Exception:
                    pass  # Don't fail task execution due to logging

                return TaskResult(
                    success=True,
                    content=content,
                    model_used=spec.model,
                    provider_used=provider_name,
                    escalations=escalations,
                    attempts=i + 1,
                    total_time_ms=elapsed_ms,
                )

            except Exception as e:
                last_error = str(e)

                # Record escalation if there's a next model to try
                if i < len(models) - 1:
                    next_spec = models[i + 1]
                    next_provider = next_spec.provider or self.llm_config.provider
                    event = EscalationEvent(
                        task_type=task_type.value,
                        from_model=f"{provider_name}/{spec.model}",
                        to_model=f"{next_provider}/{next_spec.model}",
                        reason=last_error,
                    )
                    escalations.append(event)
                    self._escalation_history.append(event)

                    if self._on_escalation:
                        self._on_escalation(event)

                    logger.info(
                        f"Escalating {task_type.value} from "
                        f"{provider_name}/{spec.model} to "
                        f"{next_provider}/{next_spec.model}: {last_error}"
                    )

        # All models failed
        elapsed_ms = int((time.time() - start_time) * 1000)
        return TaskResult(
            success=False,
            content=f"All models failed. Last error: {last_error}",
            model_used=models[-1].model if models else "none",
            provider_used=(models[-1].provider or self.llm_config.provider) if models else "none",
            escalations=escalations,
            attempts=len(models),
            total_time_ms=elapsed_ms,
        )

    def execute_code(
        self,
        task_type: TaskType,
        system: str,
        user_message: str,
        tools: Optional[list[dict]] = None,
        tool_handlers: Optional[dict[str, Callable]] = None,
        max_tokens: int = 12288,
        complexity: str = "medium",
    ) -> TaskResult:
        """Execute and extract code from response."""
        result = self.execute(
            task_type=task_type,
            system=system,
            user_message=user_message,
            tools=tools,
            tool_handlers=tool_handlers,
            max_tokens=max_tokens,
            complexity=complexity,
        )

        if result.success:
            # Extract code from markdown
            result.content = self._extract_code(result.content)

        return result

    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code blocks.

        Handles various cases:
        - Complete markdown blocks: ```python ... ```
        - Incomplete blocks (no closing fence)
        - Nested or multiple code blocks
        """
        text = text.strip()

        # Case 1: Complete markdown code block
        pattern = r"```(?:python|sql)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Case 2: Opening fence without closing (truncated response)
        # Matches: ```python\n... or ```\n...
        if text.startswith("```"):
            lines = text.split("\n", 1)
            if len(lines) > 1:
                # Skip the first line (```python or ```)
                code = lines[1]
                # Remove trailing ``` if present
                if code.rstrip().endswith("```"):
                    code = code.rstrip()[:-3]
                return code.strip()

        return text

    def get_escalation_stats(self) -> dict[str, Any]:
        """Get escalation statistics for observability."""
        by_task_type: dict[str, dict] = {}
        for event in self._escalation_history:
            if event.task_type not in by_task_type:
                by_task_type[event.task_type] = {
                    "count": 0,
                    "from_models": {},
                }
            by_task_type[event.task_type]["count"] += 1
            from_model = event.from_model
            if from_model not in by_task_type[event.task_type]["from_models"]:
                by_task_type[event.task_type]["from_models"][from_model] = 0
            by_task_type[event.task_type]["from_models"][from_model] += 1

        return {
            "total_escalations": len(self._escalation_history),
            "by_task_type": by_task_type,
        }

    def clear_stats(self) -> None:
        """Clear escalation history."""
        self._escalation_history.clear()

    def clear_cache(self) -> None:
        """Clear the provider cache."""
        self._provider_cache.clear()

    def generate(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        """Simple text generation without task routing.

        This is a convenience method for simple generation tasks that don't
        need the full execute() machinery with task types and escalation.

        Args:
            system: System prompt
            user_message: User's request
            max_tokens: Max tokens to generate

        Returns:
            Generated text content
        """
        result = self.execute(
            task_type=TaskType.SUMMARIZATION,  # Use summarization as generic task
            system=system,
            user_message=user_message,
            max_tokens=max_tokens,
            complexity="low",
        )
        return result.content if result.success else ""
