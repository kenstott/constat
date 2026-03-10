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
    # Index of the model used within the chain (for skip_models alignment)
    model_index: int = 0


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

        # Domain-aware routing: domain_path → TaskRoutingConfig
        # Checked before system routing, walking up the domain hierarchy
        self._domain_routing: dict[str, TaskRoutingConfig] = {}

        # User-level routing override (checked after domain, before system)
        self._user_routing: Optional[TaskRoutingConfig] = None

        # Escalation history for observability
        self._escalation_history: list[EscalationEvent] = []

        # Escalation callback
        self._on_escalation: Optional[Callable[[EscalationEvent], None]] = None

    def set_domain_routing(self, domain_path: str, config: TaskRoutingConfig) -> None:
        """Set task routing for a specific domain.

        Args:
            domain_path: Dot-delimited domain path (e.g., "sales.north-america")
            config: TaskRoutingConfig for this domain
        """
        self._domain_routing[domain_path] = config

    def set_user_routing(self, config: TaskRoutingConfig) -> None:
        """Set user-level task routing override."""
        self._user_routing = config

    def _resolve_models_for_domain(
        self, task_type: str, complexity: str, domain: Optional[str]
    ) -> list[ModelSpec]:
        """Resolve model chain by walking domain hierarchy → user → system.

        Escalation order:
            1. Exact domain match
            2. Walk up domain hierarchy (trim rightmost path segment)
            3. User-level routing
            4. System-level routing (self.routing_config)

        First tier with a chain for this task_type wins.
        Within that chain, the existing escalation (try each model) applies.
        """
        # Walk domain hierarchy
        if domain:
            parts = domain.split(".")
            for i in range(len(parts), 0, -1):
                ancestor = ".".join(parts[:i])
                routing = self._domain_routing.get(ancestor)
                if routing:
                    models = routing.get_models_for_task(task_type, complexity)
                    if models:
                        return models

        # User-level routing
        if self._user_routing:
            models = self._user_routing.get_models_for_task(task_type, complexity)
            if models:
                return models

        # System-level routing (existing behavior)
        return self.routing_config.get_models_for_task(task_type, complexity)

    def resolve_model_family(
        self,
        task_type: TaskType,
        complexity: str = "medium",
        domain: Optional[str] = None,
        skip_models: int = 0,
        model_override: Optional[str] = None,
    ) -> str:
        """Resolve the provider family for the first model that would handle this task.

        Returns the provider name (e.g., 'anthropic', 'ollama', 'openai').
        """
        models = self._resolve_models_for_domain(task_type.value, complexity, domain)
        if not models:
            fallback_map = {"user_input": "python_analysis"}
            fallback_type = fallback_map.get(task_type.value)
            if fallback_type:
                models = self._resolve_models_for_domain(fallback_type, complexity, domain)
        if not models:
            models = self.routing_config.get_models_for_task("general", complexity)
        if not models:
            models = [ModelSpec(model=self.llm_config.model)]
        if model_override:
            models = [ModelSpec(model=model_override)] + models
        if skip_models > 0 and len(models) > 1:
            models = models[min(skip_models, len(models) - 1):]
        return (models[0].provider or self.llm_config.provider).lower()

    def get_routing_layers(
        self, active_domains: Optional[list[str]] = None
    ) -> dict[str, dict[str, list[ModelSpec]]]:
        """Return routing layers: system defaults, user overrides, and per-domain overrides.

        Returns:
            dict with keys "system", optionally "user", and domain paths.
            Each value maps task_type → list of ModelSpec.
            "system" includes all task types; other layers include only overrides.
        """
        layers: dict[str, dict[str, list[ModelSpec]]] = {}

        # System layer — full routing (defaults merged with config)
        system: dict[str, list[ModelSpec]] = {}
        for task_type, entry in self.routing_config.routes.items():
            system[task_type] = entry.models
        layers["system"] = system

        # User layer — overrides only
        if self._user_routing and self._user_routing.routes:
            user: dict[str, list[ModelSpec]] = {}
            for task_type, entry in self._user_routing.routes.items():
                user[task_type] = entry.models
            layers["user"] = user

        # Domain layers — only show active domains, overrides only
        domains_to_show = set(active_domains or [])
        for domain_path, config in self._domain_routing.items():
            if not domains_to_show or domain_path in domains_to_show:
                domain_routes: dict[str, list[ModelSpec]] = {}
                for task_type, entry in config.routes.items():
                    domain_routes[task_type] = entry.models
                if domain_routes:
                    layers[domain_path] = domain_routes

        return layers

    def on_escalation(self, callback: Callable[[EscalationEvent], None]) -> None:
        """Register callback for escalation events."""
        self._on_escalation = callback

    @property
    def max_output_tokens(self) -> int:
        """Get max output tokens from the default provider."""
        # Get default provider spec
        spec = ModelSpec(model=self.llm_config.model)
        provider = self._get_provider(spec)
        return provider.max_output_tokens

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

    @staticmethod
    def _get_cache_key(
        provider_name: str,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """Generate cache key for a provider configuration."""
        key = provider_name.lower()
        if base_url:
            key += f":{base_url}"
        if timeout:
            key += f":t{timeout}"
        return key

    def _get_provider(self, spec: ModelSpec) -> BaseLLMProvider:
        """Get or create a provider for a model spec."""
        # Determine provider name
        provider_name = spec.provider or self.llm_config.provider
        base_url = spec.base_url or self.llm_config.base_url

        cache_key = self._get_cache_key(provider_name, base_url, spec.timeout_seconds)

        if cache_key not in self._provider_cache:
            provider_class = self._get_provider_class(provider_name)

            # Build kwargs
            kwargs = {"model": spec.model}

            # API key resolution: spec.api_key → global key (if same provider) → provider env var
            if spec.api_key:
                kwargs["api_key"] = spec.api_key
            elif self.llm_config.api_key and provider_name == (self.llm_config.provider or "").lower():
                kwargs["api_key"] = self.llm_config.api_key

            # Add base_url if provided
            if base_url:
                kwargs["base_url"] = base_url

            # Set client-level timeout from model spec
            if spec.timeout_seconds:
                kwargs["timeout"] = float(spec.timeout_seconds)

            logger.info(f"Creating provider {provider_name}/{spec.model} (api_key={'set' if 'api_key' in kwargs else 'from env'}, timeout={spec.timeout_seconds or 120}s)")
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
        domain: Optional[str] = None,
        skip_models: int = 0,
        model_override: Optional[str] = None,
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
            domain: Optional domain path for domain-aware routing.
                    Walks domain hierarchy → user → system to find model chain.
            model_override: Optional model to prepend to the chain (agent override).

        Returns:
            TaskResult with content and escalation info
        """
        start_time = time.time()

        effective_domain = domain

        models = self._resolve_models_for_domain(
            task_type.value,
            complexity,
            effective_domain,
        )

        # If no models configured for this task type, try related task types
        # before falling back to the generic "general" chain.
        if not models:
            # user_input steps generate simple Python — use python_analysis chain
            fallback_map = {"user_input": "python_analysis"}
            fallback_type = fallback_map.get(task_type.value)
            if fallback_type:
                models = self._resolve_models_for_domain(fallback_type, complexity, effective_domain)

        if not models:
            models = self.routing_config.get_models_for_task("general", complexity)

        # If still no models, use default from llm_config
        if not models:
            models = [ModelSpec(model=self.llm_config.model)]

        # Prepend agent model override (tried first, falls back to normal chain)
        if model_override:
            models = [ModelSpec(model=model_override)] + models

        # Skip leading models (used for runtime-error escalation)
        # Clamp to keep at least the last model in the chain
        if skip_models > 0 and len(models) > 1:
            effective_skip = min(skip_models, len(models) - 1)
            skipped = models[:effective_skip]
            models = models[effective_skip:]
            logger.info(
                f"[ESCALATION] Skipping {len(skipped)} model(s) for {task_type.value} "
                f"due to runtime errors: {[f'{(s.provider or self.llm_config.provider)}/{s.model}' for s in skipped]}"
            )

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
                    timeout=float(spec.timeout_seconds) if spec.timeout_seconds else None,
                )

                if content is None:
                    raise ValueError("Provider returned empty response")

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
                except (ImportError, OSError, ValueError):
                    pass  # Don't fail task execution due to logging

                return TaskResult(
                    success=True,
                    content=content,
                    model_used=spec.model,
                    provider_used=provider_name,
                    escalations=escalations,
                    attempts=i + 1,
                    total_time_ms=elapsed_ms,
                    model_index=skip_models + i,
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
        domain: Optional[str] = None,
        skip_models: int = 0,
        model_override: Optional[str] = None,
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
            domain=domain,
            skip_models=skip_models,
            model_override=model_override,
        )

        if result.success:
            # Extract code from markdown
            result.content = self._extract_code(result.content)

        return result

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract code from Markdown code blocks.

        Handles various cases:
        - Complete markdown blocks: ```python ... ```
        - Incomplete blocks (no closing fence)
        - Nested or multiple code blocks
        """
        text = text.strip()

        # Case 1: Complete Markdown code block
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

    def set_domain_models(self, models: list) -> None:
        """Inject fine-tuned models into domain-aware routing.

        For each model with status='ready', prepends to the appropriate
        domain's routing chain (or system routing if no domain).
        """
        for ft_model in models:
            if ft_model.status != "ready" or not ft_model.fine_tuned_model_id:
                continue
            spec = ModelSpec(
                provider=ft_model.provider,
                model=ft_model.fine_tuned_model_id,
            )
            domain = getattr(ft_model, "domain", None)
            if domain:
                # Inject into domain-specific routing
                if domain not in self._domain_routing:
                    self._domain_routing[domain] = TaskRoutingConfig(routes={})
                for task_type in ft_model.task_types:
                    self._domain_routing[domain].prepend_model(task_type, spec)
                    logger.info(
                        f"Prepended fine-tuned model {ft_model.name} "
                        f"({ft_model.fine_tuned_model_id}) to {domain}/{task_type}"
                    )
            else:
                # No domain — prepend to system routing
                for task_type in ft_model.task_types:
                    self.routing_config.prepend_model(task_type, spec)
                    logger.info(
                        f"Prepended fine-tuned model {ft_model.name} "
                        f"({ft_model.fine_tuned_model_id}) to system/{task_type}"
                    )

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

        Raises:
            RuntimeError: If the LLM call fails
        """
        result = self.execute(
            task_type=TaskType.SUMMARIZATION,  # Use summarization as generic task
            system=system,
            user_message=user_message,
            max_tokens=max_tokens,
            complexity="low",
        )
        if not result.success:
            raise RuntimeError(f"LLM generation failed: {result.content}")
        return result.content
