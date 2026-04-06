from __future__ import annotations

# Copyright (c) 2025 Kenneth Stott
# Canary: f0e226dd-3488-41ae-80c7-40d2fbe56fa8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared/base provider tests: BaseLLMProvider, TaskRouter, multi-provider routing.

Provider test strategy:
- Anthropic: tests/test_providers_anthropic.py
- Ollama:    tests/test_providers_ollama.py
- OpenAI:    tests/test_providers_openai.py
- Gemini:    tests/test_providers_google.py
- Grok/Together/Groq: tests/test_providers_other.py
- Mistral:   tests/test_providers_mistral.py
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


def get_ollama_model() -> str | None:
    """Get an available Ollama model for testing."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
            for preferred in ["llama3.2:3b", "llama3.1:8b", "llama3:8b"]:
                if preferred in models:
                    return preferred
            for model in models:
                if "llama" in model.lower():
                    return model
            if models:
                return models[0]
    except Exception:
        pass  # Probe: Ollama not reachable; return None as sentinel
    return None


OLLAMA_TEST_MODEL = get_ollama_model()


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not set — required for this test")


@pytest.fixture
def require_ollama_model():
    if not OLLAMA_TEST_MODEL:
        pytest.fail("No suitable Ollama model available for testing — required for this test")


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
# Task Router Tests (replaces ProviderFactory tests)
# =============================================================================

class TestTaskRouter:
    """Tests for TaskRouter with task-type routing and automatic escalation."""

    def test_router_creates_provider_from_config(self):
        """Router creates provider from LLM config."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )
        router = TaskRouter(llm_config)

        # Router should be initialized
        assert router.llm_config == llm_config
        assert router.routing_config is not None

    def test_router_uses_task_routing_config(self):
        """Router uses task routing configuration."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[
                    ModelSpec(provider="ollama", model="sqlcoder:7b"),
                    ModelSpec(model="claude-sonnet-4-20250514"),
                ]
            )
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Router should have the custom routing
        models = router.routing_config.get_models_for_task("sql_generation")
        assert len(models) == 2
        assert models[0].provider == "ollama"

    def test_router_caches_providers(self):
        """Router caches provider instances."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )
        router = TaskRouter(llm_config)

        # Access the provider twice
        spec1 = router.routing_config.get_models_for_task("planning")[0]
        spec2 = router.routing_config.get_models_for_task("planning")[0]

        # Models should be equivalent
        assert spec1.model == spec2.model

    def test_router_high_complexity_uses_different_models(self):
        """Router selects different models for high complexity tasks."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "python_analysis": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")],
                high_complexity_models=[ModelSpec(model="claude-sonnet-4-20250514")],
            )
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Medium complexity uses standard model
        medium_models = router.routing_config.get_models_for_task("python_analysis", "medium")
        assert medium_models[0].model == "claude-haiku-4-5-20251001"

        # High complexity uses advanced model
        high_models = router.routing_config.get_models_for_task("python_analysis", "high")
        assert high_models[0].model == "claude-sonnet-4-20250514"

    def test_router_escalation_stats(self):
        """Router tracks escalation statistics."""
        from constat.core.config import LLMConfig
        from constat.providers import TaskRouter

        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
        )
        router = TaskRouter(llm_config)

        # Initially no escalations
        stats = router.get_escalation_stats()
        assert stats["total_escalations"] == 0

        # Clear stats should work
        router.clear_stats()
        stats = router.get_escalation_stats()
        assert stats["total_escalations"] == 0

    def test_router_unknown_provider_returns_failed_result(self):
        """Router returns failed result for unknown provider."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter
        from constat.core.models import TaskType

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(provider="unknown_provider", model="some-model")],
            )
        })
        llm_config = LLMConfig(
            provider="unknown_provider",
            model="some-model",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Execute catches exceptions and returns a failed TaskResult
        result = router.execute(
            task_type=TaskType.PLANNING,
            system="Test",
            user_message="Test",
        )

        assert not result.success
        assert "Unknown provider" in result.content


# =============================================================================
# Multi-Provider Integration Tests (updated for task routing)
# =============================================================================

class TestMultiProviderIntegration:
    """Integration tests for multi-provider task routing.

    These tests verify that different providers can be used for different task types.
    """

    def test_anthropic_planning_ollama_sql(self, require_anthropic_key, require_ollama_model):
        """Use Anthropic for planning, Ollama for SQL generation."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(provider="ollama", model=OLLAMA_TEST_MODEL)]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-haiku-4-5-20251001",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Verify routing configuration
        planning_models = router.routing_config.get_models_for_task("planning")
        assert planning_models[0].model == "claude-haiku-4-5-20251001"
        assert planning_models[0].provider is None  # Uses default

        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == OLLAMA_TEST_MODEL

    def test_all_tasks_same_provider_different_models(self, require_anthropic_key):
        """All task types use same provider but different models."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-sonnet-4-20250514")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
            "summarization": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            task_routing=routing,
        )
        router = TaskRouter(llm_config)

        # Verify different models for different task types
        planning_models = router.routing_config.get_models_for_task("planning")
        sql_models = router.routing_config.get_models_for_task("sql_generation")

        assert planning_models[0].model == "claude-sonnet-4-20250514"
        assert sql_models[0].model == "claude-haiku-4-5-20251001"


# =============================================================================
# Task Routing Integration Tests
# =============================================================================

class TestTaskRoutingIntegration:
    """
    Integration tests verifying that task-type routing works correctly
    between Planner and TaskRouter components.

    Note: These are integration tests, not true E2E tests. They test
    Planner + mock Router, not the full Session.solve() pipeline.

    These tests verify that:
    1. Planning phase uses the PLANNING task type routing
    2. SQL generation uses SQL_GENERATION task type routing
    3. Different providers can be used for different task types
    """

    def test_routing_tracks_models_used_at_each_phase(self):
        """
        Integration: Track which models are called at each phase of NLQ processing.

        Uses a mock router to verify correct task type routing.
        """
        from unittest.mock import MagicMock, patch
        from constat.core.config import Config, LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter
        from constat.providers.router import TaskResult
        from constat.execution.planner import Planner
        from constat.catalog.schema_manager import SchemaManager
        from constat.core.models import TaskType

        # Track all execute calls
        execute_calls = []

        # Create routing with specific models for each task type
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="planning-model-v1")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="sql-model-v1")]
            ),
            "python_analysis": TaskRoutingEntry(
                models=[ModelSpec(model="python-model-v1")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="default-model-v1",
            api_key="test-key",
            task_routing=routing,
        )

        # Create a minimal config
        config = Config(
            databases={"test": {"uri": "sqlite:///:memory:"}},
            llm=llm_config,
        )

        # Create a mock router that tracks calls
        mock_router = MagicMock(spec=TaskRouter)

        def track_execute(task_type=None, **kwargs):
            execute_calls.append({
                "task_type": task_type.value if hasattr(task_type, 'value') else str(task_type),
            })
            return TaskResult(
                success=True,
                content='''```json
{
    "reasoning": "Test plan",
    "steps": [
        {"number": 1, "goal": "Load data", "inputs": [], "outputs": ["data"], "depends_on": [], "task_type": "sql_generation"}
    ]
}
```''',
                model_used="planning-model-v1",
                provider_used="anthropic",
            )

        mock_router.execute = MagicMock(side_effect=track_execute)

        # Create schema manager with mock
        schema_manager = MagicMock(spec=SchemaManager)
        schema_manager.get_overview.return_value = "test: 1 table"
        schema_manager.get_table_schema.return_value = {"table": "test", "columns": []}
        schema_manager.find_relevant_tables.return_value = []

        # Create planner with mock router
        planner = Planner(config, schema_manager, mock_router)

        # Run planning
        execute_calls.clear()
        result = planner.plan("Show me all customers")

        # Verify planning task type was used
        assert len(execute_calls) >= 1
        planning_call = execute_calls[0]
        assert planning_call["task_type"] == "planning", (
            f"Planning should use 'planning' task type, got '{planning_call['task_type']}'"
        )

    def test_routing_different_providers_per_task(self):
        """
        Integration: Verify different providers can be used for different task types.

        Configures Anthropic for planning and Ollama for SQL generation.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Configure different providers for different task types
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-sonnet-4-20250514")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(provider="ollama", model="sqlcoder:7b")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # Verify routing configuration
        planning_models = router.routing_config.get_models_for_task("planning")
        sql_models = router.routing_config.get_models_for_task("sql_generation")

        # Planning uses default provider (anthropic)
        assert planning_models[0].provider is None
        assert planning_models[0].model == "claude-sonnet-4-20250514"

        # SQL generation uses ollama
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == "sqlcoder:7b"

    def test_routing_with_escalation_chain(self):
        """
        Integration: Verify escalation chain configuration works.

        Configures multiple models per task type for fallback behavior.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Configure escalation chain: ollama -> haiku -> sonnet
        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[
                    ModelSpec(provider="ollama", model="sqlcoder:7b"),
                    ModelSpec(model="claude-haiku-4-5-20251001"),
                    ModelSpec(model="claude-sonnet-4-20250514"),
                ]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # Verify escalation chain
        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert len(sql_models) == 3
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == "sqlcoder:7b"
        assert sql_models[1].model == "claude-haiku-4-5-20251001"
        assert sql_models[2].model == "claude-sonnet-4-20250514"

    def test_routing_falls_back_to_defaults(self):
        """
        Integration: Verify that unconfigured task types fall back to defaults.
        """
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec
        from constat.providers import TaskRouter

        # Only configure sql_generation, leave others as default
        routing = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="special-sql-model")]
            ),
        })
        llm_config = LLMConfig(
            provider="anthropic",
            model="default-model",
            api_key="test-key",
            task_routing=routing,
        )

        router = TaskRouter(llm_config)

        # SQL generation should use configured model
        sql_models = router.routing_config.get_models_for_task("sql_generation")
        assert sql_models[0].model == "special-sql-model"

        # Planning should fall back to default routing
        planning_models = router.routing_config.get_models_for_task("planning")
        # Default routing uses claude-sonnet-4-20250514 for planning
        assert len(planning_models) >= 1

        # Unknown task type should return empty (router handles fallback)
        unknown_models = router.routing_config.get_models_for_task("nonexistent")
        assert unknown_models == []
