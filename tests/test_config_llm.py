# Copyright (c) 2025 Kenneth Stott
# Canary: 1dcb97c6-8791-4f5a-a805-4b8312a18447
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for LLM configuration."""

from __future__ import annotations

from constat.core.config import (
    LLMConfig, ModelSpec, TaskRoutingEntry, TaskRoutingConfig,
    DEFAULT_TASK_ROUTING,
)


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_model(self):
        """Test default model in LLMConfig."""
        config = LLMConfig(model="claude-sonnet-4-20250514")
        assert config.model == "claude-sonnet-4-20250514"
        assert config.provider == "anthropic"

    def test_task_routing_not_configured(self):
        """Test get_task_routing returns defaults when not configured."""
        config = LLMConfig(model="claude-sonnet-4-20250514")
        routing = config.get_task_routing()

        # Should have default routes for all standard task types
        assert "planning" in routing.routes
        assert "sql_generation" in routing.routes
        assert "python_analysis" in routing.routes

    def test_task_routing_uses_default_model_when_not_configured(self):
        """Test that default routing uses the config's model."""
        config = LLMConfig(model="my-custom-model")
        routing = config.get_task_routing()

        # All tasks should use the default model
        for task_type in routing.routes:
            models = routing.get_models_for_task(task_type)
            assert len(models) >= 1
            assert models[0].model == "my-custom-model"

    def test_custom_task_routing(self):
        """Test custom task routing configuration."""
        routing_config = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[
                    ModelSpec(provider="ollama", model="sqlcoder:7b"),
                    ModelSpec(model="claude-sonnet-4-20250514"),
                ]
            )
        })
        config = LLMConfig(
            model="claude-sonnet-4-20250514",
            task_routing=routing_config,
        )
        routing = config.get_task_routing()

        # Custom route should be used
        sql_models = routing.get_models_for_task("sql_generation")
        assert len(sql_models) == 2
        assert sql_models[0].provider == "ollama"
        assert sql_models[0].model == "sqlcoder:7b"

        # Default routes should still exist for other task types
        assert "planning" in routing.routes


class TestModelSpec:
    """Tests for ModelSpec model."""

    def test_model_spec_minimal(self):
        """ModelSpec with just model."""
        spec = ModelSpec(model="gpt-4")
        assert spec.model == "gpt-4"
        assert spec.provider is None
        assert spec.base_url is None

    def test_model_spec_with_provider(self):
        """ModelSpec with provider override."""
        spec = ModelSpec(provider="ollama", model="llama3.2:3b")
        assert spec.provider == "ollama"
        assert spec.model == "llama3.2:3b"

    def test_model_spec_with_base_url(self):
        """ModelSpec with custom base URL."""
        spec = ModelSpec(
            provider="ollama",
            model="llama3.2:3b",
            base_url="http://192.168.1.100:11434/v1",
        )
        assert spec.base_url == "http://192.168.1.100:11434/v1"


class TestTaskRoutingEntry:
    """Tests for TaskRoutingEntry model."""

    def test_routing_entry_single_model(self):
        """TaskRoutingEntry with single model."""
        entry = TaskRoutingEntry(
            models=[ModelSpec(model="claude-sonnet-4-20250514")]
        )
        assert len(entry.models) == 1
        assert entry.models[0].model == "claude-sonnet-4-20250514"

    def test_routing_entry_escalation_chain(self):
        """TaskRoutingEntry with escalation chain."""
        entry = TaskRoutingEntry(
            models=[
                ModelSpec(provider="ollama", model="sqlcoder:7b"),
                ModelSpec(model="claude-3-5-haiku-20241022"),
                ModelSpec(model="claude-sonnet-4-20250514"),
            ]
        )
        assert len(entry.models) == 3
        assert entry.models[0].provider == "ollama"
        assert entry.models[1].model == "claude-3-5-haiku-20241022"

    def test_routing_entry_high_complexity_models(self):
        """TaskRoutingEntry with high complexity override."""
        entry = TaskRoutingEntry(
            models=[ModelSpec(model="claude-3-5-haiku-20241022")],
            high_complexity_models=[ModelSpec(model="claude-sonnet-4-20250514")],
        )
        assert len(entry.models) == 1
        assert len(entry.high_complexity_models) == 1


class TestTaskRoutingConfig:
    """Tests for TaskRoutingConfig."""

    def test_get_models_for_task_medium_complexity(self):
        """Get models for medium complexity uses standard models."""
        config = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="standard-model")],
                high_complexity_models=[ModelSpec(model="advanced-model")],
            )
        })
        models = config.get_models_for_task("sql_generation", complexity="medium")
        assert len(models) == 1
        assert models[0].model == "standard-model"

    def test_get_models_for_task_high_complexity(self):
        """Get models for high complexity uses high_complexity_models."""
        config = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="standard-model")],
                high_complexity_models=[ModelSpec(model="advanced-model")],
            )
        })
        models = config.get_models_for_task("sql_generation", complexity="high")
        assert len(models) == 1
        assert models[0].model == "advanced-model"

    def test_get_models_for_task_high_complexity_fallback(self):
        """High complexity falls back to standard if no high_complexity_models."""
        config = TaskRoutingConfig(routes={
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="standard-model")],
            )
        })
        models = config.get_models_for_task("sql_generation", complexity="high")
        assert len(models) == 1
        assert models[0].model == "standard-model"

    def test_get_models_for_unknown_task(self):
        """Unknown task returns empty list."""
        config = TaskRoutingConfig(routes={})
        models = config.get_models_for_task("unknown_task")
        assert models == []


class TestDefaultTaskRouting:
    """Tests for default task routing configuration."""

    def test_default_routing_has_all_task_types(self):
        """Default routing has entries for all expected task types."""
        expected_tasks = [
            "planning", "replanning", "sql_generation", "python_analysis",
            "intent_classification", "summarization", "fact_resolution", "general"
        ]
        for task in expected_tasks:
            assert task in DEFAULT_TASK_ROUTING, f"Missing default route for {task}"

    def test_default_routing_entries_have_models(self):
        """Each default routing entry has at least one model."""
        for task_type, entry in DEFAULT_TASK_ROUTING.items():
            assert len(entry.models) >= 1, f"Task {task_type} has no models"
