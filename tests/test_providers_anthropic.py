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

"""Tests for Anthropic provider."""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

from constat.providers import AnthropicProvider

from tests.test_providers_shared import SAMPLE_TOOLS, TOOL_HANDLERS


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not set — required for this test")


# =============================================================================
# Anthropic Provider Tests
# =============================================================================

class TestAnthropicProvider:
    """Tests for Anthropic Claude provider."""

    def test_instantiation(self, require_anthropic_key):
        """Provider can be instantiated."""
        provider = AnthropicProvider()
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.supports_tools is True

    def test_instantiation_with_custom_model(self, require_anthropic_key):
        """Provider accepts custom model."""
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        assert provider.model == "claude-haiku-4-5-20251001"

    def test_generate_simple(self, require_anthropic_key):
        """Basic generation without tools."""
        last_exc = None
        for attempt in range(3):
            try:
                provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                response = provider.generate(
                    system="You are a helpful assistant. Be concise.",
                    user_message="What is 2 + 2? Reply with just the number.",
                    max_tokens=50,
                )
                assert "4" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_code(self, require_anthropic_key):
        """Code generation extracts from markdown blocks."""
        last_exc = None
        for attempt in range(3):
            try:
                provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                response = provider.generate_code(
                    system="You are a Python expert. Return only code in markdown blocks.",
                    user_message="Write a function that adds two numbers. Just the function, no explanation.",
                    max_tokens=200,
                )
                assert "def" in response
                assert "```" not in response  # Should be extracted
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_tools(self, require_anthropic_key):
        """Generation with tool calling."""
        last_exc = None
        for attempt in range(3):
            try:
                provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                response = provider.generate(
                    system="You have access to tools. Use them to answer questions.",
                    user_message="What's the weather in Paris?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "Paris" in response or "72" in response or "sunny" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_generate_with_calculation_tool(self, require_anthropic_key):
        """Tool calling with calculation."""
        last_exc = None
        for attempt in range(3):
            try:
                provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
                response = provider.generate(
                    system="Use the calculate tool for math. Report the result.",
                    user_message="What is 15 * 7?",
                    tools=SAMPLE_TOOLS,
                    tool_handlers=TOOL_HANDLERS,
                    max_tokens=500,
                )
                assert "105" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_model_override_in_generate(self, require_anthropic_key):
        """Model can be overridden per-call (for task-type routing)."""
        # Initialize with default sonnet model
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.model == "claude-sonnet-4-20250514"

        last_exc = None
        for attempt in range(3):
            try:
                # Override to haiku for this specific call (simulating task routing)
                response = provider.generate(
                    system="You are a helpful assistant. Be concise.",
                    user_message="What is 3 + 3? Reply with just the number.",
                    model="claude-haiku-4-5-20251001",  # Override for this call
                    max_tokens=50,
                )
                assert "6" in response

                # Original model should still be sonnet
                assert provider.model == "claude-sonnet-4-20250514"
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_task_routing_integration(self, require_anthropic_key):
        """Test task-type routing as used by planner/session."""
        from constat.core.config import LLMConfig, TaskRoutingConfig, TaskRoutingEntry, ModelSpec

        # Configure task routing like production would
        routing = TaskRoutingConfig(routes={
            "planning": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
            "sql_generation": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
            "summarization": TaskRoutingEntry(
                models=[ModelSpec(model="claude-haiku-4-5-20251001")]
            ),
        })
        llm_config = LLMConfig(
            model="claude-sonnet-4-20250514",  # Default is sonnet
            task_routing=routing,
        )

        # Provider initialized with default model
        provider = AnthropicProvider(model=llm_config.model)
        assert provider.model == "claude-sonnet-4-20250514"

        # Get routing config and select model for planning task
        task_routing = llm_config.get_task_routing()
        planning_models = task_routing.get_models_for_task("planning")
        assert len(planning_models) >= 1
        planning_model = planning_models[0].model
        assert planning_model == "claude-haiku-4-5-20251001"

        last_exc = None
        for attempt in range(3):
            try:
                response = provider.generate(
                    system="You are a helpful assistant.",
                    user_message="What is 1 + 1?",
                    model=planning_model,  # Use planning task model
                    max_tokens=50,
                )
                assert "2" in response
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc
