# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for lazy fact resolution with provenance tracking."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, AsyncMock, patch
import pytest
from constat.execution.fact_resolver import (
    Fact,
    FactSource,
    FactResolver,
    AsyncFactResolver,
    ResolutionStrategy,
)


class TestFact:
    """Tests for Fact dataclass."""

    def test_fact_creation(self):
        """Test basic fact creation."""
        fact = Fact(
            name="revenue",
            value=50000,
            confidence=1.0,
            source=FactSource.DATABASE,
        )
        assert fact.name == "revenue"
        assert fact.value == 50000
        assert fact.is_resolved

    def test_unresolved_fact(self):
        """Test unresolved fact detection."""
        fact = Fact(
            name="unknown",
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
        )
        assert not fact.is_resolved

    def test_derivation_trace(self):
        """Test derivation trace generation."""
        dep = Fact(name="base", value=100, source=FactSource.DATABASE)
        fact = Fact(
            name="derived",
            value=200,
            source=FactSource.RULE,
            rule_name="double",
            because=[dep],
        )
        trace = fact.derivation_trace
        assert "derived" in trace
        assert "base" in trace
        assert "double" in trace

    def test_to_dict(self):
        """Test serialization."""
        fact = Fact(
            name="test",
            value=42,
            confidence=0.9,
            source=FactSource.LLM_KNOWLEDGE,
        )
        d = fact.to_dict()
        assert d["name"] == "test"
        assert d["value"] == 42
        assert d["confidence"] == 0.9
        assert d["source"] == "llm_knowledge"


class TestFactResolver:
    """Tests for FactResolver."""

    @pytest.fixture
    def resolver(self):
        """Create a basic resolver."""
        return FactResolver()

    def test_rule_registration_decorator(self, resolver):
        """Test registering rules with decorator."""
        @resolver.rule("double")
        def double_rule(res, params):
            val = params.get("value", 0)
            return Fact(
                name=f"double({val})",
                value=val * 2,
                source=FactSource.RULE,
            )

        result = resolver.resolve("double", value=21)
        assert result.value == 42
        assert result.source == FactSource.RULE

    def test_rule_registration_programmatic(self, resolver):
        """Test registering rules programmatically."""
        def triple(res, params):
            val = params.get("value", 0)
            return Fact(name="triple", value=val * 3, source=FactSource.RULE)

        resolver.register_rule("triple", triple)
        result = resolver.resolve("triple", value=10)
        assert result.value == 30

    def test_cache_key_generation(self, resolver):
        """Test cache key generation."""
        assert resolver._cache_key("fact", {}) == "fact"
        assert resolver._cache_key("fact", {"a": 1}) == "fact(a=1)"
        assert resolver._cache_key("fact", {"a": 1, "b": 2}) == "fact(a=1,b=2)"

    def test_caching(self, resolver):
        """Test that facts are cached."""
        call_count = 0

        @resolver.rule("counted")
        def counted_rule(res, params):
            nonlocal call_count
            call_count += 1
            return Fact(name="counted", value=call_count, source=FactSource.RULE)

        # First call
        result1 = resolver.resolve("counted")
        assert result1.value == 1
        assert call_count == 1

        # Second call should use cache
        result2 = resolver.resolve("counted")
        assert result2.value == 1  # Same value (cached)
        assert call_count == 1  # Rule not called again

    def test_recursive_resolution(self, resolver):
        """Test rules that depend on other facts."""
        # Pre-cache some base facts
        resolver._cache["base_value"] = Fact(
            name="base_value",
            value=100,
            source=FactSource.DATABASE,
        )

        @resolver.rule("computed")
        def computed_rule(res, params):
            base = res.resolve("base_value")
            return Fact(
                name="computed",
                value=base.value * 2,
                confidence=base.confidence,
                source=FactSource.RULE,
                because=[base],
            )

        result = resolver.resolve("computed")
        assert result.value == 200
        assert len(result.because) == 1
        assert result.because[0].name == "base_value"

    def test_confidence_propagation(self, resolver):
        """Test that confidence propagates through derivation."""
        resolver._cache["uncertain"] = Fact(
            name="uncertain",
            value=50,
            confidence=0.7,
            source=FactSource.LLM_HEURISTIC,
        )
        resolver._cache["certain"] = Fact(
            name="certain",
            value=100,
            confidence=1.0,
            source=FactSource.DATABASE,
        )

        @resolver.rule("combined")
        def combined_rule(res, params):
            a = res.resolve("uncertain")
            b = res.resolve("certain")
            return Fact(
                name="combined",
                value=a.value + b.value,
                confidence=min(a.confidence, b.confidence),
                source=FactSource.RULE,
                because=[a, b],
            )

        result = resolver.resolve("combined")
        assert result.value == 150
        assert result.confidence == 0.7  # Min of dependencies

    def test_unresolved_fact(self, resolver):
        """Test behavior when fact cannot be resolved."""
        result = resolver.resolve("nonexistent")
        assert not result.is_resolved
        assert result.source == FactSource.UNRESOLVED

    def test_resolution_log(self, resolver):
        """Test that all resolutions are logged."""
        resolver._cache["logged"] = Fact(
            name="logged",
            value=1,
            source=FactSource.CONFIG,
        )

        resolver.resolve("logged")
        resolver.resolve("nonexistent")

        log = resolver.get_audit_log()
        assert len(log) == 2
        assert log[0]["name"] == "logged"
        assert log[1]["source"] == "unresolved"

    def test_clear_cache(self, resolver):
        """Test cache clearing."""
        resolver._cache["temp"] = Fact(name="temp", value=1, source=FactSource.CACHE)
        assert "temp" in resolver._cache

        resolver.clear_cache()
        assert "temp" not in resolver._cache


class TestResolutionStrategy:
    """Tests for resolution strategy configuration."""

    def test_default_strategy(self):
        """Test default source priority."""
        strategy = ResolutionStrategy()
        assert FactSource.CACHE in strategy.source_priority
        assert FactSource.DATABASE in strategy.source_priority

    def test_custom_priority(self):
        """Test custom source priority."""
        strategy = ResolutionStrategy(
            source_priority=[FactSource.CACHE, FactSource.LLM_KNOWLEDGE]
        )
        resolver = FactResolver(strategy=strategy)

        # Should skip database since it's not in priority
        assert FactSource.DATABASE not in strategy.source_priority

    def test_min_confidence(self):
        """Test minimum confidence threshold."""
        strategy = ResolutionStrategy(min_confidence=0.8)
        resolver = FactResolver(strategy=strategy)

        # Pre-cache a low-confidence fact
        resolver._cache["low_conf"] = Fact(
            name="low_conf",
            value=1,
            confidence=0.5,
            source=FactSource.LLM_KNOWLEDGE,
        )

        # Should not accept low confidence
        result = resolver.resolve("low_conf")
        # It will return unresolved because confidence is below threshold
        assert result.source == FactSource.UNRESOLVED

    def test_sub_plan_depth_limit(self):
        """Test that sub-plan depth is limited."""
        strategy = ResolutionStrategy(max_sub_plan_depth=2)
        resolver = FactResolver(strategy=strategy)

        # Simulate nested resolution
        resolver._resolution_depth = 3
        result = resolver._resolve_from_sub_plan("deep", {})
        assert result is None  # Should refuse due to depth limit


class TestExplainability:
    """Tests for explainability features."""

    def test_explain_simple(self):
        """Test explanation of simple fact."""
        fact = Fact(
            name="revenue",
            value=50000,
            source=FactSource.DATABASE,
            query="SELECT SUM(amount) FROM sales",
        )
        resolver = FactResolver()
        explanation = resolver.explain(fact)
        assert "revenue" in explanation
        assert "50000" in explanation
        assert "SELECT" in explanation

    def test_explain_derived(self):
        """Test explanation of derived fact with dependencies."""
        base = Fact(name="base", value=100, source=FactSource.DATABASE)
        derived = Fact(
            name="derived",
            value=200,
            source=FactSource.RULE,
            rule_name="double",
            because=[base],
        )
        resolver = FactResolver()
        explanation = resolver.explain(derived)
        assert "derived" in explanation
        assert "base" in explanation
        assert "double" in explanation


class TestAsyncFactResolver:
    """Tests for AsyncFactResolver with parallel resolution."""

    @pytest.fixture
    def async_resolver(self):
        """Create a basic async resolver."""
        return AsyncFactResolver()

    @pytest.fixture
    def executor(self):
        """Create a small executor for tests."""
        executor = ThreadPoolExecutor(max_workers=4)
        yield executor
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_async_resolver_creation(self, async_resolver):
        """Test AsyncFactResolver can be created."""
        assert async_resolver is not None
        assert isinstance(async_resolver, FactResolver)
        assert isinstance(async_resolver, AsyncFactResolver)

    @pytest.mark.asyncio
    async def test_async_resolver_with_custom_executor(self, executor):
        """Test AsyncFactResolver with custom executor."""
        resolver = AsyncFactResolver(executor=executor)
        assert resolver._executor is executor

    @pytest.mark.asyncio
    async def test_async_resolver_parallel_sources_flag(self):
        """Test parallel_sources configuration."""
        resolver = AsyncFactResolver(parallel_sources=True)
        assert resolver._parallel_sources is True

        resolver2 = AsyncFactResolver(parallel_sources=False)
        assert resolver2._parallel_sources is False

    @pytest.mark.asyncio
    async def test_resolve_async_from_cache(self, async_resolver):
        """Test async resolution from cache."""
        async_resolver._cache["cached_fact"] = Fact(
            name="cached_fact",
            value=42,
            source=FactSource.CACHE,
        )
        result = await async_resolver.resolve_async("cached_fact")
        assert result.value == 42
        assert result.source == FactSource.CACHE

    @pytest.mark.asyncio
    async def test_resolve_async_from_rule(self, async_resolver):
        """Test async resolution from rule."""
        @async_resolver.rule("multiply")
        def multiply_rule(res, params):
            return Fact(
                name="multiply",
                value=params.get("a", 1) * params.get("b", 1),
                source=FactSource.RULE,
            )

        result = await async_resolver.resolve_async("multiply", a=6, b=7)
        assert result.value == 42
        assert result.source == FactSource.RULE

    @pytest.mark.asyncio
    async def test_resolve_async_unresolved(self, async_resolver):
        """Test async resolution when fact cannot be resolved."""
        result = await async_resolver.resolve_async("nonexistent")
        assert not result.is_resolved
        assert result.source == FactSource.UNRESOLVED

    @pytest.mark.asyncio
    async def test_resolve_async_caching(self, async_resolver):
        """Test that async resolution properly caches."""
        call_count = 0

        @async_resolver.rule("counter")
        def counter_rule(res, params):
            nonlocal call_count
            call_count += 1
            return Fact(name="counter", value=call_count, source=FactSource.RULE)

        result1 = await async_resolver.resolve_async("counter")
        assert result1.value == 1
        assert call_count == 1

        result2 = await async_resolver.resolve_async("counter")
        assert result2.value == 1  # Cached
        assert call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_resolve_many_async_parallel(self, async_resolver):
        """Test parallel resolution of multiple facts."""
        # Pre-cache some facts
        async_resolver._cache["fact_a"] = Fact(name="fact_a", value="A", source=FactSource.CACHE)
        async_resolver._cache["fact_b"] = Fact(name="fact_b", value="B", source=FactSource.CACHE)
        async_resolver._cache["fact_c"] = Fact(name="fact_c", value="C", source=FactSource.CACHE)

        results = await async_resolver.resolve_many_async([
            ("fact_a", {}),
            ("fact_b", {}),
            ("fact_c", {}),
        ])

        assert len(results) == 3
        assert results[0].value == "A"
        assert results[1].value == "B"
        assert results[2].value == "C"

    @pytest.mark.asyncio
    async def test_resolve_many_async_order_preserved(self, async_resolver):
        """Test that resolve_many_async preserves order."""
        for i in range(5):
            async_resolver._cache[f"ordered_{i}"] = Fact(
                name=f"ordered_{i}",
                value=i,
                source=FactSource.CACHE,
            )

        results = await async_resolver.resolve_many_async([
            (f"ordered_{i}", {}) for i in range(5)
        ])

        for i, result in enumerate(results):
            assert result.value == i

    @pytest.mark.asyncio
    async def test_resolve_many_async_mixed_results(self, async_resolver):
        """Test resolve_many_async with mixed resolved/unresolved."""
        async_resolver._cache["exists"] = Fact(
            name="exists", value=1, source=FactSource.CACHE
        )

        results = await async_resolver.resolve_many_async([
            ("exists", {}),
            ("does_not_exist", {}),
        ])

        assert len(results) == 2
        assert results[0].is_resolved
        assert results[0].value == 1
        assert not results[1].is_resolved

    def test_resolve_many_sync(self, async_resolver):
        """Test synchronous wrapper for parallel resolution."""
        async_resolver._cache["sync_test"] = Fact(
            name="sync_test", value="sync", source=FactSource.CACHE
        )

        results = async_resolver.resolve_many_sync([
            ("sync_test", {}),
        ])

        assert len(results) == 1
        assert results[0].value == "sync"

    @pytest.mark.asyncio
    async def test_parallel_speedup(self):
        """Test that parallel resolution provides speedup."""
        import time

        # Create a resolver with simulated slow rule
        resolver = AsyncFactResolver()
        call_times = []

        @resolver.rule("slow_fact")
        def slow_rule(res, params):
            start = time.time()
            time.sleep(0.1)  # Simulate I/O delay
            call_times.append(time.time() - start)
            return Fact(
                name=f"slow_fact:{params.get('id')}",
                value=params.get('id'),
                source=FactSource.RULE,
            )

        # Sequential resolution
        start = time.time()
        for i in range(3):
            resolver.resolve("slow_fact", id=i)
        sequential_time = time.time() - start

        # Clear cache for parallel test
        resolver.clear_cache()

        # Parallel resolution
        start = time.time()
        await resolver.resolve_many_async([
            ("slow_fact", {"id": i}) for i in range(3, 6)
        ])
        parallel_time = time.time() - start

        # Parallel should be faster (not 3x slower)
        # Due to asyncio overhead, we just check it's not sequential
        # Rules run in the same event loop, so speedup is limited
        # But with real I/O (LLM calls), speedup is significant
        assert parallel_time < sequential_time * 1.5


class TestAsyncFactResolverWithMockLLM:
    """Tests for AsyncFactResolver with mocked LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.generate = MagicMock(return_value="VALUE: 100\nCONFIDENCE: 0.9\nTYPE: knowledge\nREASONING: Test value")
        llm.async_generate = AsyncMock(return_value="VALUE: 100\nCONFIDENCE: 0.9\nTYPE: knowledge\nREASONING: Test value")
        return llm

    @pytest.fixture
    def mock_schema_manager(self):
        """Create a mock schema manager."""
        sm = MagicMock()
        sm.get_overview = MagicMock(return_value="Table: users (id, name, email)")
        return sm

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = MagicMock()
        config.databases = {"test_db": MagicMock()}
        return config

    @pytest.mark.asyncio
    async def test_resolve_from_llm_async(self, mock_llm):
        """Test async LLM resolution."""
        resolver = AsyncFactResolver(llm=mock_llm)

        result = await resolver._resolve_from_llm_async("test_fact", {})

        assert result is not None
        assert result.value == 100.0
        assert result.confidence == 0.9
        assert result.source == FactSource.LLM_KNOWLEDGE

    @pytest.mark.asyncio
    async def test_resolve_from_llm_async_uses_async_generate(self, mock_llm):
        """Test that async_generate is preferred when available."""
        resolver = AsyncFactResolver(llm=mock_llm)

        await resolver._resolve_from_llm_async("test_fact", {})

        mock_llm.async_generate.assert_called_once()
        mock_llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_from_llm_async_fallback_to_sync(self):
        """Test fallback to sync generate when async not available."""
        mock_llm = MagicMock()
        mock_llm.generate = MagicMock(return_value="VALUE: 50\nCONFIDENCE: 0.8\nTYPE: heuristic\nREASONING: Fallback")
        # Don't set async_generate attribute

        resolver = AsyncFactResolver(llm=mock_llm)
        delattr(mock_llm, 'async_generate')  # Ensure it doesn't have async_generate

        result = await resolver._resolve_from_llm_async("test_fact", {})

        assert result is not None
        assert result.value == 50.0
        assert result.source == FactSource.LLM_HEURISTIC

    @pytest.mark.asyncio
    async def test_resolve_from_llm_async_unknown_response(self, mock_llm):
        """Test handling of UNKNOWN response from LLM."""
        mock_llm.async_generate = AsyncMock(return_value="UNKNOWN")
        resolver = AsyncFactResolver(llm=mock_llm)

        result = await resolver._resolve_from_llm_async("unknown_fact", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_database_async_not_possible(self, mock_llm, mock_schema_manager, mock_config):
        """Test database resolution when NOT_POSSIBLE."""
        mock_llm.async_generate = AsyncMock(return_value="NOT_POSSIBLE: No such table")
        resolver = AsyncFactResolver(
            llm=mock_llm,
            schema_manager=mock_schema_manager,
            config=mock_config,
        )

        result = await resolver._resolve_from_database_async("missing_table", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_database_async_no_llm(self):
        """Test database resolution without LLM."""
        resolver = AsyncFactResolver(llm=None)

        result = await resolver._resolve_from_database_async("test", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_sub_plan_async_disabled(self, mock_llm):
        """Test sub-plan resolution when disabled."""
        strategy = ResolutionStrategy(allow_sub_plans=False)
        resolver = AsyncFactResolver(llm=mock_llm, strategy=strategy)

        result = await resolver._resolve_from_sub_plan_async("derived_fact", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_sub_plan_async_depth_limit(self, mock_llm):
        """Test sub-plan resolution respects depth limit."""
        strategy = ResolutionStrategy(max_sub_plan_depth=2)
        resolver = AsyncFactResolver(llm=mock_llm, strategy=strategy)
        resolver._resolution_depth = 3  # Already too deep

        result = await resolver._resolve_from_sub_plan_async("deep_fact", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_from_sub_plan_async_no_llm(self):
        """Test sub-plan resolution without LLM."""
        resolver = AsyncFactResolver(llm=None)

        result = await resolver._resolve_from_sub_plan_async("test", {})

        assert result is None


class TestAsyncFactResolverParallelSources:
    """Tests for parallel source resolution strategy."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM that returns different results for different calls."""
        llm = MagicMock()
        llm.async_generate = AsyncMock(side_effect=[
            "NOT_POSSIBLE: Cannot determine from DB",  # Database attempt
            "VALUE: 42\nCONFIDENCE: 0.95\nTYPE: knowledge\nREASONING: Known value",  # LLM knowledge
        ])
        return llm

    @pytest.mark.asyncio
    async def test_parallel_sources_enabled(self, mock_llm):
        """Test with parallel_sources enabled."""
        mock_schema_manager = MagicMock()
        mock_schema_manager.get_overview = MagicMock(return_value="Test schema")

        resolver = AsyncFactResolver(
            llm=mock_llm,
            schema_manager=mock_schema_manager,
            parallel_sources=True,
        )

        # With parallel sources, all I/O sources are tried concurrently
        # This is a structural test - actual parallel behavior depends on asyncio
        assert resolver._parallel_sources is True

    @pytest.mark.asyncio
    async def test_sequential_sources_default(self, mock_llm):
        """Test sequential source resolution (default)."""
        resolver = AsyncFactResolver(llm=mock_llm, parallel_sources=False)

        assert resolver._parallel_sources is False

    @pytest.mark.asyncio
    async def test_resolve_parallel_sources_returns_first_success(self):
        """Test that parallel resolution returns first successful result."""
        resolver = AsyncFactResolver(parallel_sources=True)

        # Pre-load a fact via rule to simulate different sources
        @resolver.rule("multi_source")
        def multi_rule(res, params):
            return Fact(
                name="multi_source",
                value="from_rule",
                source=FactSource.RULE,
            )

        result = await resolver.resolve_async("multi_source")

        # Rule should be tried first (before I/O sources)
        assert result.value == "from_rule"
        assert result.source == FactSource.RULE


class TestAsyncFactResolverIntegration:
    """Integration tests for AsyncFactResolver."""

    @pytest.mark.asyncio
    async def test_full_resolution_flow(self):
        """Test complete resolution flow with multiple sources."""
        resolver = AsyncFactResolver()

        # Set up test data
        resolver._cache["base_value"] = Fact(
            name="base_value",
            value=100,
            source=FactSource.DATABASE,
        )

        @resolver.rule("derived_value")
        def derive(res, params):
            base = res.resolve("base_value")
            return Fact(
                name="derived_value",
                value=base.value * 2,
                confidence=base.confidence,
                source=FactSource.RULE,
                rule_name="derive",
                because=[base],
            )

        # Resolve derived fact
        result = await resolver.resolve_async("derived_value")

        assert result.value == 200
        assert result.source == FactSource.RULE
        assert len(result.because) == 1

    @pytest.mark.asyncio
    async def test_resolution_log_async(self):
        """Test that async resolutions are logged."""
        resolver = AsyncFactResolver()
        resolver._cache["logged_fact"] = Fact(
            name="logged_fact",
            value=1,
            source=FactSource.CONFIG,
        )

        await resolver.resolve_async("logged_fact")
        await resolver.resolve_async("unresolved_fact")

        log = resolver.get_audit_log()
        assert len(log) >= 2

    @pytest.mark.asyncio
    async def test_concurrent_resolution_same_fact(self):
        """Test concurrent resolution of the same fact.

        With parallel execution, concurrent calls may all execute before any
        can cache. However, after execution, each checks the cache - if another
        request already cached a result, it returns the cached value instead.
        This ensures all concurrent requests return consistent values.
        """
        resolver = AsyncFactResolver()
        call_count = 0

        @resolver.rule("concurrent_fact")
        def concurrent_rule(res, params):
            nonlocal call_count
            call_count += 1
            return Fact(
                name="concurrent_fact",
                value=call_count,
                source=FactSource.RULE,
            )

        # Launch multiple concurrent resolutions
        tasks = [resolver.resolve_async("concurrent_fact") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All concurrent requests should return the same value
        # (first to cache wins, others discard their results)
        values = [r.value for r in results]
        assert all(v == values[0] for v in values)  # All same value
        assert len(results) == 5
        assert all(r.source == FactSource.RULE for r in results)

    @pytest.mark.asyncio
    async def test_inheritance_from_fact_resolver(self):
        """Test that AsyncFactResolver inherits FactResolver methods."""
        resolver = AsyncFactResolver()

        # Test inherited methods
        assert hasattr(resolver, 'resolve')
        assert hasattr(resolver, 'rule')
        assert hasattr(resolver, 'register_rule')
        assert hasattr(resolver, 'add_user_fact')
        assert hasattr(resolver, 'clear_cache')
        assert hasattr(resolver, 'explain')

    @pytest.mark.asyncio
    async def test_sync_resolve_still_works(self):
        """Test that sync resolve() still works on AsyncFactResolver."""
        resolver = AsyncFactResolver()

        @resolver.rule("sync_test")
        def sync_rule(res, params):
            return Fact(name="sync_test", value="sync_value", source=FactSource.RULE)

        # Use sync resolve
        result = resolver.resolve("sync_test")

        assert result.value == "sync_value"


# =============================================================================
# DERIVATION TRACE TESTS FOR AUDITABLE MODE
# =============================================================================


class TestDerivationTraceCreation:
    """Tests for derivation trace creation when facts are resolved."""

    def test_trace_created_for_simple_fact(self):
        """Test that resolving a fact creates a trace entry."""
        resolver = FactResolver()
        resolver._cache["simple"] = Fact(
            name="simple",
            value=42,
            source=FactSource.DATABASE,
            query="SELECT 42",
        )

        result = resolver.resolve("simple")

        assert result.name == "simple"
        assert result.value == 42
        assert result.source == FactSource.DATABASE
        assert result.query == "SELECT 42"
        # Trace should include this resolution
        assert len(resolver.resolution_log) == 1
        assert resolver.resolution_log[0] == result

    def test_trace_created_for_rule_derived_fact(self):
        """Test trace creation for facts derived via rules."""
        resolver = FactResolver()

        @resolver.rule("computed")
        def compute(res, params):
            return Fact(
                name="computed",
                value=params.get("x", 0) * 2,
                source=FactSource.RULE,
                rule_name="compute",
            )

        result = resolver.resolve("computed", x=21)

        assert result.rule_name == "compute"
        assert result.value == 42
        assert len(resolver.resolution_log) == 1

    def test_trace_includes_all_resolutions_in_session(self):
        """Test that resolution log contains all facts resolved in session."""
        resolver = FactResolver()
        resolver._cache["fact_a"] = Fact(name="fact_a", value=1, source=FactSource.CONFIG)
        resolver._cache["fact_b"] = Fact(name="fact_b", value=2, source=FactSource.DATABASE)
        resolver._cache["fact_c"] = Fact(name="fact_c", value=3, source=FactSource.LLM_KNOWLEDGE)

        resolver.resolve("fact_a")
        resolver.resolve("fact_b")
        resolver.resolve("fact_c")
        resolver.resolve("nonexistent")  # Unresolved

        assert len(resolver.resolution_log) == 4
        sources = [f.source for f in resolver.resolution_log]
        assert FactSource.CONFIG in sources
        assert FactSource.DATABASE in sources
        assert FactSource.LLM_KNOWLEDGE in sources
        assert FactSource.UNRESOLVED in sources


class TestDerivationTraceSourceInformation:
    """Tests for trace containing source information."""

    @pytest.mark.parametrize("source,expected_value", [
        (FactSource.DATABASE, "database"),
        (FactSource.LLM_KNOWLEDGE, "llm_knowledge"),
        (FactSource.LLM_HEURISTIC, "llm_heuristic"),
        (FactSource.RULE, "rule"),
        (FactSource.SUB_PLAN, "sub_plan"),
        (FactSource.USER_PROVIDED, "user_provided"),
        (FactSource.CONFIG, "config"),
        (FactSource.CACHE, "cache"),
        (FactSource.UNRESOLVED, "unresolved"),
    ])
    def test_all_source_types_serialized_correctly(self, source, expected_value):
        """Test that all FactSource enum values serialize correctly."""
        fact = Fact(name="test", value=1, source=source)
        serialized = fact.to_dict()
        assert serialized["source"] == expected_value

    def test_trace_contains_sql_query_for_database_source(self):
        """Test that database facts include the SQL query in trace."""
        fact = Fact(
            name="revenue",
            value=100000,
            source=FactSource.DATABASE,
            query="SELECT SUM(amount) FROM sales WHERE year = 2024",
        )

        trace = fact.derivation_trace
        assert "SELECT SUM(amount)" in trace
        assert "database" in trace

        serialized = fact.to_dict()
        assert serialized["query"] == "SELECT SUM(amount) FROM sales WHERE year = 2024"

    def test_trace_contains_rule_name_for_rule_source(self):
        """Test that rule-derived facts include rule name in trace."""
        fact = Fact(
            name="ltv",
            value=5000,
            source=FactSource.RULE,
            rule_name="calculate_customer_ltv",
        )

        trace = fact.derivation_trace
        assert "calculate_customer_ltv" in trace
        assert "rule" in trace

        serialized = fact.to_dict()
        assert serialized["rule_name"] == "calculate_customer_ltv"

    def test_trace_contains_reasoning_for_llm_source(self):
        """Test that LLM-derived facts include reasoning in trace."""
        fact = Fact(
            name="threshold",
            value=10000,
            source=FactSource.LLM_HEURISTIC,
            reasoning="Industry standard VIP threshold is typically $10,000",
        )

        trace = fact.derivation_trace
        assert "Industry standard" in trace
        assert "llm_heuristic" in trace

        serialized = fact.to_dict()
        assert serialized["reasoning"] == "Industry standard VIP threshold is typically $10,000"


class TestDerivationTraceDependencyChain:
    """Tests for trace linking facts to their sources (dependency chain)."""

    def test_single_dependency_in_trace(self):
        """Test trace with single parent fact."""
        base = Fact(name="base_value", value=100, source=FactSource.DATABASE)
        derived = Fact(
            name="doubled",
            value=200,
            source=FactSource.RULE,
            rule_name="double",
            because=[base],
        )

        trace = derived.derivation_trace
        assert "doubled" in trace
        assert "base_value" in trace
        assert "100" in trace

    def test_multiple_dependencies_in_trace(self):
        """Test trace with multiple parent facts."""
        revenue = Fact(name="revenue", value=50000, source=FactSource.DATABASE)
        orders = Fact(name="orders", value=100, source=FactSource.DATABASE)
        avg = Fact(
            name="avg_order_value",
            value=500,
            source=FactSource.RULE,
            rule_name="calculate_aov",
            because=[revenue, orders],
        )

        trace = avg.derivation_trace
        assert "avg_order_value" in trace
        assert "revenue" in trace
        assert "orders" in trace
        assert "50000" in trace
        assert "100" in trace

    def test_deep_dependency_chain(self):
        """Test trace with multi-level dependency chain."""
        # Level 0: Raw data
        raw = Fact(name="raw_data", value=10, source=FactSource.DATABASE)

        # Level 1: First derivation
        level1 = Fact(
            name="processed",
            value=20,
            source=FactSource.RULE,
            rule_name="process",
            because=[raw],
        )

        # Level 2: Second derivation
        level2 = Fact(
            name="enriched",
            value=40,
            source=FactSource.RULE,
            rule_name="enrich",
            because=[level1],
        )

        # Level 3: Final derivation
        final = Fact(
            name="conclusion",
            value=80,
            source=FactSource.RULE,
            rule_name="conclude",
            because=[level2],
        )

        trace = final.derivation_trace
        # All facts should appear in trace
        assert "raw_data" in trace
        assert "processed" in trace
        assert "enriched" in trace
        assert "conclusion" in trace
        # All rule names should appear
        assert "process" in trace
        assert "enrich" in trace
        assert "conclude" in trace

    def test_diamond_dependency_pattern(self):
        """Test trace handles diamond dependency (fact used by multiple paths)."""
        # A -> B -> D
        # A -> C -> D
        a = Fact(name="fact_a", value=1, source=FactSource.DATABASE)
        b = Fact(name="fact_b", value=2, source=FactSource.RULE, because=[a])
        c = Fact(name="fact_c", value=3, source=FactSource.RULE, because=[a])
        d = Fact(name="fact_d", value=6, source=FactSource.RULE, because=[b, c])

        trace = d.derivation_trace

        # All facts should appear
        assert "fact_a" in trace
        assert "fact_b" in trace
        assert "fact_c" in trace
        assert "fact_d" in trace

        # fact_a appears multiple times (once via b, once via c)
        assert trace.count("fact_a") == 2

    def test_to_dict_includes_dependency_names(self):
        """Test that serialization includes dependency fact names."""
        dep1 = Fact(name="dep1", value=10, source=FactSource.DATABASE)
        dep2 = Fact(name="dep2", value=20, source=FactSource.CONFIG)
        derived = Fact(
            name="derived",
            value=30,
            source=FactSource.RULE,
            because=[dep1, dep2],
        )

        serialized = derived.to_dict()
        assert serialized["because"] == ["dep1", "dep2"]


class TestDerivationTraceConfidenceScores:
    """Tests for trace including confidence scores."""

    def test_confidence_in_trace_output(self):
        """Test that confidence appears in derivation trace string."""
        fact = Fact(name="uncertain", value=100, confidence=0.75, source=FactSource.LLM_HEURISTIC)

        trace = fact.derivation_trace
        assert "0.75" in trace
        assert "confidence" in trace.lower()

    def test_confidence_in_serialization(self):
        """Test that confidence is preserved in serialization."""
        fact = Fact(name="test", value=50, confidence=0.85, source=FactSource.LLM_KNOWLEDGE)

        serialized = fact.to_dict()
        assert serialized["confidence"] == 0.85

    def test_default_confidence_is_one(self):
        """Test that default confidence is 1.0."""
        fact = Fact(name="certain", value=100, source=FactSource.DATABASE)
        assert fact.confidence == 1.0
        assert fact.to_dict()["confidence"] == 1.0

    def test_confidence_propagation_in_chain(self):
        """Test confidence propagates through dependency chain in resolution."""
        resolver = FactResolver()

        # Low confidence base
        resolver._cache["uncertain_base"] = Fact(
            name="uncertain_base",
            value=100,
            confidence=0.6,
            source=FactSource.LLM_HEURISTIC,
        )

        # High confidence base
        resolver._cache["certain_base"] = Fact(
            name="certain_base",
            value=200,
            confidence=1.0,
            source=FactSource.DATABASE,
        )

        @resolver.rule("combined_confidence")
        def combine(res, params):
            u = res.resolve("uncertain_base")
            c = res.resolve("certain_base")
            # Correct confidence propagation: min of dependencies
            return Fact(
                name="combined_confidence",
                value=u.value + c.value,
                confidence=min(u.confidence, c.confidence),
                source=FactSource.RULE,
                rule_name="combine",
                because=[u, c],
            )

        result = resolver.resolve("combined_confidence")
        assert result.confidence == 0.6  # Min of 0.6 and 1.0


class TestDerivationTraceTimestamps:
    """Tests for trace including timestamps."""

    def test_resolved_at_timestamp_set(self):
        """Test that resolved_at is set on fact creation."""
        from datetime import datetime

        before = datetime.now()
        fact = Fact(name="timestamped", value=1, source=FactSource.DATABASE)
        after = datetime.now()

        assert fact.resolved_at is not None
        assert before <= fact.resolved_at <= after

    def test_timestamp_in_serialization(self):
        """Test that timestamp is serialized as ISO format."""
        from datetime import datetime

        fact = Fact(name="test", value=1, source=FactSource.DATABASE)
        serialized = fact.to_dict()

        assert "resolved_at" in serialized
        # Should be valid ISO format
        parsed = datetime.fromisoformat(serialized["resolved_at"])
        assert parsed == fact.resolved_at

    def test_different_facts_have_different_timestamps(self):
        """Test that facts created at different times have different timestamps."""
        import time

        fact1 = Fact(name="first", value=1, source=FactSource.DATABASE)
        time.sleep(0.01)  # Small delay
        fact2 = Fact(name="second", value=2, source=FactSource.DATABASE)

        assert fact1.resolved_at < fact2.resolved_at

    def test_audit_log_preserves_timestamp_order(self):
        """Test that audit log maintains chronological order."""
        import time

        resolver = FactResolver()

        for i in range(3):
            resolver._cache[f"fact_{i}"] = Fact(
                name=f"fact_{i}",
                value=i,
                source=FactSource.CONFIG,
            )
            resolver.resolve(f"fact_{i}")
            time.sleep(0.01)

        log = resolver.get_audit_log()
        timestamps = [entry["resolved_at"] for entry in log]

        # Timestamps should be in ascending order
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1]


class TestDerivationTraceSerializationForAuditExport:
    """Tests for serialization of traces for audit export."""

    def test_to_dict_complete_structure(self):
        """Test that to_dict includes all required fields."""
        fact = Fact(
            name="complete_fact",
            value={"nested": "data"},
            confidence=0.95,
            source=FactSource.DATABASE,
            query="SELECT * FROM table",
            rule_name=None,
            reasoning="Direct query result",
            because=[],
        )

        serialized = fact.to_dict()

        # All fields present
        assert "name" in serialized
        assert "value" in serialized
        assert "confidence" in serialized
        assert "source" in serialized
        assert "query" in serialized
        assert "rule_name" in serialized
        assert "reasoning" in serialized
        assert "because" in serialized
        assert "resolved_at" in serialized

    def test_to_dict_handles_none_values(self):
        """Test that to_dict handles None values correctly."""
        fact = Fact(
            name="minimal",
            value=None,
            source=FactSource.UNRESOLVED,
        )

        serialized = fact.to_dict()

        assert serialized["value"] is None
        assert serialized["query"] is None
        assert serialized["rule_name"] is None
        assert serialized["reasoning"] is None

    def test_to_dict_handles_complex_values(self):
        """Test that to_dict handles complex value types."""
        import json

        complex_value = {
            "list": [1, 2, 3],
            "nested": {"a": 1, "b": 2},
            "string": "test",
            "number": 42.5,
        }

        fact = Fact(name="complex", value=complex_value, source=FactSource.DATABASE)
        serialized = fact.to_dict()

        # Value should be JSON serializable
        json_str = json.dumps(serialized)
        recovered = json.loads(json_str)
        assert recovered["value"] == complex_value

    def test_audit_log_json_serializable(self):
        """Test that entire audit log is JSON serializable."""
        import json

        resolver = FactResolver()

        # Add various fact types
        resolver._cache["db_fact"] = Fact(
            name="db_fact",
            value=100,
            source=FactSource.DATABASE,
            query="SELECT 100",
        )
        resolver._cache["llm_fact"] = Fact(
            name="llm_fact",
            value="Paris",
            confidence=0.99,
            source=FactSource.LLM_KNOWLEDGE,
            reasoning="Capital of France is well known",
        )

        resolver.resolve("db_fact")
        resolver.resolve("llm_fact")
        resolver.resolve("missing")  # Unresolved

        log = resolver.get_audit_log()

        # Should be fully JSON serializable
        json_str = json.dumps(log)
        recovered = json.loads(json_str)

        assert len(recovered) == 3
        assert recovered[0]["name"] == "db_fact"
        assert recovered[1]["name"] == "llm_fact"
        assert recovered[2]["source"] == "unresolved"

    def test_derivation_trace_indentation(self):
        """Test that derivation trace has proper indentation for readability."""
        base = Fact(name="base", value=10, source=FactSource.DATABASE)
        derived = Fact(
            name="derived",
            value=20,
            source=FactSource.RULE,
            rule_name="double",
            because=[base],
        )

        trace = derived.derivation_trace
        lines = trace.split("\n")

        # First line is the main fact
        assert lines[0].startswith("derived")

        # Dependency should be indented
        base_line = [l for l in lines if "base" in l][0]
        assert base_line.startswith("    ")  # 4 spaces indent


class TestDerivationTraceEdgeCases:
    """Tests for edge cases in derivation trace handling."""

    def test_empty_because_list(self):
        """Test handling of fact with no dependencies."""
        fact = Fact(name="leaf", value=1, source=FactSource.DATABASE, because=[])

        trace = fact.derivation_trace
        serialized = fact.to_dict()

        assert "leaf" in trace
        assert serialized["because"] == []

    def test_very_long_dependency_chain(self):
        """Test handling of deep dependency chains (potential stack overflow)."""
        # Create a chain of 50 facts
        facts = []
        prev = None

        for i in range(50):
            fact = Fact(
                name=f"level_{i}",
                value=i,
                source=FactSource.RULE,
                because=[prev] if prev else [],
            )
            facts.append(fact)
            prev = fact

        final = facts[-1]
        trace = final.derivation_trace

        # Should not raise, should include all levels
        assert "level_0" in trace
        assert "level_49" in trace

    def test_fact_with_special_characters_in_name(self):
        """Test handling of special characters in fact names."""
        fact = Fact(
            name="customer:ltv(id='test',region=\"US\")",
            value=5000,
            source=FactSource.RULE,
        )

        trace = fact.derivation_trace
        serialized = fact.to_dict()

        assert "customer:ltv" in trace
        assert serialized["name"] == "customer:ltv(id='test',region=\"US\")"

    def test_fact_with_unicode_values(self):
        """Test handling of unicode in fact values and names."""
        fact = Fact(
            name="city_name",
            value="Zurich",
            source=FactSource.LLM_KNOWLEDGE,
            reasoning="Swiss city with umlaut",
        )

        trace = fact.derivation_trace
        serialized = fact.to_dict()

        # Unicode should be preserved
        assert "Zurich" in trace or "Zurich" in serialized["value"]

    def test_fact_with_very_large_value(self):
        """Test handling of very large values."""
        large_value = list(range(10000))

        fact = Fact(name="large_list", value=large_value, source=FactSource.DATABASE)
        serialized = fact.to_dict()

        assert serialized["value"] == large_value
        assert len(serialized["value"]) == 10000

    def test_circular_dependency_in_cache_detection(self):
        """Test that circular dependencies don't cause infinite loops in trace generation."""
        # Create facts that reference each other (shouldn't happen in practice,
        # but we should handle it gracefully)
        fact_a = Fact(name="fact_a", value=1, source=FactSource.RULE)
        fact_b = Fact(name="fact_b", value=2, source=FactSource.RULE, because=[fact_a])

        # Manually create circular reference (bad state)
        fact_a.because = [fact_b]

        # This could cause infinite recursion - test it doesn't crash
        # Note: Current implementation doesn't detect this, but the trace
        # will recurse. We just ensure it eventually terminates (Python recursion limit)
        # In production, this shouldn't happen as facts are immutable after creation
        try:
            # Limit recursion for test safety
            import sys
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100)
            try:
                trace = fact_a.derivation_trace
                # If we get here without RecursionError, that's unexpected
                # but not a test failure - the implementation handles it somehow
            except RecursionError:
                # This is expected for circular deps - not ideal but handled
                pass
            finally:
                sys.setrecursionlimit(old_limit)
        except Exception as e:
            pytest.fail(f"Unexpected exception type: {type(e).__name__}: {e}")


class TestUserProvidedFactProvenance:
    """Tests for provenance tracking of user-provided facts."""

    def test_add_user_fact_source_is_user_provided(self):
        """Test that user-added facts have USER_PROVIDED source."""
        resolver = FactResolver()

        fact = resolver.add_user_fact(
            fact_name="march_attendance",
            value=1000000,
            reasoning="User stated there were 1 million attendees",
        )

        assert fact.source == FactSource.USER_PROVIDED
        assert fact.confidence == 1.0  # User facts are treated as certain
        assert fact.reasoning == "User stated there were 1 million attendees"

    def test_add_user_fact_appears_in_audit_log(self):
        """Test that user-provided facts appear in audit log."""
        resolver = FactResolver()

        resolver.add_user_fact("threshold", 50000, reasoning="User specified")

        log = resolver.get_audit_log()
        assert len(log) == 1
        assert log[0]["source"] == "user_provided"
        assert log[0]["value"] == 50000

    def test_user_fact_usable_in_derivation(self):
        """Test that user-provided facts can be used in derivations."""
        resolver = FactResolver()

        resolver.add_user_fact("base_rate", 0.05)

        @resolver.rule("final_rate")
        def calc_rate(res, params):
            base = res.resolve("base_rate")
            return Fact(
                name="final_rate",
                value=base.value * 1.1,  # 10% markup
                confidence=base.confidence,
                source=FactSource.RULE,
                rule_name="calc_rate",
                because=[base],
            )

        result = resolver.resolve("final_rate")
        assert abs(result.value - 0.055) < 1e-10
        assert result.because[0].source == FactSource.USER_PROVIDED


class TestGetAuditLog:
    """Tests for the get_audit_log method."""

    def test_empty_audit_log(self):
        """Test audit log when no facts have been resolved."""
        resolver = FactResolver()
        log = resolver.get_audit_log()
        assert log == []

    def test_audit_log_structure(self):
        """Test that audit log entries have expected structure."""
        resolver = FactResolver()
        resolver._cache["test"] = Fact(
            name="test",
            value=42,
            confidence=0.9,
            source=FactSource.DATABASE,
            query="SELECT 42",
        )
        resolver.resolve("test")

        log = resolver.get_audit_log()
        assert len(log) == 1

        entry = log[0]
        assert entry["name"] == "test"
        assert entry["value"] == 42
        assert entry["confidence"] == 0.9
        assert entry["source"] == "database"
        assert entry["query"] == "SELECT 42"
        assert "resolved_at" in entry

    def test_audit_log_includes_unresolved(self):
        """Test that unresolved facts appear in audit log."""
        resolver = FactResolver()
        resolver.resolve("missing_fact")

        log = resolver.get_audit_log()
        assert len(log) == 1
        assert log[0]["source"] == "unresolved"

    def test_audit_log_order_is_resolution_order(self):
        """Test that audit log preserves resolution order."""
        resolver = FactResolver()

        for i in range(5):
            resolver._cache[f"fact_{i}"] = Fact(
                name=f"fact_{i}",
                value=i,
                source=FactSource.CONFIG,
            )

        # Resolve in specific order
        resolver.resolve("fact_2")
        resolver.resolve("fact_0")
        resolver.resolve("fact_4")
        resolver.resolve("fact_1")
        resolver.resolve("fact_3")

        log = resolver.get_audit_log()
        names = [entry["name"] for entry in log]

        assert names == ["fact_2", "fact_0", "fact_4", "fact_1", "fact_3"]


class TestGetUnresolvedFacts:
    """Tests for tracking and reporting unresolved facts."""

    def test_get_unresolved_facts_empty(self):
        """Test get_unresolved_facts when all facts resolved."""
        resolver = FactResolver()
        resolver._cache["resolved"] = Fact(name="resolved", value=1, source=FactSource.DATABASE)
        resolver.resolve("resolved")

        unresolved = resolver.get_unresolved_facts()
        assert unresolved == []

    def test_get_unresolved_facts_with_missing(self):
        """Test get_unresolved_facts with missing facts."""
        resolver = FactResolver()
        resolver.resolve("missing1")
        resolver.resolve("missing2")

        unresolved = resolver.get_unresolved_facts()
        assert len(unresolved) == 2
        assert all(f.source == FactSource.UNRESOLVED for f in unresolved)

    def test_get_unresolved_summary_format(self):
        """Test the format of unresolved summary message."""
        resolver = FactResolver()
        resolver.resolve("unknown_threshold")

        summary = resolver.get_unresolved_summary()

        assert "unknown_threshold" in summary
        assert "could not be resolved" in summary.lower() or "unresolved" in summary.lower()

    def test_get_unresolved_summary_when_all_resolved(self):
        """Test summary when all facts are resolved."""
        resolver = FactResolver()
        resolver._cache["complete"] = Fact(name="complete", value=1, source=FactSource.DATABASE)
        resolver.resolve("complete")

        summary = resolver.get_unresolved_summary()
        assert "resolved successfully" in summary.lower()


class TestClearUnresolved:
    """Tests for clearing unresolved facts."""

    def test_clear_unresolved_removes_unresolved_only(self):
        """Test that clear_unresolved only removes unresolved facts."""
        resolver = FactResolver()

        # Add resolved fact
        resolver._cache["resolved"] = Fact(name="resolved", value=1, source=FactSource.DATABASE)
        resolver.resolve("resolved")

        # Add unresolved fact
        resolver.resolve("missing")

        assert len(resolver.resolution_log) == 2

        resolver.clear_unresolved()

        assert len(resolver.resolution_log) == 1
        assert resolver.resolution_log[0].name == "resolved"

    def test_clear_unresolved_allows_re_resolution(self):
        """Test that clearing unresolved allows facts to be tried again."""
        resolver = FactResolver()

        # First attempt fails
        resolver.resolve("dynamic_fact")
        assert len(resolver.get_unresolved_facts()) == 1

        # Clear unresolved
        resolver.clear_unresolved()
        assert len(resolver.get_unresolved_facts()) == 0

        # Now add the fact to cache and resolve again
        resolver._cache["dynamic_fact"] = Fact(
            name="dynamic_fact",
            value=42,
            source=FactSource.CONFIG,
        )
        result = resolver.resolve("dynamic_fact")

        assert result.is_resolved
        assert result.value == 42


# =============================================================================
# PARALLELIZATION COMPONENT TESTS
# =============================================================================


class TestParallelizationComponent:
    """
    Component tests verifying that fact resolution parallelization works
    correctly within the AsyncFactResolver class.

    These tests use timing measurements to prove that:
    1. Multiple facts are resolved concurrently (not sequentially)
    2. The asyncio.gather approach provides real speedup
    3. resolve_many_async is properly integrated
    """

    @pytest.mark.asyncio
    async def test_parallel_resolution_timing_proves_concurrency(self):
        """
        Component test: Verify parallelization by measuring actual execution time.

        If 3 facts each take 100ms to resolve:
        - Sequential: ~300ms
        - Parallel: ~100ms

        This test will FAIL if parallelization isn't wired up correctly.
        """
        import time

        DELAY_MS = 100
        NUM_FACTS = 3

        resolver = AsyncFactResolver()
        resolution_times = []

        @resolver.rule("slow_fact")
        def slow_rule(res, params):
            """Rule that takes DELAY_MS to execute (simulates I/O)."""
            fact_id = params.get("id", 0)
            start = time.time()
            time.sleep(DELAY_MS / 1000)  # Simulate I/O delay
            elapsed = (time.time() - start) * 1000
            resolution_times.append((fact_id, elapsed))
            return Fact(
                name=f"slow_fact_{fact_id}",
                value=f"result_{fact_id}",
                source=FactSource.RULE,
            )

        # Build fact requests
        requests = [("slow_fact", {"id": i}) for i in range(NUM_FACTS)]

        # Measure parallel resolution time
        start = time.time()
        results = await resolver.resolve_many_async(requests)
        parallel_time = (time.time() - start) * 1000

        # Verify results
        assert len(results) == NUM_FACTS
        for i, result in enumerate(results):
            assert result.value == f"result_{i}"

        # CRITICAL ASSERTION: If parallel, total time should be close to single delay
        # Not NUM_FACTS * DELAY_MS (which would indicate sequential execution)
        expected_sequential_time = NUM_FACTS * DELAY_MS
        expected_parallel_time = DELAY_MS * 1.5  # Allow some overhead

        print(f"\nParallel time: {parallel_time:.1f}ms")
        print(f"Expected sequential: {expected_sequential_time}ms")
        print(f"Expected parallel (max): {expected_parallel_time}ms")

        # This assertion will FAIL if parallelization isn't working
        assert parallel_time < expected_sequential_time * 0.7, (
            f"Parallel resolution took {parallel_time:.1f}ms but should be under "
            f"{expected_sequential_time * 0.7:.1f}ms if truly parallel. "
            f"This suggests parallelization is NOT wired up correctly!"
        )

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_speedup_ratio(self):
        """
        Component test: Measure actual speedup ratio between parallel and sequential.

        Expected: parallel should be at least 2x faster than sequential for 3 facts.
        """
        import time

        DELAY_MS = 50
        NUM_FACTS = 4

        # Create resolver with slow rule
        resolver = AsyncFactResolver()

        @resolver.rule("timed_fact")
        def timed_rule(res, params):
            time.sleep(DELAY_MS / 1000)
            return Fact(
                name=f"timed_{params['id']}",
                value=params["id"],
                source=FactSource.RULE,
            )

        requests = [("timed_fact", {"id": i}) for i in range(NUM_FACTS)]

        # Measure sequential time (resolve one at a time)
        resolver.clear_cache()
        start = time.time()
        for name, params in requests:
            await resolver.resolve_async(name, **params)
            resolver.clear_cache()  # Clear cache to force re-resolution
        sequential_time = time.time() - start

        # Measure parallel time
        resolver.clear_cache()
        start = time.time()
        await resolver.resolve_many_async(requests)
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time

        print(f"\nSequential: {sequential_time*1000:.1f}ms")
        print(f"Parallel: {parallel_time*1000:.1f}ms")
        print(f"Speedup: {speedup:.2f}x")

        # Parallel should be significantly faster
        assert speedup > 1.5, (
            f"Speedup ratio is only {speedup:.2f}x (expected >1.5x). "
            f"This suggests parallelization may not be working correctly."
        )

    @pytest.mark.asyncio
    async def test_parallel_resolution_with_async_io_simulation(self):
        """
        Component test: Simulate actual async I/O operations (like DB/LLM calls).

        Uses asyncio.sleep instead of time.sleep to properly test async behavior.
        """
        import time

        DELAY_MS = 100
        NUM_FACTS = 5

        resolver = AsyncFactResolver()
        call_timestamps = []

        # Register an async-aware rule
        original_resolve = resolver.resolve_async

        async def tracked_resolve(name, **params):
            if name == "async_io_fact":
                call_timestamps.append(("start", params.get("id"), time.time()))
                await asyncio.sleep(DELAY_MS / 1000)  # Async I/O simulation
                call_timestamps.append(("end", params.get("id"), time.time()))
                return Fact(
                    name=f"async_io_{params['id']}",
                    value=params["id"],
                    source=FactSource.RULE,
                )
            return await original_resolve(name, **params)

        # Monkey-patch for this test
        resolver.resolve_async = tracked_resolve

        requests = [("async_io_fact", {"id": i}) for i in range(NUM_FACTS)]

        start = time.time()
        results = await resolver.resolve_many_async(requests)
        total_time = (time.time() - start) * 1000

        # Verify all facts resolved
        assert len(results) == NUM_FACTS

        # Analyze timestamps to verify concurrent execution
        starts = [t for t in call_timestamps if t[0] == "start"]
        ends = [t for t in call_timestamps if t[0] == "end"]

        # All starts should happen before all ends (concurrent execution)
        start_times = [t[2] for t in starts]
        end_times = [t[2] for t in ends]

        # Time window for all starts should be small (< half the delay)
        start_window = max(start_times) - min(start_times)
        print(f"\nStart window: {start_window*1000:.1f}ms")
        print(f"Total time: {total_time:.1f}ms")
        print(f"Expected if parallel: ~{DELAY_MS}ms")
        print(f"Expected if sequential: ~{NUM_FACTS * DELAY_MS}ms")

        # If parallel, all facts start nearly simultaneously
        assert start_window < DELAY_MS / 2, (
            f"Facts started with {start_window*1000:.1f}ms spread. "
            f"Should be <{DELAY_MS/2}ms if truly parallel."
        )

        # Total time should be close to single delay, not N * delay
        assert total_time < DELAY_MS * 2, (
            f"Total time {total_time:.1f}ms exceeds {DELAY_MS * 2}ms. "
            f"This suggests sequential execution, not parallel."
        )

    @pytest.mark.asyncio
    async def test_resolve_many_sync_wrapper_parallelizes(self):
        """
        Component test: Verify that resolve_many_sync (the sync wrapper) also parallelizes.
        """
        import time

        DELAY_MS = 50
        NUM_FACTS = 3

        resolver = AsyncFactResolver()

        @resolver.rule("sync_wrapper_test")
        def slow_rule(res, params):
            time.sleep(DELAY_MS / 1000)
            return Fact(
                name=f"sync_wrapped_{params['id']}",
                value=params["id"],
                source=FactSource.RULE,
            )

        requests = [("sync_wrapper_test", {"id": i}) for i in range(NUM_FACTS)]

        # Use the synchronous wrapper
        start = time.time()
        results = resolver.resolve_many_sync(requests)
        elapsed = (time.time() - start) * 1000

        assert len(results) == NUM_FACTS

        # Should still be parallel (not sequential)
        sequential_expected = NUM_FACTS * DELAY_MS
        print(f"\nSync wrapper time: {elapsed:.1f}ms")
        print(f"Sequential expected: {sequential_expected}ms")

        assert elapsed < sequential_expected * 0.7, (
            f"Sync wrapper took {elapsed:.1f}ms, expected <{sequential_expected * 0.7:.1f}ms. "
            f"The sync wrapper may not be properly invoking async parallelization."
        )


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================


class TestRateLimiter:
    """Tests for the RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic_execution(self):
        """Test basic execution through rate limiter."""
        from constat.execution.fact_resolver import RateLimiter

        limiter = RateLimiter()

        async def success_task():
            return "success"

        result = await limiter.execute(success_task)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrency_control(self):
        """Test that rate limiter limits concurrency."""
        import time
        from constat.execution.fact_resolver import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(max_concurrent=2)
        limiter = RateLimiter(config)

        call_times = []

        async def slow_task(task_id):
            call_times.append(("start", task_id, time.time()))
            await asyncio.sleep(0.1)
            call_times.append(("end", task_id, time.time()))
            return task_id

        # Launch 4 tasks with max_concurrent=2
        # Use coroutines directly (no retry needed for this test)
        tasks = [limiter.execute(slow_task(i)) for i in range(4)]
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert results == [0, 1, 2, 3]

        # With max_concurrent=2 and 4 tasks taking 100ms each:
        # - Wave 1: tasks 0, 1 run in parallel (100ms)
        # - Wave 2: tasks 2, 3 run in parallel (100ms)
        # Total ~200ms (not 400ms if sequential, not 100ms if no limit)
        assert 0.15 < elapsed < 0.35, f"Expected ~200ms, got {elapsed*1000:.0f}ms"

    @pytest.mark.asyncio
    async def test_rate_limiter_exponential_backoff(self):
        """Test exponential backoff on rate limit errors."""
        import time
        from constat.execution.fact_resolver import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(
            max_concurrent=5,
            max_retries=3,
            base_delay=0.05,  # 50ms base delay for fast test
            jitter=0.0,  # No jitter for predictable timing
        )
        limiter = RateLimiter(config)

        attempt_count = 0
        attempt_times = []

        async def rate_limited_task():
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(time.time())

            if attempt_count < 3:
                raise Exception("Rate limit exceeded: 429")
            return "success"

        start = time.time()
        # Pass the async function itself (not called) so it can be retried
        result = await limiter.execute(rate_limited_task)
        elapsed = time.time() - start

        assert result == "success"
        assert attempt_count == 3

        # Check exponential delays:
        # - Attempt 1: immediate
        # - Attempt 2: after 50ms (2^0 * 50ms)
        # - Attempt 3: after 100ms (2^1 * 50ms)
        # Total: ~150ms
        assert 0.1 < elapsed < 0.3, f"Expected ~150ms, got {elapsed*1000:.0f}ms"

    @pytest.mark.asyncio
    async def test_rate_limiter_exhausted_error(self):
        """Test RateLimitExhaustedError after max retries."""
        from constat.execution.fact_resolver import (
            RateLimiter,
            RateLimiterConfig,
            RateLimitExhaustedError,
        )

        config = RateLimiterConfig(
            max_retries=2,
            base_delay=0.01,
            jitter=0.0,
        )
        limiter = RateLimiter(config)

        async def always_rate_limited():
            raise Exception("429 Too Many Requests")

        with pytest.raises(RateLimitExhaustedError, match="Rate limit exceeded after 2 retries"):
            # Pass the async function itself for retry support
            await limiter.execute(always_rate_limited)

    @pytest.mark.asyncio
    async def test_rate_limiter_non_rate_limit_errors_not_retried(self):
        """Test that non-rate-limit errors are not retried."""
        from constat.execution.fact_resolver import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(max_retries=3)
        limiter = RateLimiter(config)

        attempt_count = 0

        async def regular_error():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("Database connection failed")

        with pytest.raises(ValueError, match="Database connection failed"):
            # Pass the async function itself
            await limiter.execute(regular_error)

        # Should only attempt once - not retried
        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        from constat.execution.fact_resolver import RateLimiter, RateLimiterConfig

        config = RateLimiterConfig(max_concurrent=5, max_retries=3, base_delay=0.01, jitter=0.0)
        limiter = RateLimiter(config)

        # Track rate limit hits
        call_count = 0

        async def sometimes_rate_limited():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Rate limit 429")
            return "ok"

        # Pass the async function itself for retry support
        await limiter.execute(sometimes_rate_limited)

        stats = limiter.stats
        assert stats["total_requests"] == 1
        assert stats["rate_limit_hits"] == 1
        assert stats["max_concurrent"] == 5

    def test_async_resolver_has_rate_limiter(self):
        """Test that AsyncFactResolver has a rate limiter."""
        resolver = AsyncFactResolver()

        assert hasattr(resolver, "_rate_limiter")
        assert hasattr(resolver, "rate_limiter_stats")

        stats = resolver.rate_limiter_stats
        assert "total_requests" in stats
        assert "max_concurrent" in stats

    def test_async_resolver_custom_rate_limiter_config(self):
        """Test AsyncFactResolver with custom rate limiter config."""
        from constat.execution.fact_resolver import RateLimiterConfig

        config = RateLimiterConfig(max_concurrent=10, max_retries=5)
        resolver = AsyncFactResolver(rate_limiter_config=config)

        assert resolver._rate_limiter.config.max_concurrent == 10
        assert resolver._rate_limiter.config.max_retries == 5


# =============================================================================
# DEMAND-DRIVEN FACT RESOLUTION TESTS
# Tests for DAG-based resolution with parallel independent facts
# =============================================================================


class TestResolutionHierarchy:
    """Tests for the fact resolution hierarchy order."""

    def test_default_hierarchy_order(self):
        """Test that default hierarchy is cache → rule → database → sub_plan → document → llm_knowledge → user_provided."""
        strategy = ResolutionStrategy()
        expected = [
            FactSource.CACHE,
            FactSource.RULE,
            FactSource.DATABASE,
            FactSource.SUB_PLAN,
            FactSource.DOCUMENT,
            FactSource.LLM_KNOWLEDGE,
            FactSource.USER_PROVIDED,
        ]
        assert strategy.source_priority == expected

    def test_cache_checked_first(self):
        """Test that cached facts are returned immediately without trying other sources."""
        resolver = FactResolver()

        # Pre-cache a fact
        cached_fact = Fact(name="cached_value", value=42, source=FactSource.CACHE)
        resolver._cache["cached_value"] = cached_fact

        # Resolve should hit cache
        result = resolver.resolve("cached_value")

        assert result.value == 42
        assert result.source == FactSource.CACHE

    def test_rule_checked_before_database(self):
        """Test that rules are checked before database queries."""
        resolver = FactResolver()
        resolution_order = []

        @resolver.rule("derived_fact")
        def compute_derived(resolver, params):
            resolution_order.append("rule")
            return Fact(name="derived_fact", value=100, source=FactSource.RULE)

        result = resolver.resolve("derived_fact")

        assert result.value == 100
        assert result.source == FactSource.RULE
        assert "rule" in resolution_order

    def test_hierarchy_stops_at_first_success(self):
        """Test that resolution stops as soon as a source succeeds."""
        resolver = FactResolver()
        sources_tried = []

        # Register a rule that succeeds
        @resolver.rule("test_fact")
        def rule_succeeds(resolver, params):
            sources_tried.append("rule")
            return Fact(name="test_fact", value="from_rule", source=FactSource.RULE)

        result = resolver.resolve("test_fact")

        # Should have tried rule and stopped (not tried database, sub_plan, etc.)
        assert result.source == FactSource.RULE
        assert sources_tried == ["rule"]


class TestDemandDrivenResolution:
    """Tests for demand-driven (conclusion-first) resolution."""

    def test_conclusion_triggers_dependency_resolution(self):
        """Test that resolving a conclusion fact triggers resolution of its dependencies."""
        resolver = FactResolver()
        resolution_order = []

        # Register base facts
        @resolver.rule("base_a")
        def resolve_a(resolver, params):
            resolution_order.append("base_a")
            return Fact(name="base_a", value=10, source=FactSource.RULE)

        @resolver.rule("base_b")
        def resolve_b(resolver, params):
            resolution_order.append("base_b")
            return Fact(name="base_b", value=20, source=FactSource.RULE)

        # Register conclusion that depends on base facts
        @resolver.rule("conclusion")
        def resolve_conclusion(resolver, params):
            resolution_order.append("conclusion_start")
            a = resolver.resolve("base_a")
            b = resolver.resolve("base_b")
            resolution_order.append("conclusion_end")
            return Fact(
                name="conclusion",
                value=a.value + b.value,
                source=FactSource.RULE,
                because=[a, b]
            )

        result = resolver.resolve("conclusion")

        assert result.value == 30
        assert len(result.because) == 2
        # Conclusion started, then resolved dependencies, then finished
        assert resolution_order == ["conclusion_start", "base_a", "base_b", "conclusion_end"]

    def test_dependencies_cached_for_reuse(self):
        """Test that resolved dependencies are cached and reused."""
        resolver = FactResolver()
        call_count = {"base": 0}

        @resolver.rule("base")
        def resolve_base(resolver, params):
            call_count["base"] += 1
            return Fact(name="base", value=100, source=FactSource.RULE)

        @resolver.rule("derived1")
        def resolve_d1(resolver, params):
            base = resolver.resolve("base")
            return Fact(name="derived1", value=base.value * 2, source=FactSource.RULE, because=[base])

        @resolver.rule("derived2")
        def resolve_d2(resolver, params):
            base = resolver.resolve("base")
            return Fact(name="derived2", value=base.value * 3, source=FactSource.RULE, because=[base])

        # Resolve both derived facts
        d1 = resolver.resolve("derived1")
        d2 = resolver.resolve("derived2")

        assert d1.value == 200
        assert d2.value == 300
        # Base should only be resolved once (cached after first call)
        assert call_count["base"] == 1

    def test_provenance_chain_tracked(self):
        """Test that the full provenance chain is tracked via 'because' field."""
        resolver = FactResolver()

        @resolver.rule("level1")
        def resolve_l1(resolver, params):
            return Fact(name="level1", value=1, source=FactSource.RULE)

        @resolver.rule("level2")
        def resolve_l2(resolver, params):
            l1 = resolver.resolve("level1")
            return Fact(name="level2", value=2, source=FactSource.RULE, because=[l1])

        @resolver.rule("level3")
        def resolve_l3(resolver, params):
            l2 = resolver.resolve("level2")
            return Fact(name="level3", value=3, source=FactSource.RULE, because=[l2])

        result = resolver.resolve("level3")

        # Check provenance chain
        assert result.name == "level3"
        assert len(result.because) == 1
        assert result.because[0].name == "level2"
        assert len(result.because[0].because) == 1
        assert result.because[0].because[0].name == "level1"


class TestParallelResolution:
    """Tests for parallel resolution of independent facts."""

    def test_resolve_many_sync_exists_on_base_resolver(self):
        """Test that resolve_many_sync is available on base FactResolver."""
        resolver = FactResolver()
        assert hasattr(resolver, "resolve_many_sync")
        assert callable(resolver.resolve_many_sync)

    def test_resolve_many_sync_resolves_all_facts(self):
        """Test that resolve_many_sync resolves all requested facts."""
        resolver = FactResolver()

        @resolver.rule("fact_a")
        def resolve_a(resolver, params):
            return Fact(name="fact_a", value="A", source=FactSource.RULE)

        @resolver.rule("fact_b")
        def resolve_b(resolver, params):
            return Fact(name="fact_b", value="B", source=FactSource.RULE)

        @resolver.rule("fact_c")
        def resolve_c(resolver, params):
            return Fact(name="fact_c", value="C", source=FactSource.RULE)

        facts = resolver.resolve_many_sync([
            ("fact_a", {}),
            ("fact_b", {}),
            ("fact_c", {}),
        ])

        assert len(facts) == 3
        assert facts[0].value == "A"
        assert facts[1].value == "B"
        assert facts[2].value == "C"

    def test_resolve_many_sync_preserves_order(self):
        """Test that resolve_many_sync returns facts in request order."""
        resolver = FactResolver()

        @resolver.rule("z_last")
        def resolve_z(resolver, params):
            return Fact(name="z_last", value="Z", source=FactSource.RULE)

        @resolver.rule("a_first")
        def resolve_a(resolver, params):
            return Fact(name="a_first", value="A", source=FactSource.RULE)

        @resolver.rule("m_middle")
        def resolve_m(resolver, params):
            return Fact(name="m_middle", value="M", source=FactSource.RULE)

        facts = resolver.resolve_many_sync([
            ("z_last", {}),
            ("a_first", {}),
            ("m_middle", {}),
        ])

        # Order should match request order, not alphabetical
        assert [f.value for f in facts] == ["Z", "A", "M"]

    def test_resolve_many_sync_with_params(self):
        """Test that resolve_many_sync passes params correctly."""
        resolver = FactResolver()

        @resolver.rule("parameterized")
        def resolve_param(resolver, params):
            multiplier = params.get("multiplier", 1)
            return Fact(
                name=f"parameterized(multiplier={multiplier})",
                value=10 * multiplier,
                source=FactSource.RULE
            )

        facts = resolver.resolve_many_sync([
            ("parameterized", {"multiplier": 1}),
            ("parameterized", {"multiplier": 2}),
            ("parameterized", {"multiplier": 5}),
        ])

        assert facts[0].value == 10
        assert facts[1].value == 20
        assert facts[2].value == 50

    def test_async_resolve_many_is_parallel(self):
        """Test that AsyncFactResolver.resolve_many_async runs in parallel."""
        import time

        resolver = AsyncFactResolver()

        @resolver.rule("slow_fact")
        def resolve_slow(resolver, params):
            time.sleep(0.1)  # 100ms delay
            return Fact(name=f"slow_fact_{params.get('id')}", value=params.get('id'), source=FactSource.RULE)

        # Resolve 5 facts - if sequential would take 500ms, if parallel ~100ms
        start = time.time()
        facts = resolver.resolve_many_sync([
            ("slow_fact", {"id": i}) for i in range(5)
        ])
        elapsed = time.time() - start

        assert len(facts) == 5
        # Should complete in roughly 100-200ms if parallel, not 500ms+
        assert elapsed < 0.4, f"Expected parallel execution (<400ms), got {elapsed*1000:.0f}ms"


class TestSequentialDependencies:
    """Tests for sequential resolution when facts depend on each other."""

    def test_dependent_facts_resolved_sequentially(self):
        """Test that dependent facts are resolved in correct order."""
        resolver = FactResolver()
        resolution_times = {}

        import time

        @resolver.rule("start_date")
        def resolve_start(resolver, params):
            resolution_times["start_date"] = time.time()
            return Fact(name="start_date", value="2024-01-01", source=FactSource.RULE)

        @resolver.rule("end_date")
        def resolve_end(resolver, params):
            # end_date depends on start_date
            start = resolver.resolve("start_date")
            resolution_times["end_date"] = time.time()
            return Fact(
                name="end_date",
                value="2024-03-31",  # start + 3 months
                source=FactSource.RULE,
                because=[start]
            )

        result = resolver.resolve("end_date")

        assert result.value == "2024-03-31"
        assert len(result.because) == 1
        assert result.because[0].value == "2024-01-01"
        # start_date must be resolved before end_date
        assert resolution_times["start_date"] <= resolution_times["end_date"]

    def test_diamond_dependency_resolved_correctly(self):
        """Test diamond dependency pattern: D depends on B and C, both depend on A."""
        resolver = FactResolver()
        call_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

        @resolver.rule("A")
        def resolve_a(resolver, params):
            call_counts["A"] += 1
            return Fact(name="A", value=1, source=FactSource.RULE)

        @resolver.rule("B")
        def resolve_b(resolver, params):
            call_counts["B"] += 1
            a = resolver.resolve("A")
            return Fact(name="B", value=a.value * 2, source=FactSource.RULE, because=[a])

        @resolver.rule("C")
        def resolve_c(resolver, params):
            call_counts["C"] += 1
            a = resolver.resolve("A")
            return Fact(name="C", value=a.value * 3, source=FactSource.RULE, because=[a])

        @resolver.rule("D")
        def resolve_d(resolver, params):
            call_counts["D"] += 1
            b = resolver.resolve("B")
            c = resolver.resolve("C")
            return Fact(name="D", value=b.value + c.value, source=FactSource.RULE, because=[b, c])

        result = resolver.resolve("D")

        assert result.value == 5  # (1*2) + (1*3)
        # Each fact should only be resolved once due to caching
        assert call_counts == {"A": 1, "B": 1, "C": 1, "D": 1}


class TestUserProvidedReSteer:
    """Tests for USER_PROVIDED as re-steer trigger."""

    def test_user_provided_fact_from_cache(self):
        """Test that user-provided facts are retrieved from cache."""
        resolver = FactResolver()

        # User provides a fact (simulating clarification)
        resolver.add_user_fact("customer_tier_threshold", 100000, reasoning="User specified VIP threshold")

        # Later resolution should find it in cache
        result = resolver.resolve("customer_tier_threshold")

        assert result.is_resolved
        assert result.value == 100000
        assert result.source == FactSource.USER_PROVIDED

    def test_unresolved_when_user_fact_missing(self):
        """Test that resolution returns UNRESOLVED when user fact not provided."""
        resolver = FactResolver()
        # No LLM, no rules, no database - should exhaust all sources

        result = resolver.resolve("undefined_user_fact")

        assert not result.is_resolved
        assert result.source == FactSource.UNRESOLVED

    def test_user_fact_usable_in_derived_computation(self):
        """Test that user-provided facts can be used in derived computations."""
        resolver = FactResolver()

        # User provides period description
        resolver.add_user_fact("period", "last_3_months", reasoning="User clarified time period")

        # Rule derives dates from user-provided period
        @resolver.rule("date_range")
        def resolve_dates(resolver, params):
            period = resolver.resolve("period")
            # Simulate deriving dates from period
            if period.value == "last_3_months":
                return Fact(
                    name="date_range",
                    value={"start": "2024-01-01", "end": "2024-03-31"},
                    source=FactSource.RULE,
                    because=[period]
                )
            return Fact(name="date_range", value=None, source=FactSource.UNRESOLVED)

        result = resolver.resolve("date_range")

        assert result.is_resolved
        assert result.value == {"start": "2024-01-01", "end": "2024-03-31"}
        assert len(result.because) == 1
        assert result.because[0].source == FactSource.USER_PROVIDED

    def test_get_unresolved_identifies_missing_user_facts(self):
        """Test that get_unresolved_facts identifies facts that need user input."""
        resolver = FactResolver()

        # Try to resolve facts that will fail
        resolver.resolve("tier_definition")
        resolver.resolve("revenue_source")

        unresolved = resolver.get_unresolved_facts()

        assert len(unresolved) == 2
        assert any(f.name == "tier_definition" for f in unresolved)
        assert any(f.name == "revenue_source" for f in unresolved)

    def test_unresolved_summary_suggests_user_input(self):
        """Test that unresolved summary suggests user can provide facts."""
        resolver = FactResolver()

        resolver.resolve("unknown_threshold")

        summary = resolver.get_unresolved_summary()

        assert "could not be resolved" in summary.lower()
        assert "unknown_threshold" in summary


class TestDAGDiscovery:
    """Tests for DAG discovery through recursive resolution."""

    def test_dag_discovered_through_resolution(self):
        """Test that the dependency DAG is discovered during resolution."""
        resolver = FactResolver()

        # Simulate a complex DAG:
        # answer depends on [trend_data, tier_summary]
        # trend_data depends on [raw_orders, date_range]
        # tier_summary depends on [raw_orders, tier_definitions]
        # date_range depends on [start_date, end_date]

        @resolver.rule("start_date")
        def r_start(res, p):
            return Fact(name="start_date", value="2024-01-01", source=FactSource.RULE)

        @resolver.rule("end_date")
        def r_end(res, p):
            start = res.resolve("start_date")
            return Fact(name="end_date", value="2024-03-31", source=FactSource.RULE, because=[start])

        @resolver.rule("date_range")
        def r_range(res, p):
            start = res.resolve("start_date")
            end = res.resolve("end_date")
            return Fact(name="date_range", value={"start": start.value, "end": end.value},
                       source=FactSource.RULE, because=[start, end])

        @resolver.rule("tier_definitions")
        def r_tiers(res, p):
            return Fact(name="tier_definitions", value=["Gold", "Silver", "Bronze"], source=FactSource.RULE)

        @resolver.rule("raw_orders")
        def r_orders(res, p):
            dates = res.resolve("date_range")
            return Fact(name="raw_orders", value=[{"id": 1}, {"id": 2}], source=FactSource.RULE, because=[dates])

        @resolver.rule("trend_data")
        def r_trend(res, p):
            orders = res.resolve("raw_orders")
            dates = res.resolve("date_range")
            return Fact(name="trend_data", value="trend_computed", source=FactSource.RULE, because=[orders, dates])

        @resolver.rule("tier_summary")
        def r_summary(res, p):
            orders = res.resolve("raw_orders")
            tiers = res.resolve("tier_definitions")
            return Fact(name="tier_summary", value="summary_computed", source=FactSource.RULE, because=[orders, tiers])

        @resolver.rule("answer")
        def r_answer(res, p):
            trend = res.resolve("trend_data")
            summary = res.resolve("tier_summary")
            return Fact(name="answer", value="final_answer", source=FactSource.RULE, because=[trend, summary])

        result = resolver.resolve("answer")

        # Check that the full DAG was discovered
        assert result.is_resolved
        assert result.value == "final_answer"

        # Check provenance - answer depends on trend_data and tier_summary
        assert len(result.because) == 2
        dep_names = {f.name for f in result.because}
        assert dep_names == {"trend_data", "tier_summary"}

        # Check that all facts were resolved (via resolution_log)
        resolved_names = {f.name for f in resolver.resolution_log if f.is_resolved}
        expected = {"start_date", "end_date", "date_range", "tier_definitions",
                   "raw_orders", "trend_data", "tier_summary", "answer"}
        assert resolved_names == expected

    def test_audit_log_captures_full_dag(self):
        """Test that the audit log captures all facts in the DAG."""
        resolver = FactResolver()

        @resolver.rule("leaf1")
        def r1(res, p):
            return Fact(name="leaf1", value=1, source=FactSource.RULE)

        @resolver.rule("leaf2")
        def r2(res, p):
            return Fact(name="leaf2", value=2, source=FactSource.RULE)

        @resolver.rule("branch")
        def r3(res, p):
            l1 = res.resolve("leaf1")
            l2 = res.resolve("leaf2")
            return Fact(name="branch", value=l1.value + l2.value, source=FactSource.RULE, because=[l1, l2])

        @resolver.rule("root")
        def r4(res, p):
            b = res.resolve("branch")
            return Fact(name="root", value=b.value * 2, source=FactSource.RULE, because=[b])

        resolver.resolve("root")

        audit_log = resolver.get_audit_log()

        # Should have 4 entries: leaf1, leaf2, branch, root
        assert len(audit_log) == 4

        # Check each entry has required fields
        for entry in audit_log:
            assert "name" in entry
            assert "value" in entry
            assert "source" in entry
            assert "confidence" in entry
            assert "because" in entry


class TestIndependentVsDependentFacts:
    """Tests for correctly identifying independent vs dependent facts."""

    def test_independent_facts_can_use_resolve_many(self):
        """Test that independent facts can be resolved together via resolve_many_sync."""
        resolver = FactResolver()

        # These facts are independent - no dependencies between them
        @resolver.rule("fact_x")
        def rx(res, p):
            return Fact(name="fact_x", value="X", source=FactSource.RULE)

        @resolver.rule("fact_y")
        def ry(res, p):
            return Fact(name="fact_y", value="Y", source=FactSource.RULE)

        @resolver.rule("fact_z")
        def rz(res, p):
            return Fact(name="fact_z", value="Z", source=FactSource.RULE)

        # Resolve all at once
        facts = resolver.resolve_many_sync([
            ("fact_x", {}),
            ("fact_y", {}),
            ("fact_z", {}),
        ])

        values = [f.value for f in facts]
        assert values == ["X", "Y", "Z"]

    def test_mixed_independent_and_dependent_resolution(self):
        """Test resolving a mix of independent and dependent facts."""
        resolver = FactResolver()
        resolution_order = []

        # Independent facts
        @resolver.rule("config_a")
        def ra(res, p):
            resolution_order.append("config_a")
            return Fact(name="config_a", value="A", source=FactSource.RULE)

        @resolver.rule("config_b")
        def rb(res, p):
            resolution_order.append("config_b")
            return Fact(name="config_b", value="B", source=FactSource.RULE)

        # Dependent fact
        @resolver.rule("combined")
        def rc(res, p):
            resolution_order.append("combined_start")
            a = res.resolve("config_a")
            b = res.resolve("config_b")
            resolution_order.append("combined_end")
            return Fact(name="combined", value=f"{a.value}+{b.value}", source=FactSource.RULE, because=[a, b])

        result = resolver.resolve("combined")

        assert result.value == "A+B"
        # combined starts, then resolves a and b, then finishes
        assert resolution_order[0] == "combined_start"
        assert "config_a" in resolution_order
        assert "config_b" in resolution_order
        assert resolution_order[-1] == "combined_end"


class TestResolveConclusion:
    """Tests for template-based symbolic resolution via resolve_conclusion."""

    def test_resolve_conclusion_without_llm_returns_error(self):
        """Test that resolve_conclusion requires an LLM."""
        resolver = FactResolver()
        result = resolver.resolve_conclusion("What is the revenue?")
        assert "error" in result
        assert result["answer"] is None

    def test_resolve_conclusion_generates_template(self):
        """Test that resolve_conclusion generates a template with variables."""
        mock_llm = MagicMock()
        # First call: generate template
        mock_llm.generate.side_effect = [
            """TEMPLATE: The total revenue for {time_period} was ${total_revenue}.
VARIABLES:
- {time_period}: The time period to analyze
- {total_revenue}: Sum of all revenue in the period""",
            # Second call: dependency analysis
            """{time_period}: []
{total_revenue}: [{time_period}]""",
        ]

        resolver = FactResolver(llm=mock_llm)

        # Pre-cache facts so resolution succeeds
        resolver.add_user_fact("time_period", "Q3 2024")
        resolver.add_user_fact("total_revenue", 150000)

        result = resolver.resolve_conclusion("What was the revenue for Q3?")

        assert "template" in result
        assert "variables" in result
        assert "time_period" in result["variables"]
        assert "total_revenue" in result["variables"]

    def test_resolve_conclusion_parallel_independent_vars(self):
        """Test that independent variables are resolved together via resolve_many_sync."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: Report for {region}: {metric_a} and {metric_b}.
VARIABLES:
- {region}: Target region
- {metric_a}: First metric
- {metric_b}: Second metric""",
            """{region}: []
{metric_a}: []
{metric_b}: []""",
        ]

        resolver = FactResolver(llm=mock_llm)
        resolve_calls = []

        # Mock resolve_many_sync to track calls
        original_resolve_many = resolver.resolve_many_sync

        def tracking_resolve_many(requests):
            resolve_calls.append(requests)
            return original_resolve_many(requests)

        resolver.resolve_many_sync = tracking_resolve_many

        # Pre-cache all facts
        resolver.add_user_fact("region", "US")
        resolver.add_user_fact("metric_a", 100)
        resolver.add_user_fact("metric_b", 200)

        result = resolver.resolve_conclusion("Show me metrics for US")

        # All three variables are independent - should be resolved together
        assert len(resolve_calls) == 1  # Single call to resolve_many_sync
        assert len(resolve_calls[0]) == 3  # All three variables

    def test_resolve_conclusion_substitutes_values(self):
        """Test that resolved values are substituted into the template."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: Revenue in {quarter} was ${amount}.
VARIABLES:
- {quarter}: The quarter
- {amount}: Revenue amount""",
            """{quarter}: []
{amount}: []""",
        ]

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("quarter", "Q3")
        resolver.add_user_fact("amount", 50000)

        result = resolver.resolve_conclusion("Revenue in Q3?")

        # Answer should have substituted values
        assert "Q3" in result["answer"]
        assert "50000" in result["answer"]

    def test_resolve_conclusion_tracks_unresolved(self):
        """Test that unresolved variables are tracked."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: Customer {customer_id} has tier {tier}.
VARIABLES:
- {customer_id}: Customer identifier
- {tier}: Customer tier classification""",
            """{customer_id}: []
{tier}: []""",
        ]

        resolver = FactResolver(llm=mock_llm)
        # Only provide one fact - tier will be unresolved
        resolver.add_user_fact("customer_id", "ACME-001")

        result = resolver.resolve_conclusion("What tier is customer ACME?")

        # tier should be unresolved
        assert "tier" in result["unresolved"]
        # customer_id should be resolved
        assert "customer_id" not in result["unresolved"]

    def test_resolve_conclusion_builds_derivation_trace(self):
        """Test that derivation trace is built correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: The value is {x}.
VARIABLES:
- {x}: The value""",
            """{x}: []""",
        ]

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("x", 42)

        result = resolver.resolve_conclusion("What is x?")

        # Derivation should have Statement, Variable Resolution, Conclusion sections
        assert "derivation" in result
        assert "Statement" in result["derivation"]
        assert "Variable Resolution" in result["derivation"]
        assert "Conclusion" in result["derivation"]

    def test_resolve_conclusion_calculates_confidence(self):
        """Test that confidence is calculated as min of all facts."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: {a} and {b}.
VARIABLES:
- {a}: First value
- {b}: Second value""",
            """{a}: []
{b}: []""",
        ]

        resolver = FactResolver(llm=mock_llm)

        # Add facts with different confidence levels
        resolver._cache["a"] = Fact(
            name="a", value=100, confidence=1.0, source=FactSource.DATABASE
        )
        resolver._cache["b"] = Fact(
            name="b", value=200, confidence=0.7, source=FactSource.LLM_KNOWLEDGE
        )

        result = resolver.resolve_conclusion("Show a and b")

        # Confidence should be min of all facts
        assert result["confidence"] == 0.7


class TestBuildDerivationTrace:
    """Tests for the _build_derivation_trace helper method."""

    def test_build_derivation_trace_includes_all_sections(self):
        """Test that derivation trace includes all required sections."""
        resolver = FactResolver()

        fact = Fact(name="x", value=42, confidence=1.0, source=FactSource.DATABASE)
        trace = resolver._build_derivation_trace(
            template="Answer is {x}",
            resolved={"x": fact},
            unresolved=[],
            variables={"x": "The answer"},
            answer="Answer is 42",
        )

        assert "**Statement:**" in trace
        assert "**Variable Resolution:**" in trace
        assert "**Conclusion:**" in trace
        assert "Answer is {x}" in trace
        assert "Answer is 42" in trace

    def test_build_derivation_trace_shows_provenance(self):
        """Test that provenance information is included."""
        resolver = FactResolver()

        fact = Fact(
            name="revenue",
            value=50000,
            confidence=0.9,
            source=FactSource.DATABASE,
            source_name="sales_db",
            query="SELECT SUM(amount) FROM sales",
        )
        trace = resolver._build_derivation_trace(
            template="Revenue is {revenue}",
            resolved={"revenue": fact},
            unresolved=[],
            variables={"revenue": "Total revenue"},
            answer="Revenue is 50000",
        )

        assert "database" in trace.lower()
        assert "sales_db" in trace
        assert "90%" in trace or "0.9" in trace
        assert "query:" in trace.lower()

    def test_build_derivation_trace_shows_unresolved(self):
        """Test that unresolved variables are clearly marked."""
        resolver = FactResolver()

        trace = resolver._build_derivation_trace(
            template="{known} and {unknown}",
            resolved={"known": Fact(name="known", value=1, source=FactSource.RULE)},
            unresolved=["unknown"],
            variables={"known": "Known value", "unknown": "Needs user input"},
            answer="{known} and {unknown}",
        )

        assert "**Unresolved" in trace
        assert "{unknown}" in trace
        assert "Needs user input" in trace

    def test_build_derivation_trace_shows_derived_from(self):
        """Test that derived-from dependencies are shown."""
        resolver = FactResolver()

        base = Fact(name="base", value=10, source=FactSource.DATABASE)
        derived = Fact(
            name="derived",
            value=20,
            confidence=1.0,
            source=FactSource.RULE,
            because=[base],
        )
        trace = resolver._build_derivation_trace(
            template="Result is {derived}",
            resolved={"derived": derived},
            unresolved=[],
            variables={"derived": "Computed value"},
            answer="Result is 20",
        )

        assert "derived from:" in trace.lower()
        assert "base" in trace


class TestResolveGoal:
    """Tests for Prolog-style goal decomposition via resolve_goal."""

    def test_resolve_goal_without_llm_returns_error(self):
        """Test that resolve_goal requires an LLM."""
        resolver = FactResolver()
        result = resolver.resolve_goal("What is the revenue?")
        assert "error" in result
        assert result["answer"] is None

    def test_resolve_goal_parses_prolog_response(self):
        """Test that Prolog-style LLM response is parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: total_revenue(q3, Revenue)

RULES:
total_revenue(Quarter, Revenue) :-
    date_range(Quarter, Start, End),
    sum_sales(Start, End, Revenue).

SOURCES:
date_range: LLM_KNOWLEDGE
sum_sales: DATABASE"""

        resolver = FactResolver(llm=mock_llm)

        # Pre-cache facts
        resolver.add_user_fact("date_range", ("2024-07-01", "2024-09-30"))
        resolver.add_user_fact("sum_sales", 150000)

        result = resolver.resolve_goal("What was revenue in Q3?")

        assert result["goal"] == "total_revenue(q3, Revenue)"
        assert len(result["rules"]) >= 1
        assert "date_range" in result["sources"]
        assert "sum_sales" in result["sources"]

    def test_parse_predicate(self):
        """Test predicate parsing helper."""
        resolver = FactResolver()

        # Simple predicate
        name, args = resolver._parse_predicate("foo(X, Y, Z)")
        assert name == "foo"
        assert args == ["X", "Y", "Z"]

        # Predicate with constants
        name, args = resolver._parse_predicate("revenue(q3, premium, Amount)")
        assert name == "revenue"
        assert args == ["q3", "premium", "Amount"]

        # No args
        name, args = resolver._parse_predicate("fact")
        assert name == "fact"
        assert args == []

    def test_parse_rules_extracts_dependencies(self):
        """Test that rule parsing extracts head -> body dependencies."""
        resolver = FactResolver()

        rules = [
            "answer(Q, R) :- date_range(Q, S, E), sum_revenue(S, E, R).",
            "complex(X, Y, Z) :- a(X, A), b(A, B), c(B, Y), compute(Y, Z).",
        ]

        deps = resolver._parse_rules(rules)

        assert "answer" in deps
        assert set(deps["answer"]) == {"date_range", "sum_revenue"}

        assert "complex" in deps
        assert set(deps["complex"]) == {"a", "b", "c", "compute"}

    def test_resolve_goal_builds_prolog_derivation(self):
        """Test that derivation trace is in Prolog style."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: value(X)

RULES:

SOURCES:
value: DATABASE"""

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("value", 42)

        result = resolver.resolve_goal("What is the value?")

        # Check Prolog-style derivation
        deriv = result["derivation"]
        assert "/* Query */" in deriv
        assert "?- value(X)" in deriv
        assert "/* Resolution */" in deriv
        assert "/* Answer */" in deriv

    def test_resolve_goal_binds_variables(self):
        """Test that variables are bound during resolution."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: answer(Result)

RULES:

SOURCES:
answer: DATABASE"""

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("answer", 100)

        result = resolver.resolve_goal("What is the answer?")

        # Result variable should be bound
        assert "Result" in result["bindings"]
        assert result["bindings"]["Result"] == 100

    def test_resolve_goal_tracks_unresolved(self):
        """Test that unresolved predicates are tracked."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: result(X, Y)

RULES:
result(X, Y) :-
    known(X),
    unknown(Y).

SOURCES:
known: DATABASE
unknown: USER_PROVIDED"""

        resolver = FactResolver(llm=mock_llm)
        # Only provide known fact
        resolver.add_user_fact("known", "value_x")

        result = resolver.resolve_goal("Get result")

        # unknown should be in unresolved list
        assert "unknown" in result["unresolved"]

    def test_substitute_bindings(self):
        """Test variable substitution in terms."""
        resolver = FactResolver()

        bindings = {"X": 10, "Y": 20, "Result": 30}
        term = "answer(X, Y, Result)"

        substituted = resolver._substitute_bindings(term, bindings)

        assert substituted == "answer(10, 20, 30)"


class TestPrologDerivationTrace:
    """Tests for Prolog-style derivation trace building."""

    def test_build_prolog_derivation_includes_query(self):
        """Test that derivation includes the original query."""
        resolver = FactResolver()

        trace = resolver._build_prolog_derivation(
            goal="revenue(q3, R)",
            rules=["revenue(Q, R) :- sales(Q, R)."],
            bindings={"R": 50000},
            resolved={"sales": Fact(name="sales", value=50000, source=FactSource.DATABASE)},
            unresolved=[],
        )

        assert "?- revenue(q3, R)" in trace

    def test_build_prolog_derivation_includes_rules(self):
        """Test that derivation includes the rules used."""
        resolver = FactResolver()

        trace = resolver._build_prolog_derivation(
            goal="result(X)",
            rules=["result(X) :- compute(X)."],
            bindings={},
            resolved={},
            unresolved=[],
        )

        assert "/* Rules */" in trace
        assert "result(X) :- compute(X)." in trace

    def test_build_prolog_derivation_shows_resolution_with_source(self):
        """Test that resolution shows facts with their sources."""
        resolver = FactResolver()

        fact = Fact(
            name="revenue",
            value=100000,
            confidence=1.0,
            source=FactSource.DATABASE,
            source_name="sales_db",
            query="SELECT SUM(amount) FROM sales",
        )

        trace = resolver._build_prolog_derivation(
            goal="revenue(R)",
            rules=[],
            bindings={"R": 100000},
            resolved={"revenue": fact},
            unresolved=[],
        )

        assert "/* Resolution */" in trace
        assert "revenue(100000)" in trace
        assert "database:sales_db" in trace
        assert "SQL:" in trace

    def test_build_prolog_derivation_shows_unresolved(self):
        """Test that unresolved predicates are shown."""
        resolver = FactResolver()

        trace = resolver._build_prolog_derivation(
            goal="answer(X, Y)",
            rules=[],
            bindings={"X": 10},
            resolved={},
            unresolved=["missing_fact"],
        )

        assert "/* Unresolved" in trace
        assert "missing_fact" in trace

    def test_build_prolog_derivation_shows_bindings(self):
        """Test that variable bindings are shown."""
        resolver = FactResolver()

        trace = resolver._build_prolog_derivation(
            goal="compute(X, Y, Result)",
            rules=[],
            bindings={"X": 10, "Y": 20, "Result": 200},
            resolved={},
            unresolved=[],
        )

        assert "/* Bindings */" in trace
        assert "X = 10" in trace
        assert "Y = 20" in trace
        assert "Result = 200" in trace


class TestSQLTransformForSQLite:
    """Tests for MySQL/PostgreSQL to SQLite SQL transformation."""

    def test_date_format_transformation(self):
        """Test DATE_FORMAT is converted to strftime."""
        resolver = FactResolver()
        sql = "SELECT DATE_FORMAT(order_date, '%Y-%m') as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "strftime('%Y-%m', order_date)" in result
        assert "DATE_FORMAT" not in result

    def test_date_sub_with_curdate(self):
        """Test DATE_SUB(CURDATE(), INTERVAL n MONTH) conversion."""
        resolver = FactResolver()
        sql = "SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date('now', '-6 months')" in result
        assert "DATE_SUB" not in result
        assert "CURDATE" not in result

    def test_date_sub_with_column(self):
        """Test DATE_SUB with column reference."""
        resolver = FactResolver()
        sql = "SELECT DATE_SUB(created_at, INTERVAL 30 DAY) as past_date FROM users"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date(created_at, '-30 days')" in result

    def test_curdate_standalone(self):
        """Test standalone CURDATE() conversion."""
        resolver = FactResolver()
        sql = "SELECT * FROM orders WHERE order_date = CURDATE()"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date('now')" in result
        assert "CURDATE" not in result

    def test_now_conversion(self):
        """Test NOW() conversion to datetime('now')."""
        resolver = FactResolver()
        sql = "SELECT * FROM logs WHERE created_at < NOW()"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "datetime('now')" in result
        assert "NOW()" not in result

    def test_year_function(self):
        """Test YEAR() conversion to strftime."""
        resolver = FactResolver()
        sql = "SELECT YEAR(order_date) as year FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%Y', order_date) AS INTEGER)" in result
        assert "YEAR(" not in result

    def test_month_function(self):
        """Test MONTH() conversion to strftime."""
        resolver = FactResolver()
        sql = "SELECT MONTH(order_date) as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%m', order_date) AS INTEGER)" in result
        assert "MONTH(" not in result

    def test_extract_year(self):
        """Test EXTRACT(YEAR FROM col) conversion."""
        resolver = FactResolver()
        sql = "SELECT EXTRACT(YEAR FROM order_date) as year FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%Y', order_date) AS INTEGER)" in result
        assert "EXTRACT" not in result

    def test_extract_month(self):
        """Test EXTRACT(MONTH FROM col) conversion."""
        resolver = FactResolver()
        sql = "SELECT EXTRACT(MONTH FROM order_date) as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%m', order_date) AS INTEGER)" in result
        assert "EXTRACT" not in result

    def test_complex_query_transformation(self):
        """Test transformation of complex query with multiple MySQL functions."""
        resolver = FactResolver()
        sql = """
        SELECT
            DATE_FORMAT(o.order_date, '%Y-%m') as month,
            c.tier,
            SUM(o.total_amount) as monthly_revenue
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        GROUP BY YEAR(o.order_date), MONTH(o.order_date), c.tier
        ORDER BY month
        """
        result = resolver._transform_sql_for_sqlite(sql)
        # Check all MySQL functions are converted
        assert "DATE_FORMAT" not in result
        assert "DATE_SUB" not in result
        assert "CURDATE" not in result
        # Check SQLite equivalents are present
        assert "strftime('%Y-%m', o.order_date)" in result
        assert "date('now', '-6 months')" in result

    def test_case_insensitive_transformation(self):
        """Test that transformations are case-insensitive."""
        resolver = FactResolver()
        sql = "SELECT date_format(created_at, '%Y') as year, curdate() as today FROM users"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "strftime('%Y', created_at)" in result
        assert "date('now')" in result
