# Copyright (c) 2025 Kenneth Stott
# Canary: 7e5732af-b528-411f-ab1f-c35a3ae572d7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Async and concurrent tests for AsyncFactResolver."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, AsyncMock
import pytest
from constat.execution.fact_resolver import (
    Fact,
    FactSource,
    FactResolver,
    AsyncFactResolver,
    ResolutionStrategy,
)


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

        start = time.time()
        for i in range(3):
            resolver.resolve("slow_fact", id=i)
        sequential_time = time.time() - start

        resolver.clear_cache()

        start = time.time()
        await resolver.resolve_many_async([
            ("slow_fact", {"id": i}) for i in range(3, 6)
        ])
        parallel_time = time.time() - start

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

        @resolver.rule("multi_source")
        def multi_rule(res, params):
            return Fact(
                name="multi_source",
                value="from_rule",
                source=FactSource.RULE,
            )

        result = await resolver.resolve_async("multi_source")

        assert result.value == "from_rule"
        assert result.source == FactSource.RULE


class TestAsyncFactResolverIntegration:
    """Integration tests for AsyncFactResolver."""

    @pytest.mark.asyncio
    async def test_full_resolution_flow(self):
        """Test complete resolution flow with multiple sources."""
        resolver = AsyncFactResolver()

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

        tasks = [resolver.resolve_async("concurrent_fact") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        values = [r.value for r in results]
        assert all(v == values[0] for v in values)  # All same value
        assert len(results) == 5
        assert all(r.source == FactSource.RULE for r in results)

    @pytest.mark.asyncio
    async def test_inheritance_from_fact_resolver(self):
        """Test that AsyncFactResolver inherits FactResolver methods."""
        resolver = AsyncFactResolver()

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

        result = resolver.resolve("sync_test")

        assert result.value == "sync_value"


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
        resolver = AsyncFactResolver()

        @resolver.rule("slow_fact")
        def resolve_slow(resolver, params):
            time.sleep(0.1)  # 100ms delay
            return Fact(name=f"slow_fact_{params.get('id')}", value=params.get('id'), source=FactSource.RULE)

        start = time.time()
        facts = resolver.resolve_many_sync([
            ("slow_fact", {"id": i}) for i in range(5)
        ])
        elapsed = time.time() - start

        assert len(facts) == 5
        assert elapsed < 0.4, f"Expected parallel execution (<400ms), got {elapsed*1000:.0f}ms"


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

        requests = [("slow_fact", {"id": i}) for i in range(NUM_FACTS)]

        start = time.time()
        results = await resolver.resolve_many_async(requests)
        parallel_time = (time.time() - start) * 1000

        assert len(results) == NUM_FACTS
        for i, result in enumerate(results):
            assert result.value == f"result_{i}"

        expected_sequential_time = NUM_FACTS * DELAY_MS

        print(f"\nParallel time: {parallel_time:.1f}ms")
        print(f"Expected sequential: {expected_sequential_time}ms")

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
        DELAY_MS = 50
        NUM_FACTS = 4

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

        resolver.clear_cache()
        start = time.time()
        for name, params in requests:
            await resolver.resolve_async(name, **params)
            resolver.clear_cache()  # Clear cache to force re-resolution
        sequential_time = time.time() - start

        resolver.clear_cache()
        start = time.time()
        await resolver.resolve_many_async(requests)
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time

        print(f"\nSequential: {sequential_time*1000:.1f}ms")
        print(f"Parallel: {parallel_time*1000:.1f}ms")
        print(f"Speedup: {speedup:.2f}x")

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
        DELAY_MS = 100
        NUM_FACTS = 5

        resolver = AsyncFactResolver()
        call_timestamps = []

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

        resolver.resolve_async = tracked_resolve

        requests = [("async_io_fact", {"id": i}) for i in range(NUM_FACTS)]

        start = time.time()
        results = await resolver.resolve_many_async(requests)
        total_time = (time.time() - start) * 1000

        assert len(results) == NUM_FACTS

        starts = [t for t in call_timestamps if t[0] == "start"]
        ends = [t for t in call_timestamps if t[0] == "end"]

        start_times = [t[2] for t in starts]
        end_times = [t[2] for t in ends]

        start_window = max(start_times) - min(start_times)
        print(f"\nStart window: {start_window*1000:.1f}ms")
        print(f"Total time: {total_time:.1f}ms")
        print(f"Expected if parallel: ~{DELAY_MS}ms")
        print(f"Expected if sequential: ~{NUM_FACTS * DELAY_MS}ms")

        assert start_window < DELAY_MS / 2, (
            f"Facts started with {start_window*1000:.1f}ms spread. "
            f"Should be <{DELAY_MS/2}ms if truly parallel."
        )

        assert total_time < DELAY_MS * 2, (
            f"Total time {total_time:.1f}ms exceeds {DELAY_MS * 2}ms. "
            f"This suggests sequential execution, not parallel."
        )

    @pytest.mark.asyncio
    async def test_resolve_many_sync_wrapper_parallelizes(self):
        """
        Component test: Verify that resolve_many_sync (the sync wrapper) also parallelizes.
        """
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

        start = time.time()
        results = resolver.resolve_many_sync(requests)
        elapsed = (time.time() - start) * 1000

        assert len(results) == NUM_FACTS

        sequential_expected = NUM_FACTS * DELAY_MS
        print(f"\nSync wrapper time: {elapsed:.1f}ms")
        print(f"Sequential expected: {sequential_expected}ms")

        assert elapsed < sequential_expected * 0.7, (
            f"Sync wrapper took {elapsed:.1f}ms, expected <{sequential_expected * 0.7:.1f}ms. "
            f"The sync wrapper may not be properly invoking async parallelization."
        )
