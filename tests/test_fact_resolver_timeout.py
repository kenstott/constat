# Copyright (c) 2025 Kenneth Stott
# Canary: 7e5732af-b528-411f-ab1f-c35a3ae572d7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Timeout and rate-limiter tests for FactResolver."""

from __future__ import annotations

import asyncio
import pytest
from constat.execution.fact_resolver import (
    AsyncFactResolver,
    RateLimiter,
    RateLimiterConfig,
)


class TestRateLimiterBasic:
    """Tests for basic RateLimiter execution and concurrency control."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic_execution(self):
        """Test basic execution through rate limiter."""
        limiter = RateLimiter()

        async def success_task():
            return "success"

        result = await limiter.execute(success_task)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrency_control(self):
        """Test that rate limiter limits concurrency."""
        import time

        config = RateLimiterConfig(max_concurrent=2)
        limiter = RateLimiter(config)

        call_times = []

        async def slow_task(task_id):
            call_times.append(("start", task_id, time.time()))
            await asyncio.sleep(0.1)
            call_times.append(("end", task_id, time.time()))
            return task_id

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
    async def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        config = RateLimiterConfig(max_concurrent=5, max_retries=3, base_delay=0.01, jitter=0.0)
        limiter = RateLimiter(config)

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
        config = RateLimiterConfig(max_concurrent=10, max_retries=5)
        resolver = AsyncFactResolver(rate_limiter_config=config)

        assert resolver._rate_limiter.config.max_concurrent == 10
        assert resolver._rate_limiter.config.max_retries == 5
