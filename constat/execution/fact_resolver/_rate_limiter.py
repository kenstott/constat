# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Rate limiting for async fact resolution."""

from __future__ import annotations

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

# Shared thread pool for running sync operations in async context
_DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=10)


class RateLimitError(Exception):
    """Raised when an API rate limit is hit."""
    pass


class RateLimitExhaustedError(Exception):
    """Raised when max retries exceeded for rate limiting."""
    pass


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiting."""
    max_concurrent: int = 5  # Max concurrent LLM calls
    max_retries: int = 3  # Max retry attempts on rate limit
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    jitter: float = 0.5  # Random jitter factor (0-1)


class RateLimiter:
    """
    Rate limiter with semaphore for concurrency control and exponential backoff.

    Prevents overwhelming LLM APIs with too many concurrent requests and
    handles rate limit errors gracefully with retries.

    Usage:
        limiter = RateLimiter(max_concurrent=5)

        async def call_llm():
            return await llm.generate(...)

        result = await limiter.execute(call_llm())
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        self.config = config or RateLimiterConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._request_count = 0
        self._rate_limit_hits = 0

    async def execute(self, coro_or_func):
        """
        Execute a coroutine or async function with rate limiting and exponential backoff.

        Args:
            coro_or_func: Either a coroutine to execute, or an async callable that
                         creates a new coroutine (for retry support)

        Returns:
            Result of the coroutine

        Raises:
            RateLimitExhaustedError: If max retries exceeded
        """
        # Determine if we got a callable (can retry) or a coroutine (single use)
        is_callable = callable(coro_or_func) and not asyncio.iscoroutine(coro_or_func)

        async with self._semaphore:
            self._request_count += 1

            for attempt in range(self.config.max_retries):
                try:
                    if is_callable:
                        return await coro_or_func()
                    else:
                        return await coro_or_func
                except Exception as e:
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    is_rate_limit = any(
                        indicator in error_str
                        for indicator in ["rate limit", "ratelimit", "429", "too many requests"]
                    )

                    if not is_rate_limit:
                        raise  # Re-raise non-rate-limit errors

                    if not is_callable:
                        # Can't retry a single coroutine
                        raise

                    self._rate_limit_hits += 1

                    if attempt == self.config.max_retries - 1:
                        raise RateLimitExhaustedError(
                            f"Rate limit exceeded after {self.config.max_retries} retries"
                        ) from e

                    # Calculate backoff delay with exponential increase and jitter
                    delay = min(
                        self.config.base_delay * (2 ** attempt),
                        self.config.max_delay
                    )
                    jitter = random.uniform(0, self.config.jitter * delay)
                    total_delay = delay + jitter

                    await asyncio.sleep(total_delay)

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._request_count,
            "rate_limit_hits": self._rate_limit_hits,
            "max_concurrent": self.config.max_concurrent,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._request_count = 0
        self._rate_limit_hits = 0
