# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Async fact resolver with parallel resolution support."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

# Import FactResolver at module level - works because __init__.py defines
# FactResolver before importing this module.
from . import FactResolver
from ._rate_limiter import (
    RateLimiter,
    RateLimiterConfig,
    _DEFAULT_EXECUTOR,
)
from ._types import (
    Fact,
    FactSource,
    ResolutionStrategy,
)

logger = logging.getLogger(__name__)


class AsyncFactResolver(FactResolver):
    """
    Async-enabled fact resolver with parallel resolution support.

    Provides significant speedup for I/O-bound fact resolution by:
    - Running LLM calls and database queries concurrently
    - Resolving multiple independent facts in parallel
    - Optionally trying multiple sources simultaneously

    Usage:
        resolver = AsyncFactResolver(llm=provider, schema_manager=sm)

        # Single async resolution
        fact = await resolver.resolve_async("customer_ltv", customer_id="acme")

        # Parallel resolution of multiple facts (3-5x speedup)
        facts = await resolver.resolve_many_async([
            ("customer_ltv", {"customer_id": "acme"}),
            ("customer_ltv", {"customer_id": "globex"}),
            ("revenue_threshold", {}),
        ])
    """

    def __init__(
        self,
        llm=None,
        schema_manager=None,
        config=None,
        strategy: Optional[ResolutionStrategy] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        parallel_sources: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """
        Initialize AsyncFactResolver.

        Args:
            llm: LLM provider for queries/knowledge
            schema_manager: For database queries
            config: For config-based facts
            strategy: Resolution strategy configuration
            executor: Custom thread pool executor (uses shared default if not provided)
            parallel_sources: If True, try DATABASE + LLM_KNOWLEDGE + SUB_PLAN
                            concurrently instead of sequentially
            rate_limiter: Custom rate limiter instance (for shared limiting across resolvers)
            rate_limiter_config: Config for creating a new rate limiter
        """
        super().__init__(llm, schema_manager, config, strategy)
        self._executor = executor or _DEFAULT_EXECUTOR
        self._parallel_sources = parallel_sources

        # Rate limiting for LLM calls
        if rate_limiter:
            self._rate_limiter = rate_limiter
        elif rate_limiter_config:
            self._rate_limiter = RateLimiter(rate_limiter_config)
        else:
            # Default rate limiter
            self._rate_limiter = RateLimiter()

    @property
    def rate_limiter_stats(self) -> dict:
        """Get rate limiter statistics."""
        return self._rate_limiter.stats

    async def _call_llm_with_rate_limit(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Call LLM with rate limiting and exponential backoff.

        Args:
            system: System prompt
            user_message: User message
            max_tokens: Maximum tokens in response

        Returns:
            LLM response string
        """
        async def _make_call():
            if hasattr(self.llm, 'async_generate'):
                return await self.llm.async_generate(
                    system=system,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    executor=self._executor,
                )
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self._executor,
                    lambda: self.llm.generate(
                        system=system,
                        user_message=user_message,
                        max_tokens=max_tokens,
                    )
                )

        # Pass async function (not called) so it can be retried on rate limit
        return await self._rate_limiter.execute(_make_call)

    async def resolve_async(self, fact_name: str, **params) -> Fact:
        """
        Async version of resolve().

        Resolves a fact by trying sources in priority order (or parallel if configured).

        Args:
            fact_name: The fact to resolve
            **params: Parameters for the fact

        Returns:
            Fact with value, confidence, and provenance
        """
        cache_key = self._cache_key(fact_name, params)

        # Check cache first (sync, fast)
        if FactSource.CACHE in self.strategy.source_priority:
            cached = self._cache.get(cache_key)
            if cached and cached.confidence >= self.strategy.min_confidence:
                self.resolution_log.append(cached)
                return cached

        # Check config (sync, fast)
        if FactSource.CONFIG in self.strategy.source_priority:
            config_fact = self._resolve_from_config(fact_name, params)
            if config_fact and config_fact.is_resolved:
                self._cache[cache_key] = config_fact
                self.resolution_log.append(config_fact)
                return config_fact

        # Check rules - run in executor to allow true parallelism
        if FactSource.RULE in self.strategy.source_priority:
            rule_fact = await self._resolve_from_rule_async(fact_name, params)
            if rule_fact and rule_fact.is_resolved:
                self._cache[cache_key] = rule_fact
                self.resolution_log.append(rule_fact)
                return rule_fact

        # I/O-bound sources - run async
        if self._parallel_sources:
            fact = await self._resolve_parallel_sources(fact_name, params, cache_key)
        else:
            fact = await self._resolve_sequential_sources(fact_name, params, cache_key)

        if fact and fact.is_resolved:
            return fact

        # Could not resolve
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Could not resolve fact: {fact_name} with params {params}"
        )
        self.resolution_log.append(unresolved)
        return unresolved

    async def _resolve_sequential_sources(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """Try I/O-bound sources sequentially (default behavior).

        Tries cheap sources first (DATABASE, DOCUMENT, SUB_PLAN), then
        falls back to expensive sources (LLM_KNOWLEDGE) only if needed.
        """
        # Cheap I/O sources first
        cheap_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN)
        ]
        # Expensive fallback
        expensive_sources = [
            s for s in self.strategy.source_priority
            if s == FactSource.LLM_KNOWLEDGE
        ]
        io_sources = cheap_sources + expensive_sources

        for source in io_sources:
            fact = await self._try_resolve_async(source, fact_name, params, cache_key)
            if fact and fact.is_resolved:
                if fact.confidence >= self.strategy.min_confidence:
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    return fact

        return None

    async def _resolve_parallel_sources(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """
        Try I/O-bound sources in parallel, picking the best result.

        This can provide speedup when multiple sources might work,
        as we don't wait for each to fail before trying the next.

        NOTE: Only runs CHEAP I/O sources in parallel (DATABASE, DOCUMENT, SUB_PLAN).
        LLM_KNOWLEDGE is excluded - it's expensive (API cost + latency) and should
        only be used as a fallback if cheap sources fail.

        Selection strategy:
        1. Collect all successful results from parallel execution
        2. Filter by min_confidence threshold
        3. Pick best based on: (priority_index, -confidence) for stable ordering
        """
        import logging
        logger = logging.getLogger(__name__)

        # Only parallelize cheap I/O sources - exclude LLM_KNOWLEDGE (expensive)
        io_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN)
        ]

        if not io_sources:
            return None

        # Create tasks for all sources
        tasks = [
            self._try_resolve_async(source, fact_name, params, cache_key)
            for source in io_sources
        ]

        # Use asyncio.gather with return_exceptions to get all results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all valid results with their source priority
        valid_results: list[tuple[int, float, Fact]] = []
        for i, (source, result) in enumerate(zip(io_sources, results)):
            if isinstance(result, Exception):
                logger.debug(f"[_resolve_parallel] {source.value} raised: {result}")
                continue
            if result and result.is_resolved:
                if result.confidence >= self.strategy.min_confidence:
                    # Store (priority_index, confidence, fact)
                    valid_results.append((i, result.confidence, result))
                    logger.debug(f"[_resolve_parallel] {source.value}: conf={result.confidence:.2f}")

        if not valid_results:
            return None

        # Pick best result:
        # - Sort by priority index (lower = better), then by confidence (higher = better)
        # - This means DATABASE (priority 0) beats DOCUMENT (priority 1) at same confidence
        # - But if DOCUMENT has significantly higher confidence, it could win
        #   when confidence_weight_factor is set (future enhancement)
        valid_results.sort(key=lambda x: (x[0], -x[1]))
        best = valid_results[0]

        logger.debug(f"[_resolve_parallel] Picked {best[2].source.value} with confidence {best[1]:.2f}")

        self._cache[cache_key] = best[2]
        self.resolution_log.append(best[2])
        return best[2]

    async def _try_resolve_async(
        self,
        source: FactSource,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """Async version of _try_resolve for I/O-bound sources."""
        if source == FactSource.DATABASE:
            return await self._resolve_from_database_async(fact_name, params)
        elif source == FactSource.DOCUMENT:
            return await self._resolve_from_document_async(fact_name, params)
        elif source == FactSource.LLM_KNOWLEDGE:
            return await self._resolve_from_llm_async(fact_name, params)
        elif source == FactSource.SUB_PLAN:
            return await self._resolve_from_sub_plan_async(fact_name, params)
        return None

    async def _resolve_from_document_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async document resolution - runs sync method in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._resolve_from_document(fact_name, params)
        )

    async def _resolve_from_rule_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """
        Async rule resolution that runs sync rules in executor.

        This enables true parallelism for rule-based facts by running
        blocking rule functions in a thread pool instead of on the event loop.

        After execution, checks cache again - if another concurrent request
        already cached a result for this fact, we discard ours and return
        the cached one. This ensures consistent values for concurrent requests.
        """
        rule = self._rules.get(fact_name)
        if not rule:
            return None

        cache_key = self._cache_key(fact_name, params)

        try:
            loop = asyncio.get_event_loop()
            # Run the sync rule function in the thread pool executor
            # This allows multiple rules to execute truly in parallel
            result = await loop.run_in_executor(
                self._executor,
                lambda: rule(self, params)
            )

            # After execution, check if another concurrent request already cached
            # a result for this fact. If so, discard ours and return cached.
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            return result
        except Exception as e:
            logger.debug(f"[_resolve_from_rule_async] Rule failed for {fact_name}: {e}")
            return None

    async def _resolve_from_database_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async database resolution using LLM to generate SQL."""
        if not self.llm or not self.schema_manager:
            return None

        # Check if fact_name matches a table name - return "referenced" instead of loading data
        # This allows inferences to query the table directly from the original database
        fact_name_lower = fact_name.lower().strip()
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            # Match by table name (case-insensitive)
            if table_meta.name.lower() == fact_name_lower:
                # Store column metadata in reasoning for use by inferences
                columns = [c.name for c in table_meta.columns]
                reasoning = f"Table '{table_meta.name}' from database '{table_meta.database}'. Columns: {columns}"
                # Build descriptive value for UI display
                row_info = f"{table_meta.row_count:,} rows" if table_meta.row_count else "table"
                value_str = f"({table_meta.database}.{table_meta.name}) {row_info}"
                return Fact(
                    name=fact_name,
                    value=value_str,
                    source=FactSource.DATABASE,
                    source_name=table_meta.database,
                    reasoning=reasoning,
                    confidence=0.95,
                    table_name=table_meta.name,
                    row_count=table_meta.row_count,
                )

        schema_overview = self.schema_manager.get_overview()
        prompt = f"""I need to resolve this fact from the database:
Fact: {fact_name}
Parameters: {params}

Available schema:
{schema_overview}

If this fact can be DIRECTLY resolved with a SQL query, provide the query.

CRITICAL - When to respond NOT_POSSIBLE:
- If the fact asks for POLICY, RULES, GUIDELINES, or THRESHOLDS but no such table/config exists
- If you would need to ANALYZE PATTERNS or DERIVE rules from transactional data - that is NOT the same as having actual rules
- Statistical summaries (avg, count, distribution) are NOT policies - policies are prescriptive rules like "rating 5 = 10% raise"
- Do NOT return approximations or inferred rules as substitutes for explicitly stored policies

Respond in this format:
SQL: <your query here>
or
NOT_POSSIBLE: <reason>
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a SQL expert. Generate precise queries to resolve facts.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            if "NOT_POSSIBLE" in response:
                return None

            if "SQL:" in response:
                sql = response.split("SQL:", 1)[1].strip()
                sql = sql.replace("```sql", "").replace("```", "").strip()

                db_names = list(self.config.databases.keys()) if self.config else []
                db_name = db_names[0] if db_names else None
                if db_name:
                    conn = self.schema_manager.get_connection(db_name)
                    import pandas as pd

                    # Run SQL in executor (blocking I/O)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: pd.read_sql(sql, conn)
                    )

                    cache_key = self._cache_key(fact_name, params)

                    if len(result) == 1 and len(result.columns) == 1:
                        # Scalar value - store directly
                        value = result.iloc[0, 0]
                        return Fact(
                            name=cache_key,
                            value=value,
                            confidence=1.0,
                            source=FactSource.DATABASE,
                            source_name=db_name,
                            query=sql,
                            context=f"SQL Query:\n{sql}",
                        )
                    else:
                        # Multi-row result - check if should store as table
                        value = result.to_dict('records')

                        if self._datastore and self._should_store_as_table(value):
                            # Store as table and return reference
                            table_name, row_count = self._store_value_as_table(
                                fact_name, value, source_name=db_name
                            )
                            return Fact(
                                name=cache_key,
                                value=f"table:{table_name}",
                                confidence=1.0,
                                source=FactSource.DATABASE,
                                source_name=db_name,
                                query=sql,
                                table_name=table_name,
                                row_count=row_count,
                                context=f"SQL Query:\n{sql}",
                            )
                        else:
                            # Small result - store inline
                            return Fact(
                                name=cache_key,
                                value=value,
                                confidence=1.0,
                                source=FactSource.DATABASE,
                                source_name=db_name,
                                query=sql,
                                context=f"SQL Query:\n{sql}",
                            )
        except Exception as e:
            logger.debug(f"[_resolve_from_database_async] Database resolution failed for {fact_name}: {e}")

        return None

    async def _resolve_from_llm_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async LLM knowledge resolution."""
        if not self.llm:
            return None

        prompt = f"""I need to know this fact:
Fact: {fact_name}
Parameters: {params}

Do you know this from your training? This could be:
- World knowledge (e.g., "capital of France")
- Industry standards (e.g., "typical VIP threshold is $10,000")
- Common heuristics (e.g., "underperforming means <80% of target")

If you know this, respond with:
VALUE: <the value>
CONFIDENCE: <0.0-1.0, how confident are you>
TYPE: knowledge | heuristic
REASONING: <brief explanation>

If you don't know, respond with:
UNKNOWN
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a knowledgeable assistant. Provide facts you're confident about.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            if "UNKNOWN" in response:
                return None

            value = None
            confidence = 0.6
            reasoning = None
            source = FactSource.LLM_KNOWLEDGE

            for line in response.split("\n"):
                if line.startswith("VALUE:"):
                    value_str = line.split(":", 1)[1].strip()
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("TYPE:"):
                    if "heuristic" in line.lower():
                        source = FactSource.LLM_HEURISTIC
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            if value is not None:
                return Fact(
                    name=self._cache_key(fact_name, params),
                    value=value,
                    confidence=confidence,
                    source=source,
                    reasoning=reasoning,
                )
        except Exception as e:
            logger.debug(f"[_resolve_from_llm_async] LLM resolution failed for {fact_name}: {e}")

        return None

    async def _resolve_from_sub_plan_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async sub-plan resolution for complex derived facts."""
        if not self.strategy.allow_sub_plans:
            return None

        if self._resolution_depth >= self.strategy.max_sub_plan_depth:
            return None

        if not self.llm:
            return None

        prompt = f"""I need to derive this fact, but it's not directly available:
Fact: {fact_name}
Parameters: {params}

This fact needs to be computed from other facts.
Create a Python function that:
1. Uses resolver.resolve() to get the facts it depends on
2. Computes the final value
3. Returns a Fact with proper confidence (min of dependencies)

Example:
```python
def derive(resolver, params):
    revenue = resolver.resolve("total_revenue", customer_id=params["customer_id"])
    orders = resolver.resolve("order_count", customer_id=params["customer_id"])

    avg = revenue.value / orders.value if orders.value else 0

    return Fact(
        name=f"avg_order_value(customer_id={{params['customer_id']}})",
        value=avg,
        confidence=min(revenue.confidence, orders.confidence),
        source=FactSource.SUB_PLAN,
        because=[revenue, orders]
    )
```

Generate the derivation function for {fact_name}:
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a Python expert. Generate fact derivation functions.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            code = response
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]

            local_ns = {"Fact": Fact, "FactSource": FactSource}
            exec(code, local_ns)

            derive_func = local_ns.get("derive")
            if derive_func:
                self._resolution_depth += 1
                try:
                    # Run sync derive function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: derive_func(self, params)
                    )
                    return result
                finally:
                    self._resolution_depth -= 1
        except Exception as e:
            logger.debug(f"[_resolve_from_sub_plan_async] Sub-plan resolution failed for {fact_name}: {e}")

        return None

    async def resolve_many_async(
        self,
        fact_requests: list[tuple[str, dict]],
        on_resolve: Callable[[int, "Fact"], None] | None = None,
    ) -> list[Fact]:
        """
        Resolve multiple facts in parallel.

        This is the primary method for achieving speedup with parallel resolution.
        Independent facts are resolved concurrently, providing 3-5x speedup for
        I/O-bound resolutions.

        Args:
            fact_requests: List of (fact_name, params) tuples
            on_resolve: Optional callback called as each fact resolves.
                        Receives (index, fact) where index is the position in fact_requests.

        Returns:
            List of resolved Facts in same order as requests

        Example:
            facts = await resolver.resolve_many_async([
                ("customer_ltv", {"customer_id": "acme"}),
                ("customer_ltv", {"customer_id": "globex"}),
                ("revenue_threshold", {}),
            ])
        """
        if on_resolve is None:
            # No callback - use gather for efficiency
            tasks = [
                self.resolve_async(name, **params)
                for name, params in fact_requests
            ]
            return await asyncio.gather(*tasks)
        else:
            # With callback - use as_completed to emit events as each resolves
            async def resolve_with_index(idx: int, name: str, params: dict) -> tuple[int, Fact]:
                fact = await self.resolve_async(name, **params)
                return idx, fact

            tasks = [
                resolve_with_index(i, name, params)
                for i, (name, params) in enumerate(fact_requests)
            ]

            # Results will be out of order as they complete
            results = [None] * len(fact_requests)
            for coro in asyncio.as_completed(tasks):
                idx, fact = await coro
                results[idx] = fact
                # Call callback as each fact completes
                on_resolve(idx, fact)

            return results

    def resolve_many_sync(
        self,
        fact_requests: list[tuple[str, dict]],
        on_resolve: Callable[[int, "Fact"], None] | None = None,
    ) -> list[Fact]:
        """
        Synchronous wrapper for resolve_many_async.

        Useful when calling from sync code that wants parallel resolution.
        Handles both cases: when called from sync context (no event loop)
        and when called from async context (running event loop).

        Args:
            fact_requests: List of (fact_name, params) tuples
            on_resolve: Optional callback called as each fact resolves.
                        Receives (index, fact) where index is the position in fact_requests.

        Returns:
            List of resolved Facts in same order as requests
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - we're in sync context, safe to use asyncio.run()
            return asyncio.run(self.resolve_many_async(fact_requests, on_resolve))

        # We're in an async context - need to run in a separate thread
        # to avoid "asyncio.run() cannot be called from a running event loop"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.resolve_many_async(fact_requests, on_resolve)
            )
            return future.result()
