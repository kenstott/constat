# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Lazy fact resolution with provenance tracking.

This module provides on-demand fact resolution during plan execution.
Facts are resolved lazily when accessed, with automatic provenance tracking
for explainability.

Architecture:
1. Top-level plan is generated with assumed facts (e.g., "customer_ltv(X)")
2. During execution, when a fact is needed, the resolver:
   - Checks cache (already resolved this session)
   - Checks rules (Python functions that can derive the fact)
   - Tries database query (LLM generates SQL)
   - Tries LLM knowledge (world facts, heuristics)
   - Falls back to sub-plan generation (for complex derived facts)
3. Each resolution records provenance for explainability

This is an opt-in feature. Simple queries can run without it.

Parallel Resolution (AsyncFactResolver):
For I/O-bound fact resolution (database queries, LLM calls), use AsyncFactResolver
which provides:
- resolve_async(): Async single fact resolution
- resolve_many_async(): Parallel resolution of multiple facts (3-5x speedup)
- Parallel source resolution: Try DATABASE + LLM_KNOWLEDGE + SUB_PLAN concurrently
"""

from __future__ import annotations

from typing import Optional

from ._types import (
    AuditContext,
    Fact,
    FactDependency,
    FactSource,
    ProofNode,
    ResolutionSpec,
    ResolutionStrategy,
    RuleFunction,
    Tier2AssessmentResult,
    Tier2Strategy,
    format_source_attribution,
    GROUND_TRUTH_SOURCES,
    DERIVED_SOURCES,
    TIER2_ASSESSMENT_PROMPT,
    ARRAY_ROW_THRESHOLD,
    ARRAY_SIZE_THRESHOLD,
)
from ._rate_limiter import (
    RateLimitError,
    RateLimitExhaustedError,
    RateLimiter,
    RateLimiterConfig,
    _DEFAULT_EXECUTOR,
)
from ._goals import GoalsMixin
from ._session import SessionMixin
from ._sources import SourcesMixin
from ._resolution import ResolutionMixin


class FactResolver(GoalsMixin, SessionMixin, SourcesMixin, ResolutionMixin):
    """
    Lazy fact resolver with provenance tracking.

    Usage:
        resolver = FactResolver(llm=provider, schema_manager=sm)

        # Register custom rules
        @resolver.rule("customer_ltv")
        def calc_ltv(resolver, customer_id: str) -> Fact:
            transactions = resolver.resolve("customer_transactions", customer_id=customer_id)
            return Fact(
                name=f"customer_ltv:{customer_id}",
                value=sum(t["amount"] for t in transactions.value),
                confidence=transactions.confidence,
                source=FactSource.RULE,
                rule_name="calc_ltv",
                because=[transactions]
            )

        # Resolve facts (lazy, cached)
        ltv = resolver.resolve("customer_ltv", customer_id="acme")
        print(ltv.derivation_trace)
    """

    def __init__(
        self,
        llm=None,
        schema_manager=None,
        config=None,
        strategy: Optional[ResolutionStrategy] = None,
        event_callback=None,
        datastore=None,
        doc_tools=None,
        learning_callback=None,
    ):
        self.llm = llm
        self.schema_manager = schema_manager
        self.config = config
        self.strategy = strategy or ResolutionStrategy()
        self._event_callback = event_callback
        self._datastore = datastore
        self._doc_tools = doc_tools
        self._learning_callback = learning_callback

        # Caches
        self._cache: dict[str, Fact] = {}
        self._rules: dict[str, RuleFunction] = {}

        # Resolution state (for sub-plan depth tracking)
        self._resolution_depth: int = 0

        # Deadline for Tier 1 parallel resolution (None = no deadline)
        self._resolution_deadline: Optional[float] = None

        # All resolutions this session (for audit)
        self.resolution_log: list[Fact] = []

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit a resolution event if callback is registered."""
        if self._event_callback:
            self._event_callback(event_type, data)

    def rule(self, fact_pattern: str):
        """Decorator to register a rule function for a fact pattern.

        Example:
            @resolver.rule("customer_ltv")
            def calc_ltv(resolver, customer_id: str) -> Fact:
                ...
        """
        def decorator(func: RuleFunction) -> RuleFunction:
            self._rules[fact_pattern] = func
            return func
        return decorator

    def register_rule(self, fact_pattern: str, func: RuleFunction) -> None:
        """Register a rule function programmatically."""
        self._rules[fact_pattern] = func

    def set_resolution_context(
        self,
        resolved_premises: Optional[dict[str, "Fact"]] = None,
        pending_premises: Optional[list[dict]] = None,
        available_sources: Optional[str] = None,
    ) -> None:
        """Set context for Tier 2 LLM assessment.

        Called by session before resolving premises to provide context about
        the current plan's premises.

        Args:
            resolved_premises: Dict of fact_id -> Fact for already resolved premises
            pending_premises: List of premise dicts still to be resolved
            available_sources: Description of available data sources
        """
        self._resolution_context = {
            "resolved_premises": resolved_premises or {},
            "pending_premises": pending_premises or [],
            "available_sources": available_sources or "",
        }


# Import AsyncFactResolver AFTER FactResolver is defined to avoid circular imports
from ._async import AsyncFactResolver  # noqa: E402


__all__ = [
    # Types
    "AuditContext",
    "Fact",
    "FactDependency",
    "FactSource",
    "ProofNode",
    "ResolutionSpec",
    "ResolutionStrategy",
    "RuleFunction",
    "Tier2AssessmentResult",
    "Tier2Strategy",
    "format_source_attribution",
    # Constants
    "GROUND_TRUTH_SOURCES",
    "DERIVED_SOURCES",
    "TIER2_ASSESSMENT_PROMPT",
    "ARRAY_ROW_THRESHOLD",
    "ARRAY_SIZE_THRESHOLD",
    # Rate limiting
    "RateLimitError",
    "RateLimitExhaustedError",
    "RateLimiter",
    "RateLimiterConfig",
    # Resolvers
    "FactResolver",
    "AsyncFactResolver",
]
