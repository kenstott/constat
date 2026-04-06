# Copyright (c) 2025 Kenneth Stott
# Canary: 7e5732af-b528-411f-ab1f-c35a3ae572d7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the COPYRIGHT holder.

"""Tests for lazy fact resolution with provenance tracking."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from constat.execution.fact_resolver import (
    Fact,
    FactSource,
    FactResolver,
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

    def test_recursive_resolution(self, resolver):
        """Test rules that depend on other facts."""
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

        @resolver.rule("test_fact")
        def rule_succeeds(resolver, params):
            sources_tried.append("rule")
            return Fact(name="test_fact", value="from_rule", source=FactSource.RULE)

        result = resolver.resolve("test_fact")

        assert result.source == FactSource.RULE
        assert sources_tried == ["rule"]


class TestDemandDrivenResolution:
    """Tests for demand-driven (conclusion-first) resolution."""

    def test_conclusion_triggers_dependency_resolution(self):
        """Test that resolving a conclusion fact triggers resolution of its dependencies."""
        resolver = FactResolver()
        resolution_order = []

        @resolver.rule("base_a")
        def resolve_a(resolver, params):
            resolution_order.append("base_a")
            return Fact(name="base_a", value=10, source=FactSource.RULE)

        @resolver.rule("base_b")
        def resolve_b(resolver, params):
            resolution_order.append("base_b")
            return Fact(name="base_b", value=20, source=FactSource.RULE)

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
        assert resolution_order == ["conclusion_start", "base_a", "base_b", "conclusion_end"]

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

        assert result.name == "level3"
        assert len(result.because) == 1
        assert result.because[0].name == "level2"
        assert len(result.because[0].because) == 1
        assert result.because[0].because[0].name == "level1"


class TestSequentialDependencies:
    """Tests for sequential resolution when facts depend on each other."""

    def test_dependent_facts_resolved_sequentially(self):
        """Test that dependent facts are resolved in correct order."""
        import time

        resolver = FactResolver()
        resolution_times = {}

        @resolver.rule("start_date")
        def resolve_start(resolver, params):
            resolution_times["start_date"] = time.time()
            return Fact(name="start_date", value="2024-01-01", source=FactSource.RULE)

        @resolver.rule("end_date")
        def resolve_end(resolver, params):
            start = resolver.resolve("start_date")
            resolution_times["end_date"] = time.time()
            return Fact(
                name="end_date",
                value="2024-03-31",
                source=FactSource.RULE,
                because=[start]
            )

        result = resolver.resolve("end_date")

        assert result.value == "2024-03-31"
        assert len(result.because) == 1
        assert result.because[0].value == "2024-01-01"
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
        assert call_counts == {"A": 1, "B": 1, "C": 1, "D": 1}


class TestDAGDiscovery:
    """Tests for DAG discovery through recursive resolution."""

    def test_dag_discovered_through_resolution(self):
        """Test that the dependency DAG is discovered during resolution."""
        resolver = FactResolver()

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

        assert result.is_resolved
        assert result.value == "final_answer"

        assert len(result.because) == 2
        dep_names = {f.name for f in result.because}
        assert dep_names == {"trend_data", "tier_summary"}

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

        assert len(audit_log) == 4

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

        @resolver.rule("fact_x")
        def rx(res, p):
            return Fact(name="fact_x", value="X", source=FactSource.RULE)

        @resolver.rule("fact_y")
        def ry(res, p):
            return Fact(name="fact_y", value="Y", source=FactSource.RULE)

        @resolver.rule("fact_z")
        def rz(res, p):
            return Fact(name="fact_z", value="Z", source=FactSource.RULE)

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

        @resolver.rule("config_a")
        def ra(res, p):
            resolution_order.append("config_a")
            return Fact(name="config_a", value="A", source=FactSource.RULE)

        @resolver.rule("config_b")
        def rb(res, p):
            resolution_order.append("config_b")
            return Fact(name="config_b", value="B", source=FactSource.RULE)

        @resolver.rule("combined")
        def rc(res, p):
            resolution_order.append("combined_start")
            a = res.resolve("config_a")
            b = res.resolve("config_b")
            resolution_order.append("combined_end")
            return Fact(name="combined", value=f"{a.value}+{b.value}", source=FactSource.RULE, because=[a, b])

        result = resolver.resolve("combined")

        assert result.value == "A+B"
        assert resolution_order[0] == "combined_start"
        assert "config_a" in resolution_order
        assert "config_b" in resolution_order
        assert resolution_order[-1] == "combined_end"


class TestUserProvidedReSteer:
    """Tests for USER_PROVIDED as re-steer trigger."""

    def test_user_provided_fact_from_cache(self):
        """Test that user-provided facts are retrieved from cache."""
        resolver = FactResolver()

        resolver.add_user_fact("customer_tier_threshold", 100000, reasoning="User specified VIP threshold")

        result = resolver.resolve("customer_tier_threshold")

        assert result.is_resolved
        assert result.value == 100000
        assert result.source == FactSource.USER_PROVIDED

    def test_unresolved_when_user_fact_missing(self):
        """Test that resolution returns UNRESOLVED when user fact not provided."""
        resolver = FactResolver()

        result = resolver.resolve("undefined_user_fact")

        assert not result.is_resolved
        assert result.source == FactSource.UNRESOLVED

    def test_user_fact_usable_in_derived_computation(self):
        """Test that user-provided facts can be used in derived computations."""
        resolver = FactResolver()

        resolver.add_user_fact("period", "last_3_months", reasoning="User clarified time period")

        @resolver.rule("date_range")
        def resolve_dates(resolver, params):
            period = resolver.resolve("period")
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
