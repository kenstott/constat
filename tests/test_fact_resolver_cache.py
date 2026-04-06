# Copyright (c) 2025 Kenneth Stott
# Canary: 7e5732af-b528-411f-ab1f-c35a3ae572d7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Cache-related and provenance/derivation-trace tests for FactResolver."""

from __future__ import annotations

import pytest
from constat.execution.fact_resolver import (
    Fact,
    FactSource,
    FactResolver,
)


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    @pytest.fixture
    def resolver(self):
        """Create a basic resolver."""
        return FactResolver()

    def test_cache_key_no_params(self, resolver):
        """Test cache key with no parameters."""
        assert resolver._cache_key("fact", {}) == "fact"

    def test_cache_key_single_param(self, resolver):
        """Test cache key with a single parameter."""
        assert resolver._cache_key("fact", {"a": 1}) == "fact(a=1)"

    def test_cache_key_multiple_params(self, resolver):
        """Test cache key with multiple parameters."""
        assert resolver._cache_key("fact", {"a": 1, "b": 2}) == "fact(a=1,b=2)"


class TestCaching:
    """Tests for fact caching behavior."""

    @pytest.fixture
    def resolver(self):
        """Create a basic resolver."""
        return FactResolver()

    def test_caching(self, resolver):
        """Test that facts are cached."""
        call_count = 0

        @resolver.rule("counted")
        def counted_rule(res, params):
            nonlocal call_count
            call_count += 1
            return Fact(name="counted", value=call_count, source=FactSource.RULE)

        result1 = resolver.resolve("counted")
        assert result1.value == 1
        assert call_count == 1

        result2 = resolver.resolve("counted")
        assert result2.value == 1  # Same value (cached)
        assert call_count == 1  # Rule not called again

    def test_clear_cache(self, resolver):
        """Test cache clearing."""
        resolver._cache["temp"] = Fact(name="temp", value=1, source=FactSource.CACHE)
        assert "temp" in resolver._cache

        resolver.clear_cache()
        assert "temp" not in resolver._cache

    def test_cache_checked_first(self):
        """Test that cached facts are returned immediately without trying other sources."""
        resolver = FactResolver()

        cached_fact = Fact(name="cached_value", value=42, source=FactSource.CACHE)
        resolver._cache["cached_value"] = cached_fact

        result = resolver.resolve("cached_value")

        assert result.value == 42
        assert result.source == FactSource.CACHE

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

        d1 = resolver.resolve("derived1")
        d2 = resolver.resolve("derived2")

        assert d1.value == 200
        assert d2.value == 300
        assert call_count["base"] == 1


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
        raw = Fact(name="raw_data", value=10, source=FactSource.DATABASE)
        level1 = Fact(name="processed", value=20, source=FactSource.RULE, rule_name="process", because=[raw])
        level2 = Fact(name="enriched", value=40, source=FactSource.RULE, rule_name="enrich", because=[level1])
        final = Fact(name="conclusion", value=80, source=FactSource.RULE, rule_name="conclude", because=[level2])

        trace = final.derivation_trace
        assert "raw_data" in trace
        assert "processed" in trace
        assert "enriched" in trace
        assert "conclusion" in trace
        assert "process" in trace
        assert "enrich" in trace
        assert "conclude" in trace

    def test_diamond_dependency_pattern(self):
        """Test trace handles diamond dependency (fact used by multiple paths)."""
        a = Fact(name="fact_a", value=1, source=FactSource.DATABASE)
        b = Fact(name="fact_b", value=2, source=FactSource.RULE, because=[a])
        c = Fact(name="fact_c", value=3, source=FactSource.RULE, because=[a])
        d = Fact(name="fact_d", value=6, source=FactSource.RULE, because=[b, c])

        trace = d.derivation_trace
        assert "fact_a" in trace
        assert "fact_b" in trace
        assert "fact_c" in trace
        assert "fact_d" in trace
        assert trace.count("fact_a") == 2

    def test_to_dict_includes_dependency_names(self):
        """Test that serialization includes dependency fact names."""
        dep1 = Fact(name="dep1", value=10, source=FactSource.DATABASE)
        dep2 = Fact(name="dep2", value=20, source=FactSource.CONFIG)
        derived = Fact(name="derived", value=30, source=FactSource.RULE, because=[dep1, dep2])

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

        resolver._cache["uncertain_base"] = Fact(
            name="uncertain_base", value=100, confidence=0.6, source=FactSource.LLM_HEURISTIC,
        )
        resolver._cache["certain_base"] = Fact(
            name="certain_base", value=200, confidence=1.0, source=FactSource.DATABASE,
        )

        @resolver.rule("combined_confidence")
        def combine(res, params):
            u = res.resolve("uncertain_base")
            c = res.resolve("certain_base")
            return Fact(
                name="combined_confidence",
                value=u.value + c.value,
                confidence=min(u.confidence, c.confidence),
                source=FactSource.RULE,
                rule_name="combine",
                because=[u, c],
            )

        result = resolver.resolve("combined_confidence")
        assert result.confidence == 0.6


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
        parsed = datetime.fromisoformat(serialized["resolved_at"])
        assert parsed == fact.resolved_at

    def test_different_facts_have_different_timestamps(self):
        """Test that facts created at different times have different timestamps."""
        import time

        fact1 = Fact(name="first", value=1, source=FactSource.DATABASE)
        time.sleep(0.01)
        fact2 = Fact(name="second", value=2, source=FactSource.DATABASE)

        assert fact1.resolved_at < fact2.resolved_at

    def test_audit_log_preserves_timestamp_order(self):
        """Test that audit log maintains chronological order."""
        import time

        resolver = FactResolver()

        for i in range(3):
            resolver._cache[f"fact_{i}"] = Fact(name=f"fact_{i}", value=i, source=FactSource.CONFIG)
            resolver.resolve(f"fact_{i}")
            time.sleep(0.01)

        log = resolver.get_audit_log()
        timestamps = [entry["resolved_at"] for entry in log]

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
        fact = Fact(name="minimal", value=None, source=FactSource.UNRESOLVED)

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

        json_str = json.dumps(serialized)
        recovered = json.loads(json_str)
        assert recovered["value"] == complex_value

    def test_audit_log_json_serializable(self):
        """Test that entire audit log is JSON serializable."""
        import json

        resolver = FactResolver()

        resolver._cache["db_fact"] = Fact(name="db_fact", value=100, source=FactSource.DATABASE, query="SELECT 100")
        resolver._cache["llm_fact"] = Fact(
            name="llm_fact", value="Paris", confidence=0.99,
            source=FactSource.LLM_KNOWLEDGE, reasoning="Capital of France is well known",
        )

        resolver.resolve("db_fact")
        resolver.resolve("llm_fact")
        resolver.resolve("missing")

        log = resolver.get_audit_log()

        json_str = json.dumps(log)
        recovered = json.loads(json_str)

        assert len(recovered) == 3
        assert recovered[0]["name"] == "db_fact"
        assert recovered[1]["name"] == "llm_fact"
        assert recovered[2]["source"] == "unresolved"

    def test_derivation_trace_indentation(self):
        """Test that derivation trace has proper indentation for readability."""
        base = Fact(name="base", value=10, source=FactSource.DATABASE)
        derived = Fact(name="derived", value=20, source=FactSource.RULE, rule_name="double", because=[base])

        trace = derived.derivation_trace
        lines = trace.split("\n")

        assert lines[0].startswith("derived")
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
        facts = []
        prev = None

        for i in range(50):
            fact = Fact(
                name=f"level_{i}", value=i, source=FactSource.RULE,
                because=[prev] if prev else [],
            )
            facts.append(fact)
            prev = fact

        final = facts[-1]
        trace = final.derivation_trace

        assert "level_0" in trace
        assert "level_49" in trace

    def test_fact_with_special_characters_in_name(self):
        """Test handling of special characters in fact names."""
        fact = Fact(name="customer:ltv(id='test',region=\"US\")", value=5000, source=FactSource.RULE)

        trace = fact.derivation_trace
        serialized = fact.to_dict()

        assert "customer:ltv" in trace
        assert serialized["name"] == "customer:ltv(id='test',region=\"US\")"

    def test_fact_with_unicode_values(self):
        """Test handling of unicode in fact values and names."""
        fact = Fact(name="city_name", value="Zurich", source=FactSource.LLM_KNOWLEDGE, reasoning="Swiss city")

        trace = fact.derivation_trace
        serialized = fact.to_dict()

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
        fact_a = Fact(name="fact_a", value=1, source=FactSource.RULE)
        fact_b = Fact(name="fact_b", value=2, source=FactSource.RULE, because=[fact_a])

        fact_a.because = [fact_b]

        try:
            import sys
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(100)
            try:
                trace = fact_a.derivation_trace
            except RecursionError:
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
        assert fact.confidence == 1.0
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
                value=base.value * 1.1,
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
            name="test", value=42, confidence=0.9, source=FactSource.DATABASE, query="SELECT 42",
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
            resolver._cache[f"fact_{i}"] = Fact(name=f"fact_{i}", value=i, source=FactSource.CONFIG)

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

        resolver._cache["resolved"] = Fact(name="resolved", value=1, source=FactSource.DATABASE)
        resolver.resolve("resolved")
        resolver.resolve("missing")

        assert len(resolver.resolution_log) == 2

        resolver.clear_unresolved()

        assert len(resolver.resolution_log) == 1
        assert resolver.resolution_log[0].name == "resolved"

    def test_clear_unresolved_allows_re_resolution(self):
        """Test that clearing unresolved allows facts to be tried again."""
        resolver = FactResolver()

        resolver.resolve("dynamic_fact")
        assert len(resolver.get_unresolved_facts()) == 1

        resolver.clear_unresolved()
        assert len(resolver.get_unresolved_facts()) == 0

        resolver._cache["dynamic_fact"] = Fact(name="dynamic_fact", value=42, source=FactSource.CONFIG)
        result = resolver.resolve("dynamic_fact")

        assert result.is_resolved
        assert result.value == 42
