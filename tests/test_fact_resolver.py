"""Tests for lazy fact resolution with provenance tracking."""

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
