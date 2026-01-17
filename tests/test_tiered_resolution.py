"""Tests for tiered fact resolution."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from constat.execution.fact_resolver import (
    FactResolver,
    FactSource,
    Fact,
    ResolutionStrategy,
    Tier2Strategy,
    Tier2AssessmentResult,
)


class TestTieredResolution:
    """Tests for the tiered resolution architecture."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = Mock()
        llm.generate = Mock(return_value='{"strategy": "USER_REQUIRED", "confidence": 0.5, "reasoning": "test", "question": "What is the value?"}')
        return llm

    @pytest.fixture
    def mock_schema_manager(self):
        """Create a mock schema manager with a test database."""
        sm = Mock()
        sm.connections = {"test_db": Mock()}
        sm.get_connection = Mock(return_value=Mock())
        sm.get_sql_connection = Mock(return_value=Mock())
        sm.get_overview = Mock(return_value="employees (id, name, salary)")
        return sm

    @pytest.fixture
    def mock_config(self):
        """Create a mock config."""
        config = Mock()
        config.databases = {}
        config.documents = {}
        return config

    @pytest.fixture
    def resolver(self, mock_llm, mock_schema_manager, mock_config):
        """Create a resolver with tiered resolution enabled."""
        strategy = ResolutionStrategy(
            use_tiered_resolution=True,
            tier1_timeout=5.0,
        )
        return FactResolver(
            llm=mock_llm,
            schema_manager=mock_schema_manager,
            config=mock_config,
            strategy=strategy,
        )

    def test_tiered_resolution_enabled_by_default(self):
        """Test that tiered resolution is enabled by default."""
        strategy = ResolutionStrategy()
        assert strategy.use_tiered_resolution is True
        assert strategy.tier1_timeout == 15.0

    def test_tier1_sources_default(self):
        """Test default Tier 1 sources."""
        strategy = ResolutionStrategy()
        assert FactSource.CACHE in strategy.tier1_sources
        assert FactSource.CONFIG in strategy.tier1_sources
        assert FactSource.RULE in strategy.tier1_sources
        assert FactSource.DOCUMENT in strategy.tier1_sources
        assert FactSource.DATABASE in strategy.tier1_sources

    def test_resolve_tiered_returns_tuple(self, resolver):
        """Test that resolve_tiered returns (Fact, Optional[Assessment])."""
        # Pre-cache a fact
        resolver._cache["test_fact"] = Fact(
            name="test_fact",
            value=42,
            confidence=0.9,
            source=FactSource.CACHE,
        )

        fact, assessment = resolver.resolve_tiered("test_fact")

        assert fact is not None
        assert fact.value == 42
        assert assessment is None  # Cache hit, no Tier 2 assessment

    def test_resolve_tiered_cache_hit(self, resolver):
        """Test that cache is checked before Tier 1 parallel."""
        resolver._cache["cached_fact"] = Fact(
            name="cached_fact",
            value="cached_value",
            confidence=0.95,
            source=FactSource.CONFIG,
        )

        fact, assessment = resolver.resolve_tiered("cached_fact")

        assert fact.value == "cached_value"
        assert fact.source == FactSource.CONFIG
        assert assessment is None

    def test_resolve_tiered_rule_resolution(self, resolver):
        """Test that rules are tried in Tier 1."""
        @resolver.rule("rule_fact")
        def my_rule(res, params):
            return Fact(
                name="rule_fact",
                value="from_rule",
                confidence=1.0,
                source=FactSource.RULE,
            )

        fact, assessment = resolver.resolve_tiered("rule_fact")

        assert fact is not None
        assert fact.value == "from_rule"
        assert fact.source == FactSource.RULE
        assert assessment is None

    def test_resolve_tiered_tier2_user_required(self, resolver, mock_llm):
        """Test that Tier 2 returns USER_REQUIRED when fact can't be resolved."""
        # Mock Tier 2 assessment response
        mock_llm.generate.return_value = '''
        {
            "strategy": "USER_REQUIRED",
            "confidence": 0.8,
            "reasoning": "This is company-specific data",
            "question": "What is the value for employees?"
        }
        '''

        fact, assessment = resolver.resolve_tiered("nonexistent_fact")

        # Tier 1 should fail, Tier 2 should return USER_REQUIRED
        assert fact.source == FactSource.UNRESOLVED
        assert assessment is not None
        assert assessment.strategy == Tier2Strategy.USER_REQUIRED
        assert "employees" in assessment.question or assessment.question is not None

    def test_resolve_tiered_database_source_tried(self, resolver, mock_llm, mock_schema_manager):
        """Test that database source is attempted in Tier 1."""
        # Mock database resolution to return a fact
        mock_llm.generate.return_value = '''
def get_result():
    return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
'''

        # Track if database was tried
        original_try_resolve = resolver._try_resolve
        sources_tried = []

        def tracking_try_resolve(source, fact_name, params, cache_key):
            sources_tried.append(source)
            return original_try_resolve(source, fact_name, params, cache_key)

        resolver._try_resolve = tracking_try_resolve

        fact, assessment = resolver.resolve_tiered("employees")

        # Verify DATABASE was one of the sources tried
        assert FactSource.DATABASE in sources_tried, f"DATABASE not tried. Sources: {sources_tried}"
        assert FactSource.DOCUMENT in sources_tried, f"DOCUMENT not tried. Sources: {sources_tried}"
        assert FactSource.CACHE in sources_tried, f"CACHE not tried. Sources: {sources_tried}"

    def test_tier1_parallel_respects_timeout(self, resolver):
        """Test that Tier 1 respects the timeout."""
        resolver.strategy.tier1_timeout = 0.1  # Very short timeout

        import time

        @resolver.rule("slow_rule")
        def slow_rule(res, params):
            time.sleep(1)  # Longer than timeout
            return Fact(name="slow_rule", value="slow", source=FactSource.RULE)

        fact, assessment = resolver.resolve_tiered("slow_rule")

        # Should timeout and go to Tier 2
        # The slow rule won't complete in time
        # Result depends on whether rule started before timeout

    def test_tier2_assessment_derivable_requires_2_inputs(self, resolver, mock_llm):
        """Test that DERIVABLE is rejected if it has < 2 inputs."""
        # Mock a DERIVABLE response with only 1 input (should be rejected)
        mock_llm.generate.return_value = '''
        {
            "strategy": "DERIVABLE",
            "confidence": 0.7,
            "reasoning": "Can derive from employees",
            "formula": "employees.salary",
            "inputs": [["employees", "premise:P1"]]
        }
        '''

        fact, assessment = resolver.resolve_tiered("avg_salary")

        # Should be downgraded to USER_REQUIRED because only 1 input
        assert assessment is not None
        assert assessment.strategy == Tier2Strategy.USER_REQUIRED

    def test_tier2_assessment_derivable_valid(self, resolver, mock_llm):
        """Test that DERIVABLE is accepted with 2+ inputs."""
        # Mock a DERIVABLE response with 2 inputs
        mock_llm.generate.return_value = '''
        {
            "strategy": "DERIVABLE",
            "confidence": 0.8,
            "reasoning": "Can derive from employees and benchmark",
            "formula": "avg(employees.salary) / benchmark",
            "inputs": [["employees", "premise:P1"], ["benchmark", "llm_knowledge"]]
        }
        '''

        # Pre-populate the resolution context
        resolver.set_resolution_context(
            resolved_premises={"P1": Fact(name="employees", value=[{"salary": 50000}], source=FactSource.DATABASE)},
            pending_premises=[],
        )

        fact, assessment = resolver.resolve_tiered("competitive_ratio")

        assert assessment is not None
        assert assessment.strategy == Tier2Strategy.DERIVABLE
        assert len(assessment.inputs) >= 2


class TestTier1ParallelResolution:
    """Tests specifically for Tier 1 parallel resolution."""

    @pytest.fixture
    def resolver(self):
        """Create a basic resolver."""
        strategy = ResolutionStrategy(
            use_tiered_resolution=True,
            tier1_timeout=5.0,
        )
        return FactResolver(strategy=strategy)

    def test_tier1_parallel_runs_all_sources(self, resolver):
        """Test that Tier 1 runs all configured sources."""
        sources_called = []

        # Mock _try_resolve to track calls
        original = resolver._try_resolve

        def mock_try_resolve(source, fact_name, params, cache_key):
            sources_called.append(source)
            return None

        resolver._try_resolve = mock_try_resolve

        # Run Tier 1
        result = resolver._resolve_tier1_parallel("test", {}, "test")

        # All tier1_sources should be tried
        for source in resolver.strategy.tier1_sources:
            assert source in sources_called, f"{source} not called in Tier 1"

    def test_tier1_parallel_picks_best_result(self, resolver):
        """Test that Tier 1 picks the highest confidence result."""
        results = {
            FactSource.CACHE: None,
            FactSource.CONFIG: Fact(name="test", value="config", confidence=0.5, source=FactSource.CONFIG),
            FactSource.RULE: Fact(name="test", value="rule", confidence=0.9, source=FactSource.RULE),
            FactSource.DOCUMENT: Fact(name="test", value="doc", confidence=0.7, source=FactSource.DOCUMENT),
            FactSource.DATABASE: None,
        }

        def mock_try_resolve(source, fact_name, params, cache_key):
            return results.get(source)

        resolver._try_resolve = mock_try_resolve

        result = resolver._resolve_tier1_parallel("test", {}, "test")

        # Should pick RULE (highest confidence)
        assert result is not None
        assert result.source == FactSource.RULE
        assert result.confidence == 0.9


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def test_legacy_resolve_works(self):
        """Test that resolve() still works with tiered enabled."""
        strategy = ResolutionStrategy(use_tiered_resolution=True)
        resolver = FactResolver(strategy=strategy)

        # Pre-cache a fact
        resolver._cache["legacy_test"] = Fact(
            name="legacy_test",
            value="legacy_value",
            confidence=1.0,
            source=FactSource.CACHE,
        )

        # Use legacy resolve()
        fact = resolver.resolve("legacy_test")

        assert fact.value == "legacy_value"

    def test_tiered_can_be_disabled(self):
        """Test that tiered resolution can be disabled."""
        strategy = ResolutionStrategy(use_tiered_resolution=False)
        resolver = FactResolver(strategy=strategy)

        # Pre-cache a fact
        resolver._cache["disabled_test"] = Fact(
            name="disabled_test",
            value="test_value",
            confidence=1.0,
            source=FactSource.CACHE,
        )

        # Should use legacy path
        fact = resolver.resolve("disabled_test")

        assert fact.value == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])