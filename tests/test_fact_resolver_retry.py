# Copyright (c) 2025 Kenneth Stott
# Canary: 7e5732af-b528-411f-ab1f-c35a3ae572d7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Retry behavior tests for RateLimiter in FactResolver."""

from __future__ import annotations

import time
import pytest
from constat.execution.fact_resolver import (
    RateLimiter,
    RateLimiterConfig,
    RateLimitExhaustedError,
)


class TestRateLimiterRetry:
    """Tests for retry behavior in RateLimiter."""

    @pytest.mark.asyncio
    async def test_rate_limiter_exponential_backoff(self):
        """Test exponential backoff on rate limit errors."""
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


# =============================================================================
# CONCLUSION / GOAL / PROLOG / SQL TRANSFORM TESTS
# =============================================================================


class TestResolveConclusion:
    """Tests for template-based symbolic resolution via resolve_conclusion."""

    def test_resolve_conclusion_without_llm_returns_error(self):
        """Test that resolve_conclusion requires an LLM."""
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver
        resolver = FactResolver()
        result = resolver.resolve_conclusion("What is the revenue?")
        assert "error" in result
        assert result["answer"] is None

    def test_resolve_conclusion_generates_template(self):
        """Test that resolve_conclusion generates a template with variables."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            """TEMPLATE: The total revenue for {time_period} was ${total_revenue}.
VARIABLES:
- {time_period}: The time period to analyze
- {total_revenue}: Sum of all revenue in the period""",
            """{time_period}: []
{total_revenue}: [{time_period}]""",
        ]

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("time_period", "Q3 2024")
        resolver.add_user_fact("total_revenue", 150000)

        result = resolver.resolve_conclusion("What was the revenue for Q3?")

        assert "template" in result
        assert "variables" in result
        assert "time_period" in result["variables"]
        assert "total_revenue" in result["variables"]

    def test_resolve_conclusion_parallel_independent_vars(self):
        """Test that independent variables are resolved together via resolve_many_sync."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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

        original_resolve_many = resolver.resolve_many_sync

        def tracking_resolve_many(requests):
            resolve_calls.append(requests)
            return original_resolve_many(requests)

        resolver.resolve_many_sync = tracking_resolve_many

        resolver.add_user_fact("region", "US")
        resolver.add_user_fact("metric_a", 100)
        resolver.add_user_fact("metric_b", 200)

        result = resolver.resolve_conclusion("Show me metrics for US")

        assert len(resolve_calls) == 1
        assert len(resolve_calls[0]) == 3

    def test_resolve_conclusion_substitutes_values(self):
        """Test that resolved values are substituted into the template."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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

        assert "Q3" in result["answer"]
        assert "50000" in result["answer"]

    def test_resolve_conclusion_tracks_unresolved(self):
        """Test that unresolved variables are tracked."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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
        resolver.add_user_fact("customer_id", "ACME-001")

        result = resolver.resolve_conclusion("What tier is customer ACME?")

        assert "tier" in result["unresolved"]
        assert "customer_id" not in result["unresolved"]

    def test_resolve_conclusion_builds_derivation_trace(self):
        """Test that derivation trace is built correctly."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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

        assert "derivation" in result
        assert "Statement" in result["derivation"]
        assert "Variable Resolution" in result["derivation"]
        assert "Conclusion" in result["derivation"]

    def test_resolve_conclusion_calculates_confidence(self):
        """Test that confidence is calculated as min of all facts."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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
        resolver._cache["a"] = Fact(name="a", value=100, confidence=1.0, source=FactSource.DATABASE)
        resolver._cache["b"] = Fact(name="b", value=200, confidence=0.7, source=FactSource.LLM_KNOWLEDGE)

        result = resolver.resolve_conclusion("Show a and b")

        assert result["confidence"] == 0.7


class TestBuildDerivationTrace:
    """Tests for the _build_derivation_trace helper method."""

    def test_build_derivation_trace_includes_all_sections(self):
        """Test that derivation trace includes all required sections."""
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

        resolver = FactResolver()
        fact = Fact(
            name="revenue", value=50000, confidence=0.9, source=FactSource.DATABASE,
            source_name="sales_db", query="SELECT SUM(amount) FROM sales",
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
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

        resolver = FactResolver()
        base = Fact(name="base", value=10, source=FactSource.DATABASE)
        derived = Fact(name="derived", value=20, confidence=1.0, source=FactSource.RULE, because=[base])
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
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        result = resolver.resolve_goal("What is the revenue?")
        assert "error" in result
        assert result["answer"] is None

    def test_resolve_goal_parses_prolog_response(self):
        """Test that Prolog-style LLM response is parsed correctly."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import FactResolver

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
        resolver.add_user_fact("date_range", ("2024-07-01", "2024-09-30"))
        resolver.add_user_fact("sum_sales", 150000)

        result = resolver.resolve_goal("What was revenue in Q3?")

        assert result["goal"] == "total_revenue(q3, Revenue)"
        assert len(result["rules"]) >= 1
        assert "date_range" in result["sources"]
        assert "sum_sales" in result["sources"]

    def test_parse_predicate(self):
        """Test predicate parsing helper."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()

        name, args = resolver._parse_predicate("foo(X, Y, Z)")
        assert name == "foo"
        assert args == ["X", "Y", "Z"]

        name, args = resolver._parse_predicate("revenue(q3, premium, Amount)")
        assert name == "revenue"
        assert args == ["q3", "premium", "Amount"]

        name, args = resolver._parse_predicate("fact")
        assert name == "fact"
        assert args == []

    def test_parse_rules_extracts_dependencies(self):
        """Test that rule parsing extracts head -> body dependencies."""
        from constat.execution.fact_resolver import FactResolver
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
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import FactResolver

        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: value(X)

RULES:

SOURCES:
value: DATABASE"""

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("value", 42)

        result = resolver.resolve_goal("What is the value?")

        deriv = result["derivation"]
        assert "/* Query */" in deriv
        assert "?- value(X)" in deriv
        assert "/* Resolution */" in deriv
        assert "/* Answer */" in deriv

    def test_resolve_goal_binds_variables(self):
        """Test that variables are bound during resolution."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import FactResolver

        mock_llm = MagicMock()
        mock_llm.generate.return_value = """GOAL: answer(Result)

RULES:

SOURCES:
answer: DATABASE"""

        resolver = FactResolver(llm=mock_llm)
        resolver.add_user_fact("answer", 100)

        result = resolver.resolve_goal("What is the answer?")

        assert "Result" in result["bindings"]
        assert result["bindings"]["Result"] == 100

    def test_resolve_goal_tracks_unresolved(self):
        """Test that unresolved predicates are tracked."""
        from unittest.mock import MagicMock
        from constat.execution.fact_resolver import FactResolver

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
        resolver.add_user_fact("known", "value_x")

        result = resolver.resolve_goal("Get result")

        assert "unknown" in result["unresolved"]

    def test_substitute_bindings(self):
        """Test variable substitution in terms."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()

        bindings = {"X": 10, "Y": 20, "Result": 30}
        term = "answer(X, Y, Result)"

        substituted = resolver._substitute_bindings(term, bindings)

        assert substituted == "answer(10, 20, 30)"


class TestPrologDerivationTrace:
    """Tests for Prolog-style derivation trace building."""

    def test_build_prolog_derivation_includes_query(self):
        """Test that derivation includes the original query."""
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

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
        from constat.execution.fact_resolver import FactResolver

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
        from constat.execution.fact_resolver import Fact, FactSource, FactResolver

        resolver = FactResolver()
        fact = Fact(
            name="revenue", value=100000, confidence=1.0, source=FactSource.DATABASE,
            source_name="sales_db", query="SELECT SUM(amount) FROM sales",
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
        from constat.execution.fact_resolver import FactResolver

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
        from constat.execution.fact_resolver import FactResolver

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
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT DATE_FORMAT(order_date, '%Y-%m') as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "strftime('%Y-%m', order_date)" in result
        assert "DATE_FORMAT" not in result

    def test_date_sub_with_curdate(self):
        """Test DATE_SUB(CURDATE(), INTERVAL n MONTH) conversion."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT * FROM orders WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date('now', '-6 months')" in result
        assert "DATE_SUB" not in result
        assert "CURDATE" not in result

    def test_date_sub_with_column(self):
        """Test DATE_SUB with column reference."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT DATE_SUB(created_at, INTERVAL 30 DAY) as past_date FROM users"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date(created_at, '-30 days')" in result

    def test_curdate_standalone(self):
        """Test standalone CURDATE() conversion."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT * FROM orders WHERE order_date = CURDATE()"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "date('now')" in result
        assert "CURDATE" not in result

    def test_now_conversion(self):
        """Test NOW() conversion to datetime('now')."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT * FROM logs WHERE created_at < NOW()"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "datetime('now')" in result
        assert "NOW()" not in result

    def test_year_function(self):
        """Test YEAR() conversion to strftime."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT YEAR(order_date) as year FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%Y', order_date) AS INTEGER)" in result
        assert "YEAR(" not in result

    def test_month_function(self):
        """Test MONTH() conversion to strftime."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT MONTH(order_date) as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%m', order_date) AS INTEGER)" in result
        assert "MONTH(" not in result

    def test_extract_year(self):
        """Test EXTRACT(YEAR FROM col) conversion."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT EXTRACT(YEAR FROM order_date) as year FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%Y', order_date) AS INTEGER)" in result
        assert "EXTRACT" not in result

    def test_extract_month(self):
        """Test EXTRACT(MONTH FROM col) conversion."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT EXTRACT(MONTH FROM order_date) as month FROM orders"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "CAST(strftime('%m', order_date) AS INTEGER)" in result
        assert "EXTRACT" not in result

    def test_complex_query_transformation(self):
        """Test transformation of complex query with multiple MySQL functions."""
        from constat.execution.fact_resolver import FactResolver
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
        assert "DATE_FORMAT" not in result
        assert "DATE_SUB" not in result
        assert "CURDATE" not in result
        assert "strftime('%Y-%m', o.order_date)" in result
        assert "date('now', '-6 months')" in result

    def test_case_insensitive_transformation(self):
        """Test that transformations are case-insensitive."""
        from constat.execution.fact_resolver import FactResolver
        resolver = FactResolver()
        sql = "SELECT date_format(created_at, '%Y') as year, curdate() as today FROM users"
        result = resolver._transform_sql_for_sqlite(sql)
        assert "strftime('%Y', created_at)" in result
        assert "date('now')" in result
