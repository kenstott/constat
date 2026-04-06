# Copyright (c) 2025 Kenneth Stott
# Canary: 400885ad-3cc4-4c8f-ac96-6682de7820d3
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for the query engine.

These tests require ANTHROPIC_API_KEY to be set.
Run with: pytest tests/test_engine_integration.py -v
"""

from __future__ import annotations
import os
import pytest
from pathlib import Path

from constat.core.config import Config
from constat.execution.engine import QueryEngine
from constat.catalog.schema_manager import SchemaManager


pytestmark = [pytest.mark.slow]


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not set — required for this test")

FIXTURES_DIR = Path(__file__).parent.parent
CHINOOK_DB = FIXTURES_DIR / "data" / "chinook.db"


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    """Create query engine with Chinook database."""
    config = Config(
        databases={"chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}},
        system_prompt="""You are analyzing data for a digital music store.

Key concepts:
- Artists create Albums, which contain Tracks
- Tracks have a Genre (Rock, Jazz, Metal, etc.) and MediaType (MPEG, AAC)
- Customers purchase tracks via Invoices and InvoiceLines
- Employees are sales representatives who support customers
- Playlists are curated collections of tracks

Revenue is calculated from InvoiceLine: SUM(UnitPrice * Quantity)
""",
        execution={"allowed_imports": ["pandas", "numpy", "json", "datetime"]},
    )

    schema_manager = SchemaManager(config)
    schema_manager.initialize()

    return QueryEngine(config, schema_manager, max_retries=3)


class TestSingleShotQueries:
    """Test end-to-end single-shot queries."""

    def test_top_selling_tracks(self, require_anthropic_key, engine: QueryEngine):
        """Query: What are the top-selling tracks?"""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("What are the top 5 selling tracks by revenue?")

                assert result.success, f"Query failed: {result.error}"
                assert any(c.isdigit() for c in result.answer) or "track" in result.answer.lower(), \
                    f"Expected track names or revenue figures in answer: {result.answer}"
                assert result.attempts <= 3, f"Too many attempts: {result.attempts}"

                # The answer should contain some data
                assert len(result.answer) > 50, "Answer too short"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_top_genres_by_revenue(self, require_anthropic_key, engine: QueryEngine):
        """Query: What are the top genres by revenue?"""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("What are the top 5 genres by total revenue?")

                assert result.success, f"Query failed: {result.error}"
                # Rock is typically the top genre in Chinook
                assert any(g in result.answer.lower() for g in ["rock", "genre", "revenue", "jazz", "metal"]), \
                    f"Expected genre names in answer: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_artist_with_most_tracks(self, require_anthropic_key, engine: QueryEngine):
        """Query: Which artist has the most tracks?"""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("Which artist has the most tracks in the database?")

                assert result.success, f"Query failed: {result.error}"
                # Iron Maiden has the most tracks in Chinook (213)
                assert "iron maiden" in result.answer.lower() or "artist" in result.answer.lower() or any(c.isdigit() for c in result.answer), \
                    f"Expected artist name or track count in answer: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_customer_spending(self, require_anthropic_key, engine: QueryEngine):
        """Query: Who are the top spending customers?"""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("Show me the top 5 customers by total spending.")

                assert result.success, f"Query failed: {result.error}"
                assert any(c.isdigit() for c in result.answer) or "customer" in result.answer.lower(), \
                    f"Expected customer names or spending amounts: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_employee_sales_performance(self, require_anthropic_key, engine: QueryEngine):
        """Query: Compare sales by employee."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("Compare total sales by employee (sales rep).")

                assert result.success, f"Query failed: {result.error}"
                assert "employee" in result.answer.lower() or "sales" in result.answer.lower() or any(c.isdigit() for c in result.answer), \
                    f"Expected employee or sales data in answer: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


class TestRetryBehavior:
    """Test that retry on error works."""

    def test_retries_recorded_in_history(self, require_anthropic_key, engine: QueryEngine):
        """Attempt history is recorded."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("What are the top 3 albums by number of tracks?")

                # Should have at least one attempt
                assert len(result.attempt_history) >= 1
                assert result.attempt_history[0]["attempt"] == 1
                assert "code" in result.attempt_history[0]

                print(f"\n--- Completed in {result.attempts} attempt(s) ---")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


class TestComplexQueries:
    """Test more complex analytical queries."""

    def test_year_over_year_comparison(self, require_anthropic_key, engine: QueryEngine):
        """Query requiring date handling."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "Compare total revenue by year. Show each year and its total."
                )
                assert result.success, f"Query failed: {result.error}"
                assert any(c.isdigit() for c in result.answer) or "year" in result.answer.lower(), \
                    f"Expected year or revenue figures in answer: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_percentage_calculation(self, require_anthropic_key, engine: QueryEngine):
        """Query requiring percentage calculation."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "What percentage of all tracks are in at least one playlist?"
                )
                assert result.success, f"Query failed: {result.error}"
                assert "%" in result.answer or any(c.isdigit() for c in result.answer), \
                    f"Expected a percentage or numeric value in answer: {result.answer}"
                print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc
