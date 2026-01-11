"""Integration tests for the query engine.

These tests require ANTHROPIC_API_KEY to be set.
Run with: pytest tests/test_engine_integration.py -v
"""

import os
import pytest
from pathlib import Path

from constat.core.config import Config
from constat.execution.engine import QueryEngine
from constat.catalog.schema_manager import SchemaManager


# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)

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

    def test_top_selling_tracks(self, engine: QueryEngine):
        """Query: What are the top-selling tracks?"""
        result = engine.query("What are the top 5 selling tracks by revenue?")

        assert result.success, f"Query failed: {result.error}"
        assert result.answer, "No answer returned"
        assert result.attempts <= 3, f"Too many attempts: {result.attempts}"

        # Should mention some track names or revenue amounts
        answer_lower = result.answer.lower()
        # The answer should contain some data
        assert len(result.answer) > 50, "Answer too short"
        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")

    def test_top_genres_by_revenue(self, engine: QueryEngine):
        """Query: What are the top genres by revenue?"""
        result = engine.query("What are the top 5 genres by total revenue?")

        assert result.success, f"Query failed: {result.error}"
        assert result.answer, "No answer returned"

        # Rock is typically the top genre in Chinook
        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")

    def test_artist_with_most_tracks(self, engine: QueryEngine):
        """Query: Which artist has the most tracks?"""
        result = engine.query("Which artist has the most tracks in the database?")

        assert result.success, f"Query failed: {result.error}"
        assert result.answer, "No answer returned"

        # Iron Maiden has the most tracks in Chinook (213)
        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")

    def test_customer_spending(self, engine: QueryEngine):
        """Query: Who are the top spending customers?"""
        result = engine.query("Show me the top 5 customers by total spending.")

        assert result.success, f"Query failed: {result.error}"
        assert result.answer, "No answer returned"

        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")

    def test_employee_sales_performance(self, engine: QueryEngine):
        """Query: Compare sales by employee."""
        result = engine.query("Compare total sales by employee (sales rep).")

        assert result.success, f"Query failed: {result.error}"
        assert result.answer, "No answer returned"

        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")


class TestRetryBehavior:
    """Test that retry on error works."""

    def test_retries_recorded_in_history(self, engine: QueryEngine):
        """Attempt history is recorded."""
        result = engine.query("What are the top 3 albums by number of tracks?")

        # Should have at least one attempt
        assert len(result.attempt_history) >= 1
        assert result.attempt_history[0]["attempt"] == 1
        assert "code" in result.attempt_history[0]

        print(f"\n--- Completed in {result.attempts} attempt(s) ---")


class TestComplexQueries:
    """Test more complex analytical queries."""

    def test_year_over_year_comparison(self, engine: QueryEngine):
        """Query requiring date handling."""
        result = engine.query(
            "Compare total revenue by year. Show each year and its total."
        )

        assert result.success, f"Query failed: {result.error}"
        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")

    def test_percentage_calculation(self, engine: QueryEngine):
        """Query requiring percentage calculation."""
        result = engine.query(
            "What percentage of all tracks are in at least one playlist?"
        )

        assert result.success, f"Query failed: {result.error}"
        print(f"\n--- Answer ({result.attempts} attempts) ---\n{result.answer}")
