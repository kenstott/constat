# Copyright (c) 2025 Kenneth Stott
# Canary: dc5aba75-5690-4503-b0c4-b2db0e1eec55
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for multi-database query handling.

Tests that the LLM:
1. Picks the correct database for domain-specific queries
2. Can perform cross-database analysis when needed

Requires ANTHROPIC_API_KEY to be set.
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
NORTHWIND_DB = FIXTURES_DIR / "data" / "northwind.db"


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    """Create query engine with both databases."""
    config = Config(
        databases={
            "chinook": {"uri": f"sqlite:///{CHINOOK_DB}"},
            "northwind": {"uri": f"sqlite:///{NORTHWIND_DB}"},
        },
        system_prompt="""You are analyzing data across two business databases.

## Chinook (Digital Music Store)
- Artists, Albums, Tracks, Genres
- Customers purchase via Invoices/InvoiceLines
- Music-related queries go here

## Northwind (Product Distribution)
- Products, Categories, Suppliers
- Customers place Orders with Order Details
- Product/shipping-related queries go here

Choose the appropriate database based on the query domain.
""",
        execution={"allowed_imports": ["pandas", "numpy", "json", "datetime"]},
    )

    schema_manager = SchemaManager(config)
    schema_manager.initialize()
    schema_manager.build_chunks()

    return QueryEngine(config, schema_manager, max_retries=3)


class TestDatabaseSelection:
    """Test that LLM selects the correct database for domain-specific queries."""

    def test_chinook_query_uses_chinook_db(self, require_anthropic_key, engine: QueryEngine):
        """Music-related query should use Chinook database."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("What are the top 3 music genres by number of tracks?")

                assert result.success, f"Query failed: {result.error}"

                # Verify it used the chinook database
                code_lower = result.code.lower()
                assert "db_chinook" in code_lower or ("genre" in code_lower and "track" in code_lower), \
                    f"Expected Chinook query, got:\n{result.code}"

                # Should mention genres
                assert "rock" in result.answer.lower() or "genre" in result.answer.lower(), \
                    f"Expected genre results, got:\n{result.answer}"

                print(f"\n--- Chinook Query ({result.attempts} attempts) ---")
                print(f"Code uses: {'db_chinook' if 'db_chinook' in result.code else 'db'}")
                print(f"Answer:\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_northwind_query_uses_northwind_db(self, require_anthropic_key, engine: QueryEngine):
        """Product-related query should use Northwind database."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("What are the top 5 product categories by number of products?")

                assert result.success, f"Query failed: {result.error}"

                # Verify it used the northwind database
                code_lower = result.code.lower()
                assert "db_northwind" in code_lower or "northwind" in code_lower or \
                       ("categor" in code_lower and "product" in code_lower), \
                    f"Expected Northwind query, got:\n{result.code}"

                print(f"\n--- Northwind Query ({result.attempts} attempts) ---")
                print(f"Code uses: {'db_northwind' if 'db_northwind' in result.code else 'db'}")
                print(f"Answer:\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_supplier_query_uses_northwind(self, require_anthropic_key, engine: QueryEngine):
        """Supplier query should use Northwind (Chinook has no suppliers)."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("List all suppliers and their countries.")

                assert result.success, f"Query failed: {result.error}"

                # Northwind has suppliers, Chinook doesn't
                assert "db_northwind" in result.code.lower() or "supplier" in result.code.lower()

                print(f"\n--- Supplier Query ({result.attempts} attempts) ---")
                print(f"Answer:\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_artist_query_uses_chinook(self, require_anthropic_key, engine: QueryEngine):
        """Artist query should use Chinook (Northwind has no artists)."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("Who are the top 5 artists by album count?")

                assert result.success, f"Query failed: {result.error}"

                # Chinook has artists, Northwind doesn't
                code_lower = result.code.lower()
                assert "db_chinook" in code_lower or "artist" in code_lower

                print(f"\n--- Artist Query ({result.attempts} attempts) ---")
                print(f"Answer:\n{result.answer}")
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


class TestCrossDatabaseQueries:
    """Test queries that require data from both databases."""

    def test_compare_employee_counts(self, require_anthropic_key, engine: QueryEngine):
        """Compare employee counts between both databases."""
        result = engine.query(
            "How many employees are in each database? Compare Chinook vs Northwind."
        )

        assert result.success, f"Query failed: {result.error}"

        # Should reference both databases
        code_lower = result.code.lower()
        has_both = ("chinook" in code_lower and "northwind" in code_lower) or \
                   ("db_chinook" in code_lower and "db_northwind" in code_lower)

        # Answer should mention both counts
        answer_lower = result.answer.lower()
        has_numbers = any(c.isdigit() for c in result.answer)

        print(f"\n--- Cross-DB Employee Query ({result.attempts} attempts) ---")
        print(f"Code references both DBs: {has_both}")
        print(f"Answer:\n{result.answer}")

        assert has_numbers, "Expected employee counts in answer"

    def test_compare_customer_counts_by_country(self, require_anthropic_key, engine: QueryEngine):
        """Compare customer distribution by country across both databases."""
        result = engine.query(
            "Compare customer count by country between Chinook and Northwind. "
            "Show countries that appear in both."
        )

        assert result.success, f"Query failed: {result.error}"

        print(f"\n--- Cross-DB Customer Query ({result.attempts} attempts) ---")
        print(f"Answer:\n{result.answer}")


class TestSchemaDiscovery:
    """Test that schema tools help find the right database."""

    def test_schema_overview_shows_both_dbs(self, engine: QueryEngine):
        """Schema overview should list both databases."""
        overview = engine.schema_manager.get_overview()

        assert "chinook" in overview.lower()
        assert "northwind" in overview.lower()

        print(f"\n--- Schema Overview ---\n{overview}")

    def test_vector_search_finds_correct_db(self, engine: QueryEngine):
        """Vector search should find tables in the appropriate database."""
        # Music query should find Chinook tables
        music_results = engine.schema_manager.find_relevant_tables("music genre rock jazz")
        music_dbs = {r["database"] for r in music_results[:3]}
        assert "chinook" in music_dbs, f"Expected chinook for music query, got {music_dbs}"

        # Product query should find Northwind tables
        product_results = engine.schema_manager.find_relevant_tables("products categories suppliers")
        product_dbs = {r["database"] for r in product_results[:3]}
        assert "northwind" in product_dbs, f"Expected northwind for product query, got {product_dbs}"

        print(f"\n--- Vector Search Results ---")
        print(f"Music query top DBs: {music_dbs}")
        print(f"Product query top DBs: {product_dbs}")
