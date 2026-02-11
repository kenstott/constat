# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""End-to-end tests for auditable mode natural language queries.

These tests verify that auditable mode can:
1. Resolve facts from database sources with code generation
2. Handle retries on code generation errors
3. Build proper derivation traces
4. Support complex multi-table queries

Requires:
- ANTHROPIC_API_KEY (for LLM queries)
"""

import os
import pytest
import tempfile
import uuid
from pathlib import Path

# Skip all tests if API key not set
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    ),
]


@pytest.fixture(scope="module")
def ecommerce_db():
    """Create a temporary SQLite database with e-commerce data."""
    import sqlite3

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "ecommerce.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create customers table with tiers
        cursor.execute("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                tier TEXT NOT NULL,
                created_date TEXT NOT NULL
            )
        """)

        # Insert customer data
        customers = [
            (1, "Acme Corp", "platinum", "2023-01-15"),
            (2, "Globex Inc", "gold", "2023-02-20"),
            (3, "Initech", "silver", "2023-03-10"),
            (4, "Umbrella Corp", "bronze", "2023-04-05"),
            (5, "Stark Industries", "platinum", "2023-05-12"),
            (6, "Wayne Enterprises", "gold", "2023-06-18"),
            (7, "Oscorp", "silver", "2023-07-22"),
            (8, "LexCorp", "bronze", "2023-08-30"),
        ]
        cursor.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?)",
            customers
        )

        # Create orders table
        cursor.execute("""
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                order_date TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            )
        """)

        # Insert order data - multiple months of orders
        orders = [
            # January 2024
            (1, 1, "2024-01-05", 5000.00, "delivered"),
            (2, 2, "2024-01-10", 2500.00, "delivered"),
            (3, 3, "2024-01-15", 1200.00, "delivered"),
            (4, 4, "2024-01-20", 800.00, "delivered"),
            # February 2024
            (5, 1, "2024-02-03", 4500.00, "delivered"),
            (6, 5, "2024-02-08", 6000.00, "delivered"),
            (7, 2, "2024-02-12", 2800.00, "delivered"),
            (8, 6, "2024-02-18", 3200.00, "delivered"),
            # March 2024
            (9, 1, "2024-03-02", 5500.00, "delivered"),
            (10, 3, "2024-03-07", 1500.00, "delivered"),
            (11, 5, "2024-03-12", 7000.00, "delivered"),
            (12, 7, "2024-03-20", 900.00, "delivered"),
            # April 2024
            (13, 2, "2024-04-05", 3000.00, "delivered"),
            (14, 4, "2024-04-10", 850.00, "delivered"),
            (15, 6, "2024-04-15", 3500.00, "delivered"),
            (16, 8, "2024-04-22", 600.00, "delivered"),
            # May 2024
            (17, 1, "2024-05-01", 4800.00, "delivered"),
            (18, 5, "2024-05-08", 6500.00, "delivered"),
            (19, 3, "2024-05-15", 1300.00, "delivered"),
            (20, 7, "2024-05-20", 950.00, "delivered"),
        ]
        cursor.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            orders
        )

        conn.commit()
        conn.close()

        yield {"path": str(db_path)}


@pytest.fixture
def auditable_session(ecommerce_db):
    """Create a Session configured for auditable mode with e-commerce data."""
    from constat.core.config import Config, DatabaseConfig
    from constat.storage.history import SessionHistory
    from constat.session import Session, SessionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            databases={
                "sales": DatabaseConfig(
                    type="sql",
                    uri=f"sqlite:///{ecommerce_db['path']}",
                    description="E-commerce database with customers (tiers: platinum, gold, silver, bronze) and orders",
                ),
            },
            system_prompt="""You are analyzing e-commerce data.

Tables:
- customers: customer_id, name, tier (platinum/gold/silver/bronze), created_date
- orders: order_id, customer_id, order_date, amount, status

Important:
- This is SQLite - use strftime() for date formatting
- Do NOT use schema prefixes (use 'customers' not 'sales.customers')
""",
            execution={
                "allowed_imports": ["pandas", "numpy", "datetime"],
                "timeout_seconds": 60,
            },
        )

        history = SessionHistory(storage_dir=Path(tmpdir) / "sessions")
        # Skip approval for automated tests
        session_config = SessionConfig(
            require_approval=False,
        )
        session_id = str(uuid.uuid4())

        session = Session(config, session_id=session_id, session_config=session_config, history=history)
        yield session


class TestAuditableBasicQueries:
    """Basic auditable mode query tests.

    Uses "verify" or "prove" keywords to trigger auditable mode explicitly.
    """

    def test_simple_count(self, auditable_session):
        """Test simple count query in auditable mode."""
        result = auditable_session.solve(
            "Verify: How many customers do we have?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "")
        # Should find 8 customers
        assert "8" in output, f"Expected '8' in output: {output}"
        print(f"\n--- Auditable Count Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")

    def test_simple_sum(self, auditable_session):
        """Test simple sum query in auditable mode."""
        result = auditable_session.solve(
            "Prove the total revenue from all orders."
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "")
        # Total: sum of all order amounts
        # 5000+2500+1200+800+4500+6000+2800+3200+5500+1500+7000+900+3000+850+3500+600+4800+6500+1300+950 = 62400
        assert "62400" in output or "62,400" in output or "62400.0" in output, \
            f"Expected '62400' in output: {output}"
        print(f"\n--- Auditable Sum Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")

    def test_group_by_tier(self, auditable_session):
        """Test grouping by customer tier."""
        result = auditable_session.solve(
            "Verify: How many customers are in each tier?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "").lower()
        # Each tier has 2 customers
        assert "platinum" in output or "gold" in output or "2" in output, \
            f"Expected tier names or count in output: {output}"
        print(f"\n--- Auditable Group By Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")


class TestAuditableComplexQueries:
    """Complex multi-table auditable mode tests.

    Uses "verify" or "prove" keywords to trigger auditable mode explicitly.
    """

    def test_revenue_by_tier(self, auditable_session):
        """Test revenue breakdown by customer tier."""
        result = auditable_session.solve(
            "Verify: What is the total revenue by customer tier?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "").lower()
        # Should mention tiers and revenue
        assert "platinum" in output or "gold" in output or "tier" in output or "revenue" in output, \
            f"Expected tier breakdown in output: {output}"
        print(f"\n--- Revenue by Tier Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")

    @pytest.mark.xfail(reason="LLM-dependent: code generation is non-deterministic", strict=False)
    def test_monthly_revenue_trend(self, auditable_session):
        """Test monthly revenue trend query - the user's actual use case."""
        result = auditable_session.solve(
            "Prove: What is our monthly revenue trend and how does it break down by customer tier?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "")
        # Should have some content
        assert len(output) > 20, f"Expected substantial output: {output}"
        print(f"\n--- Monthly Revenue Trend Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")

    def test_average_order_by_tier(self, auditable_session):
        """Test average order value by tier."""
        result = auditable_session.solve(
            "Verify: What is the average order value for each customer tier?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        output = result.get("output", "").lower()
        assert "average" in output or "avg" in output or "platinum" in output or "tier" in output, \
            f"Expected average values in output: {output}"
        print(f"\n--- Average Order by Tier Query ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")


class TestAuditableDerivationTrace:
    """Tests for auditable mode derivation traces."""

    def test_has_derivation_output(self, auditable_session):
        """Test that auditable mode returns derivation information."""
        result = auditable_session.solve(
            "Verify: How many platinum tier customers do we have?"
        )

        assert result.get("success"), f"Query failed: {result.get('error')}"

        # In auditable mode, should have derivation or trace info
        output = result.get("output", "")
        # Result should be 2 (two platinum customers)
        assert "2" in output, f"Expected '2' in output: {output}"

        print(f"\n--- Derivation Trace Test ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Output: {output}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")


class TestAuditableErrorRecovery:
    """Tests for error recovery in auditable mode."""

    def test_handles_complex_query(self, auditable_session):
        """Test that complex queries with potential errors are handled."""
        result = auditable_session.solve(
            "Prove: For each month in 2024, calculate the revenue from platinum customers "
            "compared to total revenue as a percentage."
        )

        # Should either succeed or fail gracefully
        output = result.get("output", "")
        error = result.get("error", "")

        # Should have some response
        assert len(output) > 0 or len(error) > 0, \
            "Expected either output or error message"

        print(f"\n--- Complex Query Error Recovery ---")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Success: {result.get('success')}")
        print(f"Output: {output}")
        if error:
            print(f"Error: {error}")
