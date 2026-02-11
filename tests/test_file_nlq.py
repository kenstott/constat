# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""End-to-end tests for natural language queries against file-based data sources.

These tests verify that the system can:
1. Discover file-based data sources (CSV, JSON) via document discovery
2. Generate Python code to load and analyze file data
3. Return correct results

Requires:
- ANTHROPIC_API_KEY (for LLM queries)
"""

import csv
import json
import os
import pytest
import tempfile
import uuid
from pathlib import Path

# Skip all tests if API key not set
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    ),
]


@pytest.fixture(scope="module")
def file_data_dir():
    """Create a temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        data_dir.mkdir()

        # Create CSV file: sales data
        sales_csv = data_dir / "sales.csv"
        with open(sales_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "product", "quantity", "price", "region"])
            writer.writerows([
                ["2024-01-15", "Widget A", 10, 25.00, "North"],
                ["2024-01-16", "Widget B", 5, 45.00, "South"],
                ["2024-01-17", "Widget A", 8, 25.00, "East"],
                ["2024-01-18", "Widget C", 3, 100.00, "West"],
                ["2024-01-19", "Widget B", 12, 45.00, "North"],
                ["2024-01-20", "Widget A", 15, 25.00, "South"],
                ["2024-01-21", "Widget C", 2, 100.00, "East"],
                ["2024-01-22", "Widget B", 7, 45.00, "West"],
            ])

        # Create JSON file: events data
        events_json = data_dir / "events.json"
        events = [
            {"event_id": "E001", "type": "purchase", "user": "user_1", "amount": 150.00},
            {"event_id": "E002", "type": "signup", "user": "user_2", "amount": None},
            {"event_id": "E003", "type": "purchase", "user": "user_1", "amount": 75.50},
            {"event_id": "E004", "type": "refund", "user": "user_3", "amount": -50.00},
            {"event_id": "E005", "type": "purchase", "user": "user_2", "amount": 200.00},
            {"event_id": "E006", "type": "signup", "user": "user_4", "amount": None},
            {"event_id": "E007", "type": "purchase", "user": "user_1", "amount": 300.00},
        ]
        with open(events_json, "w") as f:
            json.dump(events, f, indent=2)

        yield {
            "dir": data_dir,
            "sales_csv": sales_csv,
            "events_json": events_json,
        }


@pytest.fixture
def file_session(file_data_dir):
    """Create a Session configured with file-based data sources."""
    from constat.core.config import Config, DatabaseConfig
    from constat.storage.history import SessionHistory
    from constat.session import Session, SessionConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            # File-based data sources in databases section
            databases={
                "sales_data": DatabaseConfig(
                    type="csv",
                    path=str(file_data_dir["sales_csv"]),
                    description="Sales transactions CSV with columns: date, product, quantity, price, region",
                ),
                "events_data": DatabaseConfig(
                    type="json",
                    path=str(file_data_dir["events_json"]),
                    description="User events JSON with fields: event_id, type (purchase/signup/refund), user, amount",
                ),
            },
            system_prompt="""You are analyzing business data from files.

Use schema discovery tools to find available data sources.
Use pandas to load and analyze the files (file_<name> variables contain paths).
""",
            execution={
                "allowed_imports": ["pandas", "numpy", "json", "datetime"],
                "timeout_seconds": 60,
            },
        )

        history = SessionHistory(storage_dir=Path(tmpdir) / "sessions")
        session_config = SessionConfig(max_retries_per_step=3)
        session_id = str(uuid.uuid4())

        session = Session(config, session_id=session_id, session_config=session_config, history=history)
        yield session


class TestFileDiscovery:
    """Tests for discovering file-based data sources via schema tools."""

    def test_schema_manager_discovers_files(self, file_session):
        """Verify that SchemaManager discovers file-based data sources."""
        # File sources appear in schema manager like SQL/NoSQL databases
        tables = file_session.schema_manager.list_tables()

        # Should find both file sources
        assert any("sales_data" in t for t in tables)
        assert any("events_data" in t for t in tables)

        # Can get schema for file sources
        sales_schema = file_session.schema_manager.get_table_schema("sales_data")
        assert sales_schema["database"] == "sales_data"
        assert len(sales_schema["columns"]) > 0  # Has inferred columns


@pytest.mark.xfail(reason="LLM-dependent: code generation is non-deterministic", strict=False)
class TestCSVNaturalLanguageQueries:
    """End-to-end tests for NLQ against CSV files."""

    def test_simple_count_csv(self, file_session):
        """Test counting rows in a CSV file."""
        result = file_session.solve(
            "How many sales transactions are in the sales data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Should find 8 rows
        assert "8" in result["output"], f"Expected '8' in output: {result['output']}"
        print(f"\n--- CSV Count Query ---")
        print(f"Output: {result['output']}")

    def test_sum_csv(self, file_session):
        """Test summing values in a CSV file."""
        result = file_session.solve(
            "What is the total quantity sold across all transactions in the sales data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Total: 10+5+8+3+12+15+2+7 = 62
        assert "62" in result["output"], f"Expected '62' in output: {result['output']}"
        print(f"\n--- CSV Sum Query ---")
        print(f"Output: {result['output']}")

    def test_group_by_csv(self, file_session):
        """Test grouping and aggregation on CSV data."""
        result = file_session.solve(
            "What is the total quantity sold for each product in the sales data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # Widget A: 10+8+15=33, Widget B: 5+12+7=24, Widget C: 3+2=5
        output = result["output"].lower()
        assert "widget" in output
        print(f"\n--- CSV Group By Query ---")
        print(f"Output: {result['output']}")

    def test_filter_csv(self, file_session):
        """Test filtering CSV data."""
        result = file_session.solve(
            "How many sales transactions were in the North region?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # 2 transactions in North
        assert "2" in result["output"], f"Expected '2' in output: {result['output']}"
        print(f"\n--- CSV Filter Query ---")
        print(f"Output: {result['output']}")


@pytest.mark.xfail(reason="LLM-dependent: code generation is non-deterministic", strict=False)
class TestJSONNaturalLanguageQueries:
    """End-to-end tests for NLQ against JSON files."""

    def test_count_json(self, file_session):
        """Test counting records in JSON file."""
        result = file_session.solve(
            "How many events are in the events data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # 7 events
        assert "7" in result["output"], f"Expected '7' in output: {result['output']}"
        print(f"\n--- JSON Count Query ---")
        print(f"Output: {result['output']}")

    def test_filter_json(self, file_session):
        """Test filtering JSON data."""
        result = file_session.solve(
            "How many purchase events are in the events data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # 4 purchase events
        assert "4" in result["output"], f"Expected '4' in output: {result['output']}"
        print(f"\n--- JSON Filter Query ---")
        print(f"Output: {result['output']}")

    def test_aggregate_json(self, file_session):
        """Test aggregation on JSON data."""
        result = file_session.solve(
            "What is the total amount from all purchase events in the events data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # 150 + 75.5 + 200 + 300 = 725.5
        output = result["output"]
        assert "725" in output or "725.5" in output or "725.50" in output
        print(f"\n--- JSON Aggregate Query ---")
        print(f"Output: {result['output']}")

    def test_unique_values_json(self, file_session):
        """Test finding unique values in JSON."""
        result = file_session.solve(
            "How many unique users are in the events data?"
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # 4 unique users
        assert "4" in result["output"], f"Expected '4' in output: {result['output']}"
        print(f"\n--- JSON Unique Query ---")
        print(f"Output: {result['output']}")


@pytest.mark.xfail(reason="LLM-dependent: code generation is non-deterministic", strict=False)
class TestMixedDataSources:
    """Tests combining file data with other operations."""

    def test_multi_step_file_analysis(self, file_session):
        """Test multi-step analysis on file data."""
        result = file_session.solve(
            "First calculate the average sale amount (quantity * price) per transaction "
            "from the sales data. Then find all transactions above that average."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        # This should create a multi-step plan
        if result.get("plan"):
            assert len(result["plan"].steps) >= 2
        print(f"\n--- Multi-Step File Analysis ---")
        print(f"Output: {result['output']}")

    def test_discover_and_analyze(self, file_session):
        """Test discovering data sources and then analyzing them."""
        result = file_session.solve(
            "What data files are available and what's in them? "
            "Then tell me the most common event type in the events data."
        )

        assert result["success"], f"Query failed: {result.get('error')}"
        output = result["output"].lower()
        # Should mention purchase (most common type with 4 occurrences)
        assert "purchase" in output
        print(f"\n--- Discover and Analyze ---")
        print(f"Output: {result['output']}")
