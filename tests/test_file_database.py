"""Tests for file-based data sources as database types.

Tests that CSV, JSON, and other file formats configured in the databases section
work correctly with schema introspection and code execution.
"""

import csv
import json
import os
import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="module")
def file_data_sources():
    """Create temporary file-based data sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create CSV file
        csv_path = data_dir / "sales.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "product", "quantity", "price"])
            writer.writerows([
                ["2024-01-15", "Widget A", 10, 25.00],
                ["2024-01-16", "Widget B", 5, 45.00],
                ["2024-01-17", "Widget A", 8, 25.00],
                ["2024-01-18", "Widget C", 3, 100.00],
            ])

        # Create JSON file
        json_path = data_dir / "events.json"
        events = [
            {"event_id": "E001", "type": "purchase", "amount": 150.00},
            {"event_id": "E002", "type": "signup", "amount": None},
            {"event_id": "E003", "type": "purchase", "amount": 75.50},
        ]
        with open(json_path, "w") as f:
            json.dump(events, f)

        yield {
            "csv_path": str(csv_path),
            "json_path": str(json_path),
        }


class TestFileConnector:
    """Tests for the FileConnector class."""

    def test_csv_schema_inference(self, file_data_sources):
        """Test that CSV files have schema inferred correctly."""
        from constat.catalog.file.connector import FileConnector, FileType

        connector = FileConnector(
            name="sales",
            path=file_data_sources["csv_path"],
            file_type=FileType.CSV,
            description="Test sales data",
        )

        metadata = connector.get_metadata()

        assert metadata.name == "sales"
        assert metadata.file_type == FileType.CSV
        assert metadata.row_count == 4
        assert len(metadata.columns) == 4

        col_names = [c.name for c in metadata.columns]
        assert "date" in col_names
        assert "product" in col_names
        assert "quantity" in col_names
        assert "price" in col_names

    def test_json_schema_inference(self, file_data_sources):
        """Test that JSON files have schema inferred correctly."""
        from constat.catalog.file.connector import FileConnector, FileType

        connector = FileConnector(
            name="events",
            path=file_data_sources["json_path"],
            file_type=FileType.JSON,
            description="Test events data",
        )

        metadata = connector.get_metadata()

        assert metadata.name == "events"
        assert metadata.file_type == FileType.JSON
        assert metadata.row_count == 3
        assert len(metadata.columns) == 3

        col_names = [c.name for c in metadata.columns]
        assert "event_id" in col_names
        assert "type" in col_names
        assert "amount" in col_names

    def test_metadata_embedding_text(self, file_data_sources):
        """Test that metadata generates appropriate embedding text."""
        from constat.catalog.file.connector import FileConnector, FileType

        connector = FileConnector(
            name="sales",
            path=file_data_sources["csv_path"],
            file_type=FileType.CSV,
            description="Sales transaction data",
        )

        metadata = connector.get_metadata()
        text = metadata.to_embedding_text()

        assert "sales" in text.lower()
        assert "csv" in text.lower()
        assert "date" in text.lower()
        assert "product" in text.lower()

    def test_overview_generation(self, file_data_sources):
        """Test that overview is generated for system prompt."""
        from constat.catalog.file.connector import FileConnector, FileType

        connector = FileConnector(
            name="sales",
            path=file_data_sources["csv_path"],
            file_type=FileType.CSV,
            description="Sales data",
        )

        overview = connector.get_overview()

        assert "sales" in overview
        assert "csv" in overview
        assert "date" in overview


class TestSchemaManagerWithFiles:
    """Tests for SchemaManager handling file data sources."""

    def test_file_source_introspection(self, file_data_sources):
        """Test that SchemaManager introspects file sources correctly."""
        from constat.core.config import Config, DatabaseConfig
        from constat.catalog.schema_manager import SchemaManager

        config = Config(
            databases={
                "sales_csv": DatabaseConfig(
                    type="csv",
                    path=file_data_sources["csv_path"],
                    description="Sales CSV data",
                ),
                "events_json": DatabaseConfig(
                    type="json",
                    path=file_data_sources["json_path"],
                    description="Events JSON data",
                ),
            }
        )

        manager = SchemaManager(config)
        manager.initialize()

        # Check tables are discovered
        table_names = manager.list_tables()

        assert "sales_csv.sales_csv" in table_names
        assert "events_json.events_json" in table_names

        # Check types are correct via get_table_schema
        sales_schema = manager.get_table_schema("sales_csv")
        assert sales_schema["database"] == "sales_csv"
        assert len(sales_schema["columns"]) == 4  # date, product, quantity, price

        events_schema = manager.get_table_schema("events_json")
        assert events_schema["database"] == "events_json"
        assert len(events_schema["columns"]) == 3  # event_id, type, amount

    def test_mixed_sql_and_file_sources(self, file_data_sources):
        """Test SchemaManager with both SQL and file sources."""
        import sqlite3
        from constat.core.config import Config, DatabaseConfig
        from constat.catalog.schema_manager import SchemaManager

        # Create a temp SQLite database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob')")
            conn.commit()
            conn.close()

            config = Config(
                databases={
                    "main_db": DatabaseConfig(
                        uri=f"sqlite:///{db_path}",
                        description="Main SQL database",
                    ),
                    "sales_csv": DatabaseConfig(
                        type="csv",
                        path=file_data_sources["csv_path"],
                        description="Sales CSV data",
                    ),
                }
            )

            manager = SchemaManager(config)
            manager.initialize()

            table_names = manager.list_tables()

            # Should have both SQL and file tables
            assert "main_db.users" in table_names  # SQL table
            assert "sales_csv.sales_csv" in table_names  # CSV file

            # Verify schemas
            users_schema = manager.get_table_schema("main_db.users")
            assert users_schema["database"] == "main_db"

            sales_schema = manager.get_table_schema("sales_csv")
            assert sales_schema["database"] == "sales_csv"

    def test_semantic_search_across_sources(self, file_data_sources):
        """Test semantic search finds file sources."""
        from constat.core.config import Config, DatabaseConfig
        from constat.catalog.schema_manager import SchemaManager

        config = Config(
            databases={
                "sales_csv": DatabaseConfig(
                    type="csv",
                    path=file_data_sources["csv_path"],
                    description="Sales transaction data with products and quantities",
                ),
                "events_json": DatabaseConfig(
                    type="json",
                    path=file_data_sources["json_path"],
                    description="User event tracking data",
                ),
            }
        )

        manager = SchemaManager(config)
        manager.initialize()

        # Search for sales-related data
        results = manager.find_relevant_tables("product sales transactions")
        assert len(results) > 0
        top_result = results[0]
        assert "sales" in top_result["table"].lower()

        # Search for events
        results = manager.find_relevant_tables("user events and signups")
        assert len(results) > 0
        top_result = results[0]
        assert "events" in top_result["table"].lower()


class TestEngineWithFiles:
    """Tests for QueryEngine execution with file sources."""

    def test_execution_globals_has_file_paths(self, file_data_sources):
        """Test that engine provides file_<name> variables."""
        from constat.core.config import Config, DatabaseConfig
        from constat.catalog.schema_manager import SchemaManager
        from constat.execution.engine import QueryEngine

        config = Config(
            databases={
                "sales_csv": DatabaseConfig(
                    type="csv",
                    path=file_data_sources["csv_path"],
                    description="Sales CSV data",
                ),
            }
        )

        manager = SchemaManager(config)
        manager.initialize()

        engine = QueryEngine(config, manager)
        globals_dict = engine._get_execution_globals()

        # Should have file_sales_csv pointing to the path
        assert "file_sales_csv" in globals_dict
        assert globals_dict["file_sales_csv"] == file_data_sources["csv_path"]

    def test_code_execution_with_file(self, file_data_sources):
        """Test that generated code can load file data."""
        from constat.core.config import Config, DatabaseConfig
        from constat.catalog.schema_manager import SchemaManager
        from constat.execution.executor import PythonExecutor

        config = Config(
            databases={
                "sales_csv": DatabaseConfig(
                    type="csv",
                    path=file_data_sources["csv_path"],
                    description="Sales CSV data",
                ),
            }
        )

        manager = SchemaManager(config)
        manager.initialize()

        # Simulate what the engine would do
        globals_dict = {"file_sales_csv": file_data_sources["csv_path"]}

        # Test code that loads the CSV
        code = """
import pandas as pd
df = pd.read_csv(file_sales_csv)
print(f"Rows: {len(df)}")
print(f"Total quantity: {df['quantity'].sum()}")
"""

        executor = PythonExecutor(timeout_seconds=30)
        result = executor.execute(code, globals_dict)

        assert result.success, f"Execution failed: {result.error_message()}"
        assert "Rows: 4" in result.stdout
        assert "Total quantity: 26" in result.stdout  # 10+5+8+3

    def test_json_file_execution(self, file_data_sources):
        """Test that JSON files can be loaded and analyzed."""
        from constat.execution.executor import PythonExecutor

        globals_dict = {"file_events": file_data_sources["json_path"]}

        code = """
import pandas as pd
df = pd.read_json(file_events)
print(f"Events: {len(df)}")
print(f"Purchase count: {len(df[df['type'] == 'purchase'])}")
"""

        executor = PythonExecutor(timeout_seconds=30)
        result = executor.execute(code, globals_dict)

        assert result.success, f"Execution failed: {result.error_message()}"
        assert "Events: 3" in result.stdout
        assert "Purchase count: 2" in result.stdout
