# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Integration tests for database connections using Docker containers.

These tests require Docker to be running. They will be automatically skipped
if Docker is not available.

Usage:
    pytest tests/test_database_integration.py -v

Markers:
    @pytest.mark.requires_mongodb - Tests requiring MongoDB
    @pytest.mark.requires_postgresql - Tests requiring PostgreSQL
"""

import pytest
import pandas as pd


class TestMongoDBIntegration:
    """Integration tests for MongoDB using Docker container."""

    @pytest.mark.requires_mongodb
    def test_mongodb_connection(self, mongodb_container):
        """Test basic MongoDB connection."""
        try:
            import pymongo
        except ImportError:
            pytest.skip("pymongo not installed")

        client = pymongo.MongoClient(mongodb_container["uri"])

        # Test connection
        server_info = client.server_info()
        assert "version" in server_info

        client.close()

    @pytest.mark.requires_mongodb
    def test_mongodb_basic_operations(self, mongodb_container):
        """Test basic MongoDB CRUD operations."""
        try:
            import pymongo
        except ImportError:
            pytest.skip("pymongo not installed")

        client = pymongo.MongoClient(mongodb_container["uri"])
        db = client["constat_test"]
        collection = db["test_collection"]

        # Clean up any existing data
        collection.delete_many({})

        # Insert
        doc = {"name": "test", "value": 42}
        result = collection.insert_one(doc)
        assert result.inserted_id is not None

        # Read
        found = collection.find_one({"name": "test"})
        assert found is not None
        assert found["value"] == 42

        # Update
        collection.update_one({"name": "test"}, {"$set": {"value": 100}})
        found = collection.find_one({"name": "test"})
        assert found["value"] == 100

        # Delete
        collection.delete_one({"name": "test"})
        found = collection.find_one({"name": "test"})
        assert found is None

        # Cleanup
        db.drop_collection("test_collection")
        client.close()

    @pytest.mark.requires_mongodb
    def test_mongodb_connector_integration(self, mongodb_container):
        """Test MongoDBConnector with real MongoDB instance."""
        try:
            import pymongo
        except ImportError:
            pytest.skip("pymongo not installed")

        from constat.catalog.nosql import MongoDBConnector

        # First, insert some test data directly
        client = pymongo.MongoClient(mongodb_container["uri"])
        db = client["integration_test"]
        collection = db["users"]
        collection.delete_many({})
        collection.insert_many([
            {"name": "Alice", "age": 30, "active": True},
            {"name": "Bob", "age": 25, "active": False},
        ])

        # Now test the connector
        connector = MongoDBConnector(
            uri=mongodb_container["uri"],
            database="integration_test",
        )
        connector.connect()

        try:
            # Test get_collections
            collections = connector.get_collections()
            assert "users" in collections

            # Test get_collection_schema
            metadata = connector.get_collection_schema("users")
            assert metadata.name == "users"
            assert metadata.document_count == 2

            # Check inferred fields
            field_names = [f.name for f in metadata.fields]
            assert "name" in field_names
            assert "age" in field_names
            assert "active" in field_names
        finally:
            connector.disconnect()
            db.drop_collection("users")
            client.close()


class TestPostgreSQLIntegration:
    """Integration tests for PostgreSQL using Docker container."""

    @pytest.mark.requires_postgresql
    def test_postgresql_connection(self, postgresql_container):
        """Test basic PostgreSQL connection."""
        try:
            import psycopg2
        except ImportError:
            pytest.skip("psycopg2 not installed")

        conn = psycopg2.connect(
            host=postgresql_container["host"],
            port=postgresql_container["port"],
            user=postgresql_container["user"],
            password=postgresql_container["password"],
            database=postgresql_container["database"],
        )

        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()[0]
        assert "PostgreSQL" in version

        cur.close()
        conn.close()

    @pytest.mark.requires_postgresql
    def test_postgresql_basic_operations(self, postgresql_container):
        """Test basic PostgreSQL CRUD operations."""
        try:
            import psycopg2
        except ImportError:
            pytest.skip("psycopg2 not installed")

        conn = psycopg2.connect(
            host=postgresql_container["host"],
            port=postgresql_container["port"],
            user=postgresql_container["user"],
            password=postgresql_container["password"],
            database=postgresql_container["database"],
        )

        cur = conn.cursor()

        try:
            # Create table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    value INTEGER
                )
            """)
            conn.commit()

            # Insert
            cur.execute(
                "INSERT INTO test_table (name, value) VALUES (%s, %s) RETURNING id",
                ("test", 42),
            )
            inserted_id = cur.fetchone()[0]
            conn.commit()
            assert inserted_id is not None

            # Read
            cur.execute("SELECT name, value FROM test_table WHERE id = %s", (inserted_id,))
            row = cur.fetchone()
            assert row == ("test", 42)

            # Update
            cur.execute("UPDATE test_table SET value = %s WHERE id = %s", (100, inserted_id))
            conn.commit()
            cur.execute("SELECT value FROM test_table WHERE id = %s", (inserted_id,))
            assert cur.fetchone()[0] == 100

            # Delete
            cur.execute("DELETE FROM test_table WHERE id = %s", (inserted_id,))
            conn.commit()
            cur.execute("SELECT COUNT(*) FROM test_table WHERE id = %s", (inserted_id,))
            assert cur.fetchone()[0] == 0

        finally:
            # Cleanup
            cur.execute("DROP TABLE IF EXISTS test_table")
            conn.commit()
            cur.close()
            conn.close()

    @pytest.mark.requires_postgresql
    def test_postgresql_with_sqlalchemy(self, postgresql_container):
        """Test PostgreSQL with SQLAlchemy (if available)."""
        try:
            from sqlalchemy import create_engine, text
        except ImportError:
            pytest.skip("sqlalchemy not installed")

        try:
            import psycopg2
        except ImportError:
            pytest.skip("psycopg2 not installed")

        engine = create_engine(postgresql_container["dsn"])

        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.fetchone()[0] == 1


class TestSampleDataFixtures:
    """Tests using pre-loaded sample data fixtures."""

    @pytest.mark.requires_mongodb
    def test_mongodb_sample_data_loaded(self, mongodb_with_sample_data):
        """Test that sample data is automatically loaded."""
        try:
            import pymongo
        except ImportError:
            pytest.skip("pymongo not installed")

        client = pymongo.MongoClient(mongodb_with_sample_data["uri"])
        db = client[mongodb_with_sample_data["database"]]

        # Verify collections exist
        collections = db.list_collection_names()
        assert "customers" in collections
        assert "products" in collections
        assert "orders" in collections

        # Verify data
        assert db.customers.count_documents({}) == 3
        assert db.products.count_documents({}) == 3
        assert db.orders.count_documents({}) == 2

        # Query sample data
        alice = db.customers.find_one({"name": "Alice Johnson"})
        assert alice is not None
        assert alice["email"] == "alice@example.com"

        laptop = db.products.find_one({"name": "Laptop Pro"})
        assert laptop is not None
        assert laptop["price"] == 1299.99

        client.close()

    @pytest.mark.requires_postgresql
    def test_postgresql_sample_data_loaded(self, postgresql_with_sample_data):
        """Test that sample data is automatically loaded."""
        try:
            import psycopg2
        except ImportError:
            pytest.skip("psycopg2 not installed")

        conn = psycopg2.connect(postgresql_with_sample_data["dsn"])
        cur = conn.cursor()

        # Verify tables exist and have data
        cur.execute("SELECT COUNT(*) FROM customers")
        assert cur.fetchone()[0] == 3

        cur.execute("SELECT COUNT(*) FROM products")
        assert cur.fetchone()[0] == 3

        cur.execute("SELECT COUNT(*) FROM orders")
        assert cur.fetchone()[0] == 2

        # Query sample data
        cur.execute("SELECT email FROM customers WHERE name = 'Alice Johnson'")
        assert cur.fetchone()[0] == "alice@example.com"

        cur.execute("SELECT price FROM products WHERE name = 'Laptop Pro'")
        assert float(cur.fetchone()[0]) == 1299.99

        cur.close()
        conn.close()


class TestPostgreSQLDataStore:
    """Integration tests for DataStore using PostgreSQL as the artifact database."""

    @pytest.mark.requires_postgresql
    def test_datastore_with_postgresql_uri(self, postgresql_container):
        """Test DataStore with PostgreSQL connection URI."""
        from constat.storage.datastore import DataStore

        # Create datastore with PostgreSQL URI
        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Test basic DataFrame operations
            df = pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "value": [100, 200, 300]
            })

            store.save_dataframe("test_table", df, step_number=1, description="Test data")

            # Verify data was saved
            loaded = store.load_dataframe("test_table")
            assert loaded is not None
            assert len(loaded) == 3
            assert loaded["value"].sum() == 600

            # Verify table registry
            tables = store.list_tables()
            assert len(tables) == 1
            assert tables[0]["name"] == "test_table"
            assert tables[0]["row_count"] == 3
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_state_variables_postgresql(self, postgresql_container):
        """Test state variable storage in PostgreSQL."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Store various types of state
            store.set_state("counter", 42, step_number=1)
            store.set_state("config", {"enabled": True, "limit": 100}, step_number=1)
            store.set_state("tags", ["a", "b", "c"], step_number=2)

            # Retrieve and verify
            assert store.get_state("counter") == 42
            assert store.get_state("config") == {"enabled": True, "limit": 100}
            assert store.get_state("tags") == ["a", "b", "c"]

            # Get all state
            all_state = store.get_all_state()
            assert len(all_state) == 3
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_artifacts_postgresql(self, postgresql_container):
        """Test artifact storage in PostgreSQL."""
        from constat.storage.datastore import DataStore
        from constat.core.models import ArtifactType

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Save HTML artifact
            html_artifact = store.save_html(
                name="test_report",
                html_content="<html><body><h1>Report</h1></body></html>",
                step_number=1,
                title="Test Report",
                description="Integration test report"
            )

            assert html_artifact.id > 0
            assert html_artifact.artifact_type == ArtifactType.HTML

            # Save chart artifact
            chart_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "data": {"values": [{"x": 1, "y": 2}, {"x": 2, "y": 4}]},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "ordinal"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            }

            chart_artifact = store.save_chart(
                name="test_chart",
                spec=chart_spec,
                step_number=2,
                title="Test Chart"
            )

            assert chart_artifact.artifact_type == ArtifactType.CHART

            # Retrieve by name
            retrieved = store.get_artifact_by_name("test_report")
            assert retrieved is not None
            assert retrieved.title == "Test Report"

            # List artifacts
            artifacts = store.list_artifacts()
            assert len(artifacts) == 2
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_scratchpad_postgresql(self, postgresql_container):
        """Test scratchpad storage in PostgreSQL."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Add scratchpad entries
            store.add_scratchpad_entry(
                step_number=1,
                goal="Load sales data",
                narrative="Loaded 1000 rows from sales.csv",
                tables_created=["raw_sales"]
            )

            store.add_scratchpad_entry(
                step_number=2,
                goal="Aggregate by region",
                narrative="Aggregated sales by region, found 5 regions",
                tables_created=["sales_by_region"]
            )

            # Retrieve entries
            entries = store.get_scratchpad()
            assert len(entries) == 2
            assert entries[0]["goal"] == "Load sales data"
            assert entries[1]["tables_created"] == ["sales_by_region"]

            # Get as markdown
            markdown = store.get_scratchpad_as_markdown()
            assert "Load sales data" in markdown
            assert "Aggregate by region" in markdown
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_plan_steps_postgresql(self, postgresql_container):
        """Test plan step storage in PostgreSQL."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Save plan steps
            store.save_plan_step(
                step_number=1,
                goal="Load and validate data",
                expected_inputs=["sales.csv"],
                expected_outputs=["raw_sales"],
                status="pending"
            )

            store.save_plan_step(
                step_number=2,
                goal="Transform data",
                expected_inputs=["raw_sales"],
                expected_outputs=["clean_sales"],
                status="pending"
            )

            # Update step status
            store.update_plan_step(
                step_number=1,
                status="completed",
                code="df = pd.read_csv('sales.csv')",
                duration_ms=1500
            )

            # Retrieve steps
            steps = store.get_plan_steps()
            assert len(steps) == 2
            assert steps[0]["status"] == "completed"
            assert steps[0]["duration_ms"] == 1500
            assert steps[1]["status"] == "pending"
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_session_meta_postgresql(self, postgresql_container):
        """Test session metadata storage in PostgreSQL."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Set session metadata
            store.set_session_meta("problem", "Analyze Q4 sales trends")
            store.set_session_meta("status", "in_progress")
            store.set_session_meta("created_by", "test_user")

            # Retrieve
            assert store.get_session_meta("problem") == "Analyze Q4 sales trends"
            assert store.get_session_meta("status") == "in_progress"

            # Get all
            all_meta = store.get_all_session_meta()
            assert len(all_meta) == 3
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_sql_query_postgresql(self, postgresql_container):
        """Test SQL queries on PostgreSQL backend."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Create test data
            df = pd.DataFrame({
                "region": ["North", "North", "South", "South", "East"],
                "product": ["A", "B", "A", "B", "A"],
                "sales": [100, 150, 200, 50, 300]
            })

            store.save_dataframe("sales_data", df, step_number=1)

            # Run aggregate query
            result = store.query("""
                SELECT region, SUM(sales) as total_sales
                FROM sales_data
                GROUP BY region
                ORDER BY total_sales DESC
            """)

            assert len(result) == 3
            assert result.iloc[0]["region"] == "East"
            assert result.iloc[0]["total_sales"] == 300
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_clear_step_data_postgresql(self, postgresql_container):
        """Test clearing step data in PostgreSQL."""
        from constat.storage.datastore import DataStore

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Create data for multiple steps
            df1 = pd.DataFrame({"x": [1, 2, 3]})
            df2 = pd.DataFrame({"y": [4, 5, 6]})

            store.save_dataframe("step1_table", df1, step_number=1)
            store.set_state("step1_var", "value1", step_number=1)

            store.save_dataframe("step2_table", df2, step_number=2)
            store.set_state("step2_var", "value2", step_number=2)

            # Verify both exist
            assert store.load_dataframe("step1_table") is not None
            assert store.load_dataframe("step2_table") is not None

            # Clear step 1
            store.clear_step_data(1)

            # Step 1 data should be gone
            assert store.load_dataframe("step1_table") is None
            assert store.get_state("step1_var") is None

            # Step 2 data should remain
            assert store.load_dataframe("step2_table") is not None
            assert store.get_state("step2_var") == "value2"
        finally:
            store.close()

    @pytest.mark.requires_postgresql
    def test_datastore_full_session_state_postgresql(self, postgresql_container):
        """Test full session state export from PostgreSQL."""
        from constat.storage.datastore import DataStore
        from sqlalchemy import text

        store = DataStore(uri=postgresql_container["dsn"])

        try:
            # Clean up any leftover data from previous tests
            with store.engine.begin() as conn:
                conn.execute(text("DELETE FROM _constat_table_registry"))
                conn.execute(text("DELETE FROM _constat_state"))
                conn.execute(text("DELETE FROM _constat_scratchpad"))
                conn.execute(text("DELETE FROM _constat_artifacts"))
                conn.execute(text("DELETE FROM _constat_session"))
                conn.execute(text("DELETE FROM _constat_plan_steps"))

            # Build up a complete session
            store.set_session_meta("problem", "Test problem")

            store.save_plan_step(1, "Step 1", status="completed")
            store.save_plan_step(2, "Step 2", status="pending")

            df = pd.DataFrame({"data": [1, 2, 3]})
            store.save_dataframe("test_data", df, step_number=1)

            store.set_state("result", 42, step_number=1)

            store.add_scratchpad_entry(1, "Step 1", "Did step 1")

            store.save_html("report", "<p>Report</p>", step_number=1)

            # Get full state
            state = store.get_full_session_state()

            assert "session" in state
            assert state["session"]["problem"] == "Test problem"

            assert "plan_steps" in state
            assert len(state["plan_steps"]) == 2

            assert "tables" in state
            assert len(state["tables"]) == 1

            assert "state" in state
            assert state["state"]["result"] == 42

            assert "scratchpad" in state
            assert len(state["scratchpad"]) == 1

            assert "artifacts" in state
            assert len(state["artifacts"]) == 1
        finally:
            store.close()


class TestDockerAvailability:
    """Tests for Docker fixture behavior."""

    def test_docker_available_fixture(self, docker_available):
        """Test that docker_available fixture returns a boolean."""
        assert isinstance(docker_available, bool)

    @pytest.mark.requires_docker
    def test_requires_docker_marker(self):
        """Test that requires_docker marker works (skips if Docker unavailable)."""
        # This test will only run if Docker is available
        import subprocess

        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
