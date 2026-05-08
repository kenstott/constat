# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for DataStore."""

import json
import pytest
import tempfile
from pathlib import Path

import pandas as pd

from constat.storage.datastore import DataStore
from constat.core.models import ArtifactType


@pytest.fixture
def temp_db():
    """Create a temporary database path (not the file itself)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.duckdb"


@pytest.fixture
def memory_store():
    """Create an in-memory datastore."""
    return DataStore()


@pytest.fixture
def file_store(temp_db):
    """Create a file-backed datastore."""
    return DataStore(db_path=temp_db)


class TestDataStore:
    """Tests for DataStore."""

    def test_save_and_load_dataframe(self, memory_store):
        """Test saving and loading a DataFrame."""
        df = pd.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
            "value": [100, 200, 300]
        })

        memory_store.save_dataframe("test_table", df, step_number=1)

        loaded = memory_store.load_dataframe("test_table")
        assert loaded is not None
        assert len(loaded) == 3
        assert list(loaded.columns) == ["name", "value"]
        assert loaded["value"].sum() == 600

    def test_save_empty_dataframe_no_columns(self, memory_store):
        """Test that saving a DataFrame with no columns doesn't crash."""
        df = pd.DataFrame()  # Empty DataFrame with no columns

        # This should not raise an error - it should silently skip
        memory_store.save_dataframe("empty_table", df, step_number=1)

        # Table should not be created
        assert memory_store.load_dataframe("empty_table") is None
        assert len([t for t in memory_store.list_tables() if t["name"] == "empty_table"]) == 0

    def test_load_nonexistent_table(self, memory_store):
        """Test loading a table that doesn't exist."""
        result = memory_store.load_dataframe("nonexistent")
        assert result is None

    def test_query(self, memory_store):
        """Test SQL queries on saved data."""
        df = pd.DataFrame({
            "category": ["A", "A", "B", "B", "C"],
            "amount": [10, 20, 30, 40, 50]
        })

        memory_store.save_dataframe("sales", df)

        result = memory_store.query("""
            SELECT category, SUM(amount) as total
            FROM sales
            GROUP BY category
            ORDER BY category
        """)

        assert len(result) == 3
        assert result.iloc[0]["total"] == 30  # A: 10 + 20
        assert result.iloc[1]["total"] == 70  # B: 30 + 40

    def test_set_and_get_state(self, memory_store):
        """Test state variable storage."""
        memory_store.set_state("total_revenue", 12345.67, step_number=1)
        memory_store.set_state("categories", ["A", "B", "C"], step_number=2)
        memory_store.set_state("config", {"limit": 100, "enabled": True}, step_number=2)

        assert memory_store.get_state("total_revenue") == 12345.67
        assert memory_store.get_state("categories") == ["A", "B", "C"]
        assert memory_store.get_state("config") == {"limit": 100, "enabled": True}
        assert memory_store.get_state("nonexistent") is None

    def test_get_all_state(self, memory_store):
        """Test getting all state variables."""
        memory_store.set_state("a", 1)
        memory_store.set_state("b", 2)

        all_state = memory_store.get_all_state()
        assert all_state == {"a": 1, "b": 2}

    def test_list_tables(self, memory_store):
        """Test listing tables with metadata."""
        df1 = pd.DataFrame({"x": [1, 2, 3]})
        df2 = pd.DataFrame({"y": [1, 2, 3, 4, 5]})

        memory_store.save_dataframe("table1", df1, step_number=1, description="First table")
        memory_store.save_dataframe("table2", df2, step_number=2, description="Second table")

        tables = memory_store.list_tables()

        assert len(tables) == 2
        assert tables[0]["name"] == "table1"
        assert tables[0]["row_count"] == 3
        assert tables[0]["step_number"] == 1
        assert tables[0]["description"] == "First table"

    def test_get_table_schema(self, memory_store):
        """Test getting table schema."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "value": [1.1, 2.2, 3.3]
        })

        memory_store.save_dataframe("test", df)
        schema = memory_store.get_table_schema("test")

        assert schema is not None
        assert len(schema) == 3

        col_names = [c["name"] for c in schema]
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names

    def test_drop_table(self, memory_store):
        """Test dropping a table."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        memory_store.save_dataframe("to_drop", df)

        assert memory_store.load_dataframe("to_drop") is not None

        memory_store.drop_table("to_drop")

        assert memory_store.load_dataframe("to_drop") is None
        assert len([t for t in memory_store.list_tables() if t["name"] == "to_drop"]) == 0

    def test_clear_step_data(self, memory_store):
        """Test clearing data from a specific step."""
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"y": [2]})

        memory_store.save_dataframe("step1_table", df1, step_number=1)
        memory_store.set_state("step1_var", "value1", step_number=1)

        memory_store.save_dataframe("step2_table", df2, step_number=2)
        memory_store.set_state("step2_var", "value2", step_number=2)

        # Clear step 1 data
        memory_store.clear_step_data(1)

        # Step 1 data should be gone
        assert memory_store.load_dataframe("step1_table") is None
        assert memory_store.get_state("step1_var") is None

        # Step 2 data should remain
        assert memory_store.load_dataframe("step2_table") is not None
        assert memory_store.get_state("step2_var") == "value2"

    def test_persistence(self, temp_db):
        """Test that data persists across store instances."""
        # Save data
        store1 = DataStore(db_path=temp_db)
        df = pd.DataFrame({"x": [1, 2, 3]})
        store1.save_dataframe("persistent", df)
        store1.set_state("key", "value")
        store1.close()

        # Load data in new instance
        store2 = DataStore(db_path=temp_db)
        loaded = store2.load_dataframe("persistent")
        state = store2.get_state("key")
        store2.close()

        assert loaded is not None
        assert len(loaded) == 3
        assert state == "value"

    def test_export_state_summary(self, memory_store):
        """Test exporting state summary."""
        df = pd.DataFrame({"x": [1, 2]})
        memory_store.save_dataframe("summary_table", df, step_number=1)
        memory_store.set_state("summary_var", 42)

        summary = memory_store.export_state_summary()

        assert "tables" in summary
        assert "state" in summary
        assert len(summary["tables"]) == 1
        assert summary["state"] == {"summary_var": 42}


class TestRichArtifacts:
    """Tests for rich artifact storage and retrieval."""

    def test_save_html_artifact(self, memory_store):
        """Test saving HTML artifact."""
        html_content = "<html><body><h1>Test Report</h1></body></html>"

        artifact = memory_store.save_html(
            name="test_report",
            html_content=html_content,
            step_number=1,
            title="Test Report",
            description="A test HTML report"
        )

        assert artifact.id > 0
        assert artifact.name == "test_report"
        assert artifact.artifact_type == ArtifactType.HTML
        assert artifact.content == html_content
        assert artifact.title == "Test Report"
        assert artifact.mime_type == "text/html"

    def test_save_chart_artifact(self, memory_store):
        """Test saving chart specification artifact."""
        vega_spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "data": {"values": [{"x": 1, "y": 2}]},
            "mark": "bar",
            "encoding": {
                "x": {"field": "x", "type": "ordinal"},
                "y": {"field": "y", "type": "quantitative"}
            }
        }

        artifact = memory_store.save_chart(
            name="revenue_chart",
            spec=vega_spec,
            step_number=2,
            title="Revenue by Region"
        )

        assert artifact.artifact_type == ArtifactType.CHART
        assert json.loads(artifact.content) == vega_spec
        assert artifact.metadata.get("chart_type") == "vega-lite"

    def test_save_plotly_chart(self, memory_store):
        """Test saving Plotly chart artifact."""
        plotly_spec = {"data": [{"x": [1, 2], "y": [3, 4], "type": "bar"}]}

        artifact = memory_store.save_chart(
            name="plotly_chart",
            spec=plotly_spec,
            step_number=1,
            chart_type="plotly"
        )

        assert artifact.artifact_type == ArtifactType.PLOTLY
        assert artifact.metadata.get("chart_type") == "plotly"

    def test_save_diagram_mermaid(self, memory_store):
        """Test saving Mermaid diagram artifact."""
        mermaid_code = """
        flowchart LR
            A[Customer] --> B[Order]
            B --> C[Invoice]
        """

        artifact = memory_store.save_diagram(
            name="order_flow",
            diagram_code=mermaid_code,
            diagram_format="mermaid",
            step_number=3,
            title="Order Flow Diagram"
        )

        assert artifact.artifact_type == ArtifactType.MERMAID
        assert artifact.content == mermaid_code
        assert artifact.metadata.get("format") == "mermaid"

    def test_save_diagram_graphviz(self, memory_store):
        """Test saving Graphviz diagram artifact."""
        dot_code = 'digraph { A -> B -> C }'

        artifact = memory_store.save_diagram(
            name="graph",
            diagram_code=dot_code,
            diagram_format="graphviz",
            step_number=1
        )

        assert artifact.artifact_type == ArtifactType.GRAPHVIZ
        assert artifact.metadata.get("format") == "graphviz"

    def test_save_image_png(self, memory_store):
        """Test saving PNG image artifact."""
        # Simulated base64 image data
        image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        artifact = memory_store.save_image(
            name="chart_image",
            image_data=image_data,
            image_format="png",
            step_number=4,
            title="Chart as Image",
            width=800,
            height=600
        )

        assert artifact.artifact_type == ArtifactType.PNG
        assert artifact.content == image_data
        assert artifact.is_binary
        assert artifact.metadata.get("width") == 800
        assert artifact.metadata.get("height") == 600

    def test_save_image_svg(self, memory_store):
        """Test saving SVG image artifact."""
        svg_content = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="50"/></svg>'

        artifact = memory_store.save_image(
            name="diagram_svg",
            image_data=svg_content,
            image_format="svg",
            step_number=1
        )

        assert artifact.artifact_type == ArtifactType.SVG
        assert not artifact.is_binary  # SVG is text, not binary
        assert artifact.mime_type == "image/svg+xml"

    def test_save_rich_artifact_with_metadata(self, memory_store):
        """Test saving artifact with custom metadata."""
        artifact = memory_store.save_rich_artifact(
            name="custom_artifact",
            artifact_type=ArtifactType.JSON,
            content='{"key": "value"}',
            step_number=1,
            title="Custom Data",
            metadata={
                "source": "api_response",
                "version": "1.0",
                "record_count": 100
            }
        )

        assert artifact.metadata["source"] == "api_response"
        assert artifact.metadata["version"] == "1.0"
        assert artifact.metadata["record_count"] == 100

    def test_get_artifact_by_name(self, memory_store):
        """Test retrieving artifact by name."""
        memory_store.save_html(
            name="unique_report",
            html_content="<p>Content</p>",
            step_number=1
        )

        retrieved = memory_store.get_artifact_by_name("unique_report")

        assert retrieved is not None
        assert retrieved.name == "unique_report"
        assert retrieved.artifact_type == ArtifactType.HTML

    def test_get_artifact_by_id(self, memory_store):
        """Test retrieving artifact by ID."""
        artifact = memory_store.save_html(
            name="id_test",
            html_content="<p>Test</p>",
            step_number=1
        )

        retrieved = memory_store.get_artifact_by_id(artifact.id)

        assert retrieved is not None
        assert retrieved.id == artifact.id
        assert retrieved.name == "id_test"

    def test_get_artifacts_by_step(self, memory_store):
        """Test filtering artifacts by step number."""
        memory_store.save_html(name="step1_a", html_content="a", step_number=1)
        memory_store.save_html(name="step1_b", html_content="b", step_number=1)
        memory_store.save_html(name="step2_a", html_content="c", step_number=2)

        step1_artifacts = memory_store.get_artifacts(step_number=1)
        step2_artifacts = memory_store.get_artifacts(step_number=2)

        assert len(step1_artifacts) == 2
        assert len(step2_artifacts) == 1

    def test_get_artifacts_by_type(self, memory_store):
        """Test filtering artifacts by type."""
        memory_store.save_html(name="html1", html_content="<p>1</p>", step_number=1)
        memory_store.save_chart(name="chart1", spec={"data": []}, step_number=1)
        memory_store.save_html(name="html2", html_content="<p>2</p>", step_number=2)

        html_artifacts = memory_store.get_artifacts_by_type(ArtifactType.HTML)
        chart_artifacts = memory_store.get_artifacts_by_type("chart")

        assert len(html_artifacts) == 2
        assert len(chart_artifacts) == 1

    def test_list_artifacts_without_content(self, memory_store):
        """Test listing artifacts without content for efficiency."""
        memory_store.save_html(
            name="large_report",
            html_content="x" * 10000,  # Large content
            step_number=1,
            title="Large Report"
        )

        listing = memory_store.list_artifacts(include_content=False)

        assert len(listing) == 1
        assert "content" not in listing[0]
        assert listing[0]["content_length"] == 10000
        assert listing[0]["title"] == "Large Report"

    def test_list_artifacts_with_content(self, memory_store):
        """Test listing artifacts with content included."""
        memory_store.save_html(
            name="small_report",
            html_content="<p>Small</p>",
            step_number=1
        )

        listing = memory_store.list_artifacts(include_content=True)

        assert len(listing) == 1
        assert listing[0]["content"] == "<p>Small</p>"

    def test_artifact_to_dict(self, memory_store):
        """Test artifact serialization to dict."""
        artifact = memory_store.save_chart(
            name="serializable",
            spec={"data": []},
            step_number=1,
            title="Serializable Chart"
        )

        as_dict = artifact.to_dict()

        assert as_dict["id"] == artifact.id
        assert as_dict["name"] == "serializable"
        assert as_dict["type"] == "chart"
        assert as_dict["title"] == "Serializable Chart"
        assert "content_type" in as_dict

    def test_backward_compatible_add_artifact(self, memory_store):
        """Test that legacy add_artifact still works."""
        artifact_id = memory_store.add_artifact(
            step_number=1,
            attempt=1,
            artifact_type="code",
            content="print('hello')"
        )

        assert artifact_id > 0

        # Retrieve via get_artifacts
        artifacts = memory_store.get_artifacts(step_number=1)
        assert len(artifacts) == 1
        assert artifacts[0].content == "print('hello')"
