# Copyright (c) 2025 Kenneth Stott
# Canary: 3b72ab1f-b6ed-4a0a-8373-e68d243a7ef8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for DuckDBSessionStore."""

from __future__ import annotations
import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from constat.storage.duckdb_session_store import DuckDBSessionStore
from constat.storage.registry import ConstatRegistry
from constat.core.models import ArtifactType


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry(tmp_dir):
    return ConstatRegistry(base_dir=tmp_dir / "registry")


@pytest.fixture
def store(tmp_dir, registry):
    s = DuckDBSessionStore(
        db_path=tmp_dir / "session.duckdb",
        registry=registry,
        user_id="test_user",
        session_id="test_session",
    )
    yield s
    s.close()


class TestSaveLoadRoundTrip:
    def test_save_and_load(self, store):
        df = pd.DataFrame({"name": ["Alice", "Bob"], "value": [100, 200]})
        store.save_dataframe("test", df, step_number=1)
        loaded = store.load_dataframe("test")
        assert loaded is not None
        assert len(loaded) == 2
        assert loaded["value"].sum() == 300

    def test_save_empty_skipped(self, store):
        store.save_dataframe("empty", pd.DataFrame(), step_number=1)
        assert store.load_dataframe("empty") is None

    def test_load_nonexistent(self, store):
        assert store.load_dataframe("nope") is None

    def test_save_list_input(self, store):
        store.save_dataframe("from_list", [{"a": 1}, {"a": 2}], step_number=1)
        loaded = store.load_dataframe("from_list")
        assert loaded is not None
        assert len(loaded) == 2


class TestQuery:
    def test_basic_query(self, store):
        df = pd.DataFrame({"cat": ["A", "A", "B"], "val": [10, 20, 30]})
        store.save_dataframe("sales", df)
        result = store.query("SELECT cat, SUM(val) as total FROM sales GROUP BY cat ORDER BY cat")
        assert len(result) == 2
        assert result.iloc[0]["total"] == 30

    def test_pg_syntax_transpiled(self, store):
        df = pd.DataFrame({"x": [1, 2, 3]})
        store.save_dataframe("nums", df)
        # PG-style LIMIT should work via transpilation
        result = store.query("SELECT * FROM nums LIMIT 2")
        assert len(result) == 2


class TestVersioning:
    def test_version_archive(self, store):
        df1 = pd.DataFrame({"v": [1]})
        df2 = pd.DataFrame({"v": [2, 3]})
        store.save_dataframe("t", df1, step_number=1)
        store.save_dataframe("t", df2, step_number=2)

        # Current version should be v2
        loaded = store.load_dataframe("t")
        assert len(loaded) == 2

        # Version history
        versions = store.get_table_versions("t")
        assert len(versions) == 2
        assert versions[0]["is_current"] is True
        assert versions[0]["version"] == 2

    def test_load_specific_version(self, store):
        store.save_dataframe("t", pd.DataFrame({"v": [1]}), step_number=1)
        store.save_dataframe("t", pd.DataFrame({"v": [2, 3]}), step_number=2)

        v1 = store.load_table_version("t", 1)
        assert v1 is not None
        assert len(v1) == 1

        v2 = store.load_table_version("t", 2)
        assert v2 is not None
        assert len(v2) == 2


class TestDropTable:
    def test_drop_removes_all(self, store):
        store.save_dataframe("t", pd.DataFrame({"v": [1]}), step_number=1)
        store.save_dataframe("t", pd.DataFrame({"v": [2]}), step_number=2)
        assert store.drop_table("t") is True
        assert store.load_dataframe("t") is None

    def test_drop_nonexistent(self, store):
        assert store.drop_table("nope") is True  # no error, just cleans up


class TestSchema:
    def test_get_table_schema(self, store):
        store.save_dataframe("t", pd.DataFrame({"a": [1], "b": ["x"]}))
        schema = store.get_table_schema("t")
        assert schema is not None
        assert len(schema) == 2
        names = {c["name"] for c in schema}
        assert names == {"a", "b"}

    def test_table_exists(self, store):
        assert store.table_exists("nope") is False
        store.save_dataframe("t", pd.DataFrame({"a": [1]}))
        assert store.table_exists("t") is True


class TestState:
    def test_set_and_get(self, store):
        store.set_state("key1", {"nested": True}, step_number=1)
        assert store.get_state("key1") == {"nested": True}

    def test_get_all(self, store):
        store.set_state("a", 1)
        store.set_state("b", "two")
        state = store.get_all_state()
        assert state == {"a": 1, "b": "two"}

    def test_upsert(self, store):
        store.set_state("x", 1)
        store.set_state("x", 2)
        assert store.get_state("x") == 2

    def test_starred_tables(self, store):
        assert store.get_starred_tables() == []
        store.set_starred_tables(["t1", "t2"])
        assert store.get_starred_tables() == ["t1", "t2"]

    def test_toggle_star(self, store):
        assert store.toggle_table_star("t1") is True
        assert store.toggle_table_star("t1") is False


class TestScratchpad:
    def test_add_and_get(self, store):
        store.add_scratchpad_entry(1, "goal1", "narrative1", ["t1"])
        entry = store.get_scratchpad_entry(1)
        assert entry is not None
        assert entry["goal"] == "goal1"
        assert entry["tables_created"] == ["t1"]

    def test_get_all(self, store):
        store.add_scratchpad_entry(1, "g1", "n1")
        store.add_scratchpad_entry(2, "g2", "n2")
        entries = store.get_scratchpad()
        assert len(entries) == 2

    def test_markdown(self, store):
        store.add_scratchpad_entry(1, "Test Goal", "Did something")
        md = store.get_scratchpad_as_markdown()
        assert "Test Goal" in md
        assert "Did something" in md


class TestArtifacts:
    def test_add_and_get(self, store):
        aid = store.add_artifact(1, 1, "code", "print('hello')")
        artifacts = store.get_artifacts(step_number=1)
        assert len(artifacts) == 1
        assert artifacts[0].content == "print('hello')"

    def test_get_by_name(self, store):
        store.add_artifact(1, 1, "html", "<h1>Hi</h1>", name="report")
        a = store.get_artifact_by_name("report")
        assert a is not None
        assert a.name == "report"

    def test_get_by_id(self, store):
        aid = store.add_artifact(1, 1, "code", "x=1")
        a = store.get_artifact_by_id(aid)
        assert a is not None
        assert a.id == aid

    def test_delete_artifact(self, store):
        aid = store.add_artifact(1, 1, "code", "x=1", name="test_art")
        assert store.delete_artifact(aid) is True
        assert store.get_artifact_by_id(aid) is None

    def test_list_artifacts(self, store):
        store.add_artifact(1, 1, "code", "x=1", name="c1")
        store.add_artifact(2, 1, "output", "result", name="o1")
        listing = store.list_artifacts()
        assert len(listing) == 2

    def test_save_rich_artifact(self, store):
        a = store.save_rich_artifact("chart1", ArtifactType.CHART, '{"data": []}')
        assert a.name == "chart1"
        assert a.artifact_type == ArtifactType.CHART

    def test_update_metadata(self, store):
        aid = store.add_artifact(1, 1, "code", "x=1")
        assert store.update_artifact_metadata(aid, {"key": "value"}) is True
        a = store.get_artifact_by_id(aid)
        assert a.metadata.get("key") == "value"


class TestSessionMeta:
    def test_set_and_get(self, store):
        store.set_session_meta("problem", "find revenue")
        assert store.get_session_meta("problem") == "find revenue"

    def test_get_all(self, store):
        store.set_session_meta("a", "1")
        store.set_session_meta("b", "2")
        meta = store.get_all_session_meta()
        assert meta == {"a": "1", "b": "2"}


class TestPlanSteps:
    def test_save_and_get(self, store):
        store.save_plan_step(1, "Query DB", expected_inputs=["db"], expected_outputs=["t1"])
        steps = store.get_plan_steps()
        assert len(steps) == 1
        assert steps[0]["goal"] == "Query DB"

    def test_update_step(self, store):
        store.save_plan_step(1, "Query DB")
        store.update_plan_step(1, status="completed", attempts=2)
        steps = store.get_plan_steps()
        assert steps[0]["status"] == "completed"
        assert steps[0]["attempts"] == 2


class TestBulkClear:
    def test_clear_state_before_step(self, store):
        store.set_state("old", 1, step_number=1)
        store.set_state("new", 2, step_number=5)
        cleared = store.clear_state_before_step(3)
        assert cleared == 1
        assert store.get_state("old") is None
        assert store.get_state("new") == 2

    def test_clear_scratchpad(self, store):
        store.add_scratchpad_entry(1, "g", "n")
        store.clear_scratchpad()
        assert store.get_scratchpad() == []

    def test_clear_artifacts(self, store):
        store.add_artifact(1, 1, "code", "x")
        store.clear_artifacts()
        assert store.get_artifacts() == []

    def test_clear_plan_steps(self, store):
        store.save_plan_step(1, "g")
        store.clear_plan_steps()
        assert store.get_plan_steps() == []

    def test_clear_state(self, store):
        store.set_state("k", "v")
        store.clear_state()
        assert store.get_all_state() == {}


class TestTruncation:
    def test_truncate_from_step(self, store):
        store.add_scratchpad_entry(1, "g1", "n1")
        store.add_scratchpad_entry(2, "g2", "n2")
        store.add_scratchpad_entry(3, "g3", "n3")
        store.set_state("s1", "v1", step_number=1)
        store.set_state("s3", "v3", step_number=3)
        store.truncate_from_step(2)
        entries = store.get_scratchpad()
        assert len(entries) == 1
        assert entries[0]["step_number"] == 1
        assert store.get_state("s1") == "v1"
        assert store.get_state("s3") is None

    def test_clear_session_data(self, store):
        store.add_scratchpad_entry(1, "g", "n")
        store.set_state("k", "v")
        store.add_artifact(1, 1, "code", "x")
        store.save_plan_step(1, "g")
        store.clear_session_data()
        assert store.get_scratchpad() == []
        assert store.get_all_state() == {}
        assert store.get_artifacts() == []
        assert store.get_plan_steps() == []


class TestCreateView:
    def test_lazy_view(self, store):
        store.save_dataframe("base", pd.DataFrame({"x": [1, 2, 3]}))
        store.create_view("filtered", "SELECT * FROM base WHERE x > 1")
        result = store.load_dataframe("filtered")
        assert result is not None
        assert len(result) == 2

    def test_view_registered_in_list_tables(self, store):
        store.save_dataframe("base", pd.DataFrame({"x": [1, 2, 3]}))
        store.create_view("my_view", "SELECT * FROM base WHERE x > 1", step_number=2, description="filtered")
        tables = {t["name"] for t in store.list_tables()}
        assert "my_view" in tables

    def test_auto_convert_query_to_view(self, store):
        """save_dataframe auto-converts to view when data is unmodified query result."""
        store.save_dataframe("source", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        df = store.query("SELECT * FROM source WHERE a > 1")
        assert "_source_sql" in df.attrs
        # save_dataframe should auto-convert to view
        store.save_dataframe("derived", df, step_number=1)
        # Verify it's queryable and has correct data
        result = store.load_dataframe("derived")
        assert len(result) == 2
        # Verify it's in list_tables
        tables = {t["name"] for t in store.list_tables()}
        assert "derived" in tables

    def test_no_auto_convert_when_columns_changed(self, store):
        """save_dataframe does NOT auto-convert when columns differ from original query."""
        store.save_dataframe("source", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        df = store.query("SELECT * FROM source")
        # Column selection: attrs are shared but columns differ from _source_columns
        subset = df[["a"]]
        assert list(subset.columns) != df.attrs.get("_source_columns")
        # This should do a normal save, not a view
        store.save_dataframe("subset", subset, step_number=1)
        result = store.load_dataframe("subset")
        assert len(result) == 3
        assert list(result.columns) == ["a"]

    def test_no_auto_convert_when_rows_filtered(self, store):
        """save_dataframe does NOT auto-convert when rows are filtered."""
        store.save_dataframe("source", pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))
        df = store.query("SELECT * FROM source")
        filtered = df[df["a"] > 1]
        # Rows differ from _source_len
        assert len(filtered) != df.attrs.get("_source_len")
        store.save_dataframe("filtered", filtered, step_number=1)
        result = store.load_dataframe("filtered")
        assert len(result) == 2

    def test_stale_view_recovery(self, store):
        """Views auto-recover when upstream schema changes (column rename)."""
        store.save_dataframe("upstream", pd.DataFrame({"salary": [100, 200]}))
        df = store.query("SELECT * FROM upstream")
        store.save_dataframe("downstream", df, step_number=1)
        assert store.load_dataframe("downstream") is not None
        # Recreate upstream with different column name
        store.save_dataframe("upstream", pd.DataFrame({"current_salary": [100, 200]}))
        # downstream view is now stale — should auto-recover
        result = store.load_dataframe("downstream")
        assert result is not None
        assert "current_salary" in result.columns

    def test_no_auto_convert_for_non_select(self, store):
        """save_dataframe does NOT auto-convert for SHOW TABLES or other non-SELECT SQL."""
        # Simulate a DataFrame from store.query("SHOW TABLES")
        df = pd.DataFrame({"name": ["table_a", "table_b"]})
        df.attrs["_source_sql"] = "SHOW TABLES"
        df.attrs["_source_columns"] = list(df.columns)
        df.attrs["_source_len"] = len(df)
        # Should save as a table, not try to create_view
        store.save_dataframe("available_tables", df, step_number=1)
        result = store.load_dataframe("available_tables")
        assert len(result) == 2
        assert list(result.columns) == ["name"]


class TestAttachSQLite:
    def test_attach_and_query(self, store, tmp_dir):
        # Create a SQLite file with test data
        sqlite_path = tmp_dir / "source.db"
        conn = sqlite3.connect(str(sqlite_path))
        conn.execute("CREATE TABLE customers (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO customers VALUES (1, 'Alice'), (2, 'Bob')")
        conn.commit()
        conn.close()

        store.attach("src", str(sqlite_path), db_type="sqlite", read_only=True)
        result = store.query("SELECT * FROM src.customers ORDER BY id")
        assert len(result) == 2
        assert result.iloc[0]["name"] == "Alice"


class TestRegisterFile:
    def test_csv_view(self, store, tmp_dir):
        csv_path = tmp_dir / "data.csv"
        csv_path.write_text("a,b\n1,x\n2,y\n")
        store.register_file("csv_data", str(csv_path), "csv")
        result = store.load_dataframe("csv_data")
        assert result is not None
        assert len(result) == 2

    def test_parquet_view(self, store, tmp_dir):
        parquet_path = tmp_dir / "data.parquet"
        pd.DataFrame({"c": [10, 20]}).to_parquet(str(parquet_path))
        store.register_file("pq_data", str(parquet_path), "parquet")
        result = store.load_dataframe("pq_data")
        assert result is not None
        assert len(result) == 2

    def test_unsupported_format(self, store):
        with pytest.raises(ValueError, match="Unsupported"):
            store.register_file("bad", "/tmp/f.xlsx", "xlsx")


class TestLegacyParquetMigration:
    def test_imports_existing_parquet(self, tmp_dir, registry):
        # Create legacy tables directory with Parquet files
        tables_dir = tmp_dir / "session" / "tables"
        tables_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1, 2]}).to_parquet(str(tables_dir / "legacy_table.parquet"))

        # Create store (parent is session/)
        s = DuckDBSessionStore(
            db_path=tmp_dir / "session" / "session.duckdb",
            registry=registry,
            user_id="test_user",
            session_id="test_session",
        )
        try:
            loaded = s.load_dataframe("legacy_table")
            assert loaded is not None
            assert len(loaded) == 2
        finally:
            s.close()


class TestEngineProperty:
    def test_raises_attribute_error(self, store):
        with pytest.raises(AttributeError, match="does not use SQLAlchemy"):
            _ = store.engine


class TestSharing:
    def test_shared_users(self, store):
        assert store.get_shared_users() == []
        store.add_shared_user("user2")
        assert store.get_shared_users() == ["user2"]
        store.remove_shared_user("user2")
        assert store.get_shared_users() == []

    def test_public(self, store):
        assert store.is_public() is False
        store.set_public(True)
        assert store.is_public() is True


class TestFullSessionState:
    def test_get_full_state(self, store):
        store.set_session_meta("problem", "test")
        store.save_plan_step(1, "g")
        store.save_dataframe("t", pd.DataFrame({"a": [1]}), step_number=1)
        store.set_state("k", "v")
        store.add_scratchpad_entry(1, "g", "n")
        store.add_artifact(1, 1, "code", "x", name="c1")

        state = store.get_full_session_state()
        assert "session" in state
        assert "plan_steps" in state
        assert "tables" in state
        assert "state" in state
        assert "scratchpad" in state
        assert "artifacts" in state


class TestCloseAndReopen:
    def test_persistence(self, tmp_dir, registry):
        db_path = tmp_dir / "session.duckdb"
        s = DuckDBSessionStore(db_path=db_path, registry=registry,
                               user_id="u", session_id="s")
        s.save_dataframe("t", pd.DataFrame({"x": [42]}), step_number=1)
        s.set_state("k", "v")
        s.add_scratchpad_entry(1, "g", "n")
        s.close()

        # Reopen
        s2 = DuckDBSessionStore(db_path=db_path, registry=registry,
                                user_id="u", session_id="s")
        try:
            loaded = s2.load_dataframe("t")
            assert loaded is not None
            assert loaded.iloc[0]["x"] == 42
            assert s2.get_state("k") == "v"
            assert len(s2.get_scratchpad()) == 1
        finally:
            s2.close()


class TestListTablesIsView:
    def test_list_tables_is_view_flag(self, store):
        """Tables should have is_view=False, views should have is_view=True."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        store.save_dataframe("real_table", df, step_number=1)
        # Create a view via the internal connection
        with store._locked_conn() as conn:
            conn.execute("CREATE VIEW my_view AS SELECT * FROM real_table WHERE a > 1")
        # Register the view so it shows up in list_tables
        store._registry.register_table(
            user_id=store._user_id,
            session_id=store._session_id,
            name="my_view",
            file_path="",
            step_number=1,
            row_count=1,
        )

        tables = {t["name"]: t for t in store.list_tables()}
        assert tables["real_table"]["is_view"] is False
        assert tables["my_view"]["is_view"] is True
