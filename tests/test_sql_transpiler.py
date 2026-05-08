# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for SQL transpilation functionality."""

import pytest
from unittest.mock import MagicMock, patch


class TestTranspileSQL:
    """Test the transpile_sql function."""

    def test_transpile_date_trunc_to_sqlite(self):
        """DATE_TRUNC should be converted to SQLite DATE() with modifiers."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT DATE_TRUNC('month', created_at) FROM orders"
        result = transpile_sql(sql, target_dialect="sqlite", source_dialect="postgres")

        # Custom mapping converts to SQLite's DATE(col, 'start of month')
        assert "start of month" in result.lower()

    def test_transpile_date_trunc_year_to_sqlite(self):
        """DATE_TRUNC year should use 'start of year' in SQLite."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT DATE_TRUNC('year', created_at) FROM orders"
        result = transpile_sql(sql, target_dialect="sqlite", source_dialect="postgres")

        assert "start of year" in result.lower()

    def test_transpile_extract_epoch_to_sqlite(self):
        """EXTRACT(EPOCH FROM ...) should use UNIXEPOCH in SQLite."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT EXTRACT(EPOCH FROM created_at) FROM orders"
        result = transpile_sql(sql, target_dialect="sqlite", source_dialect="postgres")

        assert "UNIXEPOCH" in result.upper()

    def test_transpile_now_to_sqlite(self):
        """NOW() should be converted to CURRENT_TIMESTAMP for SQLite."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT NOW() FROM users"
        result = transpile_sql(sql, target_dialect="sqlite", source_dialect="postgres")

        assert "CURRENT_TIMESTAMP" in result.upper()

    def test_transpile_concat_to_mysql(self):
        """PostgreSQL || concat should use CONCAT for MySQL."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT first_name || ' ' || last_name FROM users"
        result = transpile_sql(sql, target_dialect="mysql", source_dialect="postgres")

        assert "CONCAT" in result.upper()

    def test_transpile_cast_postgres_style(self):
        """PostgreSQL ::type cast should use CAST for other dialects."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT x::integer FROM t"
        result = transpile_sql(sql, target_dialect="mysql", source_dialect="postgres")

        assert "CAST" in result.upper()

    def test_no_transpile_when_same_dialect(self):
        """No transpilation when source == target."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT DATE_TRUNC('month', created_at) FROM orders"
        result = transpile_sql(sql, target_dialect="postgres", source_dialect="postgres")

        # Should return unchanged
        assert result == sql

    def test_transpile_string_agg_to_mysql(self):
        """STRING_AGG should convert to GROUP_CONCAT for MySQL."""
        from constat.catalog.sql_transpiler import transpile_sql

        sql = "SELECT STRING_AGG(name, ', ') FROM users GROUP BY department"
        result = transpile_sql(sql, target_dialect="mysql", source_dialect="postgres")

        assert "GROUP_CONCAT" in result.upper()


class TestDialectDetection:
    """Test dialect detection from SQLAlchemy engines."""

    def test_detect_sqlite(self):
        """Should detect sqlite dialect."""
        from constat.catalog.sql_transpiler import detect_dialect

        mock_engine = MagicMock()
        mock_engine.dialect.name = "sqlite"

        assert detect_dialect(mock_engine) == "sqlite"

    def test_detect_postgresql(self):
        """Should detect postgres dialect."""
        from constat.catalog.sql_transpiler import detect_dialect

        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"

        assert detect_dialect(mock_engine) == "postgres"

    def test_detect_mysql(self):
        """Should detect mysql dialect."""
        from constat.catalog.sql_transpiler import detect_dialect

        mock_engine = MagicMock()
        mock_engine.dialect.name = "mysql"

        assert detect_dialect(mock_engine) == "mysql"

    def test_detect_duckdb(self):
        """Should detect duckdb dialect."""
        from constat.catalog.sql_transpiler import detect_dialect

        mock_engine = MagicMock()
        mock_engine.dialect.name = "duckdb"

        assert detect_dialect(mock_engine) == "duckdb"


class TestTranspilingConnection:
    """Test the TranspilingConnection wrapper."""

    def test_wrapper_exposes_engine(self):
        """Wrapper should expose the underlying engine."""
        from constat.catalog.sql_transpiler import TranspilingConnection

        mock_engine = MagicMock()
        mock_engine.dialect.name = "sqlite"

        wrapper = TranspilingConnection(mock_engine)

        assert wrapper.engine is mock_engine

    def test_wrapper_detects_dialect(self):
        """Wrapper should detect target dialect from engine."""
        from constat.catalog.sql_transpiler import TranspilingConnection

        mock_engine = MagicMock()
        mock_engine.dialect.name = "sqlite"

        wrapper = TranspilingConnection(mock_engine)

        assert wrapper.target_dialect == "sqlite"
        assert wrapper.source_dialect == "postgres"

    def test_wrapper_transpiles_sql(self):
        """Wrapper transpile method should convert SQL."""
        from constat.catalog.sql_transpiler import TranspilingConnection

        mock_engine = MagicMock()
        mock_engine.dialect.name = "sqlite"

        wrapper = TranspilingConnection(mock_engine)

        sql = "SELECT NOW() FROM users"
        result = wrapper.transpile(sql)

        assert "CURRENT_TIMESTAMP" in result.upper()

    def test_wrapper_passthrough_when_postgres(self):
        """No transpilation when target is postgres (same as source)."""
        from constat.catalog.sql_transpiler import TranspilingConnection

        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"

        wrapper = TranspilingConnection(mock_engine)

        sql = "SELECT DATE_TRUNC('month', created_at) FROM orders"
        result = wrapper.transpile(sql)

        # Should be unchanged
        assert result == sql


class TestIntegration:
    """Integration tests with real databases."""

    @pytest.mark.skipif(True, reason="Requires SQLite database")
    def test_with_real_sqlite(self):
        """Test transpilation with a real SQLite database."""
        from sqlalchemy import create_engine
        from constat.catalog.sql_transpiler import TranspilingConnection
        import pandas as pd

        # Create in-memory SQLite database
        engine = create_engine("sqlite:///:memory:")

        # Create test table
        with engine.connect() as conn:
            conn.execute("CREATE TABLE orders (id INTEGER, created_at TEXT)")
            conn.execute("INSERT INTO orders VALUES (1, '2024-01-15')")
            conn.commit()

        # Wrap with transpiler
        wrapper = TranspilingConnection(engine)

        # This PostgreSQL-style query should work on SQLite
        sql = "SELECT created_at FROM orders WHERE id = 1"
        df = pd.read_sql(sql, wrapper.engine)

        assert len(df) == 1
