# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for JDBC source configuration and connection dispatch."""

from unittest.mock import MagicMock, patch
import pytest

from constat.core.source_config import DatabaseConfig


# ---------------------------------------------------------------------------
# DatabaseConfig — JDBC fields and is_jdbc()
# ---------------------------------------------------------------------------

def test_is_jdbc_true():
    cfg = DatabaseConfig(type="jdbc", jdbc_driver="com.example.Driver", jdbc_url="jdbc:example://host/db")
    assert cfg.is_jdbc() is True


def test_is_jdbc_false_for_sql():
    cfg = DatabaseConfig(type="sql", uri="postgresql://localhost/db")
    assert cfg.is_jdbc() is False


def test_is_jdbc_false_for_nosql():
    cfg = DatabaseConfig(type="mongodb", uri="mongodb://localhost/db")
    assert cfg.is_jdbc() is False


def test_jdbc_fields_stored():
    cfg = DatabaseConfig(
        type="jdbc",
        jdbc_driver="com.sap.db.jdbc.Driver",
        jdbc_url="jdbc:sap://host:30015/",
        jar_path="/opt/ngdbc.jar",
        username="admin",
        password="secret",
    )
    assert cfg.jdbc_driver == "com.sap.db.jdbc.Driver"
    assert cfg.jdbc_url == "jdbc:sap://host:30015/"
    assert cfg.jdbc_path == "/opt/ngdbc.jar" if hasattr(cfg, "jdbc_path") else cfg.jar_path == "/opt/ngdbc.jar"
    assert cfg.username == "admin"
    assert cfg.password == "secret"


def test_jdbc_jar_path_list():
    cfg = DatabaseConfig(
        type="jdbc",
        jdbc_driver="com.example.Driver",
        jdbc_url="jdbc:example://host/db",
        jar_path=["/opt/a.jar", "/opt/b.jar"],
    )
    assert isinstance(cfg.jar_path, list)
    assert len(cfg.jar_path) == 2


# ---------------------------------------------------------------------------
# SchemaManager._connect_jdbc — missing JayDeBeApi raises ImportError
# ---------------------------------------------------------------------------

def test_connect_jdbc_missing_jaydebeapi():
    """ImportError with install hint when JayDeBeApi is not installed."""
    from constat.catalog.schema_manager import SchemaManager

    sm = SchemaManager.__new__(SchemaManager)
    sm.connections = {}
    sm._read_only_databases = set()
    sm.config = None

    cfg = DatabaseConfig(
        type="jdbc",
        jdbc_driver="com.example.Driver",
        jdbc_url="jdbc:example://host/db",
    )

    with patch.dict("sys.modules", {"jaydebeapi": None}):
        with pytest.raises(ImportError, match="constat\\[jdbc\\]"):
            sm._connect_jdbc("mydb", cfg)


def test_connect_jdbc_missing_driver_raises():
    """ValueError when jdbc_driver is not set."""
    from constat.catalog.schema_manager import SchemaManager

    sm = SchemaManager.__new__(SchemaManager)
    sm.connections = {}
    sm._read_only_databases = set()
    sm.config = None

    cfg = DatabaseConfig(type="jdbc", jdbc_url="jdbc:example://host/db")

    fake_jaydebeapi = MagicMock()
    with patch.dict("sys.modules", {"jaydebeapi": fake_jaydebeapi}):
        with pytest.raises(ValueError, match="jdbc_driver"):
            sm._connect_jdbc("mydb", cfg)


def test_connect_jdbc_missing_url_raises():
    """ValueError when jdbc_url is not set."""
    from constat.catalog.schema_manager import SchemaManager

    sm = SchemaManager.__new__(SchemaManager)
    sm.connections = {}
    sm._read_only_databases = set()
    sm.config = None

    cfg = DatabaseConfig(type="jdbc", jdbc_driver="com.example.Driver")

    fake_jaydebeapi = MagicMock()
    with patch.dict("sys.modules", {"jaydebeapi": fake_jaydebeapi}):
        with pytest.raises(ValueError, match="jdbc_url"):
            sm._connect_jdbc("mydb", cfg)


def test_connect_jdbc_success():
    """Happy path: JayDeBeApi used to test connection, engine wrapped in TranspilingConnection."""
    from constat.catalog.schema_manager import SchemaManager

    sm = SchemaManager.__new__(SchemaManager)
    sm.connections = {}
    sm._read_only_databases = set()
    sm.config = None

    cfg = DatabaseConfig(
        type="jdbc",
        jdbc_driver="com.example.Driver",
        jdbc_url="jdbc:example://host/db",
        jar_path="/opt/driver.jar",
        username="user",
        password="pass",
    )

    # Build a fake JayDeBeApi module; connect() returns a mock DBAPI2 connection.
    fake_conn = MagicMock()
    fake_cursor = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    fake_jaydebeapi = MagicMock()
    fake_jaydebeapi.connect.return_value = fake_conn

    with patch.dict("sys.modules", {"jaydebeapi": fake_jaydebeapi}):
        with patch("constat.catalog.schema_manager.TranspilingConnection") as mock_tc:
            sm._connect_jdbc("mydb", cfg)

    # JayDeBeApi.connect() was called for the connectivity test
    fake_jaydebeapi.connect.assert_called_once_with(
        "com.example.Driver",
        "jdbc:example://host/db",
        ["user", "pass"],
        ["/opt/driver.jar"],
    )
    # TranspilingConnection wraps the constructed Engine
    mock_tc.assert_called_once()
    assert "mydb" in sm.connections
