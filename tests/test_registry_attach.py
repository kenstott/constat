"""Tests for ConstatRegistry ATTACH vectors functionality."""

import duckdb
import pytest

from constat.storage.registry import ConstatRegistry


@pytest.fixture
def registry(tmp_path):
    reg = ConstatRegistry(base_dir=tmp_path)
    yield reg
    reg.close()


def _create_vectors_db(path):
    """Create a vectors.duckdb with a test table."""
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE embeddings (id INTEGER, vec FLOAT[])")
    conn.execute("INSERT INTO embeddings VALUES (1, [0.1, 0.2]), (2, [0.3, 0.4])")
    conn.close()


def test_attach_vectors_missing_file(registry):
    assert registry.attach_vectors() is False


def test_attach_vectors_success(registry):
    _create_vectors_db(registry._vectors_path)
    assert registry.attach_vectors() is True
    rows = registry.query_vectors("SELECT * FROM vectors.embeddings ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0][0] == 1


def test_detach_vectors(registry):
    _create_vectors_db(registry._vectors_path)
    registry.attach_vectors()
    registry.detach_vectors()
    with pytest.raises(duckdb.CatalogException):
        registry.query_vectors("SELECT * FROM vectors.embeddings")


def test_attach_after_recreate(registry, tmp_path):
    vectors_path = registry._vectors_path

    # First attach
    _create_vectors_db(vectors_path)
    registry.attach_vectors()
    assert registry.query_vectors("SELECT count(*) FROM vectors.embeddings").fetchone()[0] == 2

    # Detach and delete
    registry.detach_vectors()
    vectors_path.unlink()

    # Recreate with different data
    conn = duckdb.connect(str(vectors_path))
    conn.execute("CREATE TABLE embeddings (id INTEGER, vec FLOAT[])")
    conn.execute("INSERT INTO embeddings VALUES (10, [0.5, 0.6])")
    conn.close()

    # Re-attach
    assert registry.attach_vectors() is True
    rows = registry.query_vectors("SELECT * FROM vectors.embeddings").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 10
