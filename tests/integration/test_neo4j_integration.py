from __future__ import annotations

# Copyright (c) 2025 Kenneth Stott
# Canary: 5f2a1c8e-3b74-4d9a-8e11-0c6d9f7a2b45
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Integration tests for Neo4j connector against a live Docker-started Neo4j instance."""

import time
from typing import Generator

import pytest

from tests.integration.fixtures_docker_helpers import (
    get_unique_container_name,
    get_unique_port,
    start_container,
    stop_container,
    is_docker_available,
)

pytestmark = pytest.mark.integration

try:
    import neo4j as _neo4j_mod
except ImportError:
    pytest.fail("neo4j is required but not installed — run: pip install neo4j")

from constat.catalog.nosql import Neo4jConnector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NEO4J_DEFAULT_PORT = 7687
_NEO4J_HTTP_PORT = 7474
_NEO4J_PASSWORD = "testpassword"
_NEO4J_USER = "neo4j"


def _is_neo4j_ready(bolt_port: int, timeout: int = 5) -> bool:
    """Check if Neo4j bolt port is accepting connections."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            f"bolt://localhost:{bolt_port}",
            auth=(_NEO4J_USER, _NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def _wait_for_neo4j(bolt_port: int, timeout: int = 120) -> bool:
    """Poll until Neo4j is ready or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if _is_neo4j_ready(bolt_port):
            return True
        time.sleep(3)
    return False


def _load_movies_data(connector: Neo4jConnector) -> None:
    """Seed a small movies graph into the connected Neo4j instance."""
    connector.cypher("MATCH (n) DETACH DELETE n")

    connector.cypher(
        "CREATE (p:Person {name: $name, born: $born})",
        {"name": "Tom Hanks", "born": 1956},
    )
    connector.cypher(
        "CREATE (p:Person {name: $name, born: $born})",
        {"name": "Robert Zemeckis", "born": 1951},
    )
    connector.cypher(
        "CREATE (m:Movie {title: $title, released: $released})",
        {"title": "Forrest Gump", "released": 1994},
    )
    connector.cypher(
        "CREATE (m:Movie {title: $title, released: $released})",
        {"title": "Cast Away", "released": 2000},
    )
    connector.cypher(
        """
        MATCH (p:Person {name: 'Tom Hanks'}), (m:Movie {title: 'Forrest Gump'})
        CREATE (p)-[:ACTED_IN {roles: ['Forrest']}]->(m)
        """
    )
    connector.cypher(
        """
        MATCH (p:Person {name: 'Tom Hanks'}), (m:Movie {title: 'Cast Away'})
        CREATE (p)-[:ACTED_IN {roles: ['Chuck Noland']}]->(m)
        """
    )
    connector.cypher(
        """
        MATCH (p:Person {name: 'Robert Zemeckis'}), (m:Movie {title: 'Forrest Gump'})
        CREATE (p)-[:DIRECTED]->(m)
        """
    )
    connector.cypher(
        """
        MATCH (p:Person {name: 'Robert Zemeckis'}), (m:Movie {title: 'Cast Away'})
        CREATE (p)-[:DIRECTED]->(m)
        """
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def neo4j_container(docker_available) -> Generator[dict, None, None]:
    """Start Neo4j Docker container for the test session.

    Yields connection info dict with keys: host, bolt_port, uri, username, password.
    Calls pytest.fail() (NOT pytest.skip()) if Docker is unavailable.
    """
    if not docker_available:
        pytest.fail("Docker not available — install Docker to run Neo4j integration tests")

    container_name = get_unique_container_name("constat_test_neo4j")
    bolt_port = get_unique_port(_NEO4J_DEFAULT_PORT)
    http_port = get_unique_port(_NEO4J_HTTP_PORT)

    if not start_container(
        name=container_name,
        image="neo4j:5",
        port_mapping=f"{bolt_port}:{_NEO4J_DEFAULT_PORT}",
        env={
            "NEO4J_AUTH": f"{_NEO4J_USER}/{_NEO4J_PASSWORD}",
            "NEO4J_ACCEPT_LICENSE_AGREEMENT": "yes",
            # Disable plugins and extra services to speed up startup
            "NEO4J_dbms_memory_heap_initial__size": "256m",
            "NEO4J_dbms_memory_heap_max__size": "512m",
        },
        wait_seconds=15,
    ):
        pytest.fail("Failed to start Neo4j container")

    if not _wait_for_neo4j(bolt_port, timeout=120):
        stop_container(container_name)
        pytest.fail("Neo4j container failed to become ready within 120s")

    yield {
        "host": "localhost",
        "bolt_port": bolt_port,
        "uri": f"bolt://localhost:{bolt_port}",
        "username": _NEO4J_USER,
        "password": _NEO4J_PASSWORD,
        "container_name": container_name,
    }

    stop_container(container_name)


@pytest.fixture(scope="session")
def neo4j_connector(neo4j_container) -> Generator[Neo4jConnector, None, None]:
    """A connected Neo4jConnector with movies data pre-loaded."""
    connector = Neo4jConnector(
        uri=neo4j_container["uri"],
        database="neo4j",
        username=neo4j_container["username"],
        password=neo4j_container["password"],
        name="movies_test",
    )
    connector.connect()
    _load_movies_data(connector)
    yield connector
    connector.disconnect()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestNeo4jConnect:
    """Test that the connector can establish and verify a real connection."""

    def test_connect_and_verify(self, neo4j_container) -> None:
        connector = Neo4jConnector(
            uri=neo4j_container["uri"],
            username=neo4j_container["username"],
            password=neo4j_container["password"],
            name="verify_test",
        )
        connector.connect()
        assert connector.is_connected
        connector.disconnect()
        assert not connector.is_connected

    def test_connect_wrong_password_raises(self, neo4j_container) -> None:
        connector = Neo4jConnector(
            uri=neo4j_container["uri"],
            username=neo4j_container["username"],
            password="wrong_password",
            name="bad_auth_test",
        )
        with pytest.raises(Exception):
            connector.connect()


class TestNeo4jQueryNodes:
    """Test querying nodes by label and property filters."""

    def test_query_all_persons(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("Person", {}, limit=10)
        assert len(results) == 2, f"Expected 2 Person nodes, got {len(results)}"

    def test_query_person_by_name(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("Person", {"name": "Tom Hanks"}, limit=10)
        assert len(results) == 1
        node = results[0]
        # Result is a dict with "n" key (the node)
        if "n" in node:
            assert node["n"].get("name") == "Tom Hanks" or "Tom Hanks" in str(node)
        else:
            assert any("Tom Hanks" in str(v) for v in node.values()), f"Tom Hanks not in result: {node}"

    def test_query_movies_returns_expected_count(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("Movie", {}, limit=10)
        assert len(results) == 2, f"Expected 2 Movie nodes, got {len(results)}"

    def test_query_with_limit(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("Person", {}, limit=1)
        assert len(results) == 1


class TestNeo4jRelationships:
    """Test querying relationship types."""

    def test_query_acted_in_relationships(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("rel:ACTED_IN", {}, limit=10)
        assert len(results) == 2, f"Expected 2 ACTED_IN relationships, got {len(results)}"

    def test_relationship_results_have_source_target_labels(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("rel:ACTED_IN", {}, limit=10)
        assert len(results) > 0
        row = results[0]
        assert "_source_label" in row, f"Missing _source_label in: {row}"
        assert "_target_label" in row, f"Missing _target_label in: {row}"
        assert row["_source_label"] == "Person"
        assert row["_target_label"] == "Movie"

    def test_query_directed_relationships(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.query("rel:DIRECTED", {}, limit=10)
        assert len(results) == 2, f"Expected 2 DIRECTED relationships, got {len(results)}"


class TestNeo4jCypher:
    """Test raw Cypher execution."""

    def test_cypher_match_nodes(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.cypher(
            "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name"
        )
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert "Robert Zemeckis" in names
        assert "Tom Hanks" in names

    def test_cypher_traversal(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.cypher(
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) "
            "RETURN p.name AS actor, m.title AS movie ORDER BY m.title"
        )
        assert len(results) == 2
        titles = [r["movie"] for r in results]
        assert "Cast Away" in titles
        assert "Forrest Gump" in titles

    def test_cypher_with_params(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.cypher(
            "MATCH (m:Movie) WHERE m.released >= $year RETURN m.title AS title ORDER BY m.title",
            {"year": 2000},
        )
        assert len(results) == 1
        assert results[0]["title"] == "Cast Away"

    def test_cypher_count(self, neo4j_connector: Neo4jConnector) -> None:
        results = neo4j_connector.cypher("MATCH (n) RETURN count(n) AS total")
        assert len(results) == 1
        assert results[0]["total"] == 4  # 2 Person + 2 Movie


class TestNeo4jSchemaDiscovery:
    """Test schema and collection discovery against a live graph."""

    def test_get_collections_includes_labels(self, neo4j_connector: Neo4jConnector) -> None:
        collections = neo4j_connector.get_collections()
        assert "Movie" in collections, f"Expected 'Movie' in collections: {collections}"
        assert "Person" in collections, f"Expected 'Person' in collections: {collections}"

    def test_get_collections_includes_relationship_types(self, neo4j_connector: Neo4jConnector) -> None:
        collections = neo4j_connector.get_collections()
        assert "rel:ACTED_IN" in collections, f"Expected 'rel:ACTED_IN' in collections: {collections}"
        assert "rel:DIRECTED" in collections, f"Expected 'rel:DIRECTED' in collections: {collections}"

    def test_graph_schema_labels(self, neo4j_connector: Neo4jConnector) -> None:
        schema = neo4j_connector.graph_schema()
        assert "Movie" in schema["labels"]
        assert "Person" in schema["labels"]

    def test_graph_schema_relationship_types(self, neo4j_connector: Neo4jConnector) -> None:
        schema = neo4j_connector.graph_schema()
        assert "ACTED_IN" in schema["relationship_types"]
        assert "DIRECTED" in schema["relationship_types"]

    def test_graph_schema_patterns(self, neo4j_connector: Neo4jConnector) -> None:
        schema = neo4j_connector.graph_schema()
        patterns = schema["patterns"]
        assert any("ACTED_IN" in p for p in patterns), f"Expected ACTED_IN pattern in: {patterns}"
        assert any("DIRECTED" in p for p in patterns), f"Expected DIRECTED pattern in: {patterns}"
        # Verify canonical pattern format
        assert "(:Person)-[:ACTED_IN]->(:Movie)" in patterns

    def test_get_node_schema(self, neo4j_connector: Neo4jConnector) -> None:
        schema = neo4j_connector.get_collection_schema("Person")
        field_names = [f.name for f in schema.fields]
        assert "name" in field_names, f"Expected 'name' field in Person schema, got: {field_names}"
        assert "born" in field_names, f"Expected 'born' field in Person schema, got: {field_names}"
        assert schema.document_count == 2

    def test_get_relationship_schema(self, neo4j_connector: Neo4jConnector) -> None:
        schema = neo4j_connector.get_collection_schema("rel:ACTED_IN")
        assert schema.document_count == 2
        assert "Person" in schema.description
        assert "Movie" in schema.description

    def test_schema_is_cached(self, neo4j_connector: Neo4jConnector) -> None:
        # Clear cache to ensure fresh fetch
        neo4j_connector._metadata_cache.pop("Movie", None)
        schema1 = neo4j_connector.get_collection_schema("Movie")
        schema2 = neo4j_connector.get_collection_schema("Movie")
        assert schema1 is schema2
