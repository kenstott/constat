# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Neo4j graph database connector."""

from typing import Any, Optional

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class Neo4jConnector(NoSQLConnector):
    """Connector for Neo4j graph databases.

    Node labels and relationship types are exposed as "collections":
    - Node labels appear as plain strings (e.g., "Person", "Movie")
    - Relationship types are prefixed with "rel:" (e.g., "rel:ACTED_IN")

    Usage:
        connector = Neo4jConnector(
            uri="bolt://localhost:7687",
            database="neo4j",
            name="movies",
        )
        connector.connect()

        # List node labels and relationship types.
        collections = connector.get_collections()

        # Get schema for a node label.
        schema = connector.get_collection_schema("Person")

        # Query nodes by properties.
        results = connector.query("Person", {"name": "Tom Hanks"})

        # Execute raw Cypher.
        results = connector.cypher(
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, m.title"
        )
    """

    REL_PREFIX = "rel:"

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        database: str = "neo4j",
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        sample_size: int = 100,
    ) -> None:
        super().__init__(name=name or database, description=description)
        self.uri = uri
        self.database_name = database
        self.username = username
        self.password = password
        self.sample_size = sample_size
        self._driver: Any = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.GRAPH

    def connect(self) -> None:
        """Connect to Neo4j."""
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError(
                "Neo4j connector requires the neo4j driver. "
                "Install with: pip install neo4j"
            )

        auth = None
        if self.username and self.password:
            auth = (self.username, self.password)

        self._driver = GraphDatabase.driver(self.uri, auth=auth)
        self._driver.verify_connectivity()
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Neo4j."""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._connected = False

    def _run(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """Execute a Cypher query and return results as dicts."""
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        with self._driver.session(database=self.database_name) as session:
            result = session.run(query, parameters=params or {})
            return [dict(record) for record in result]

    def get_collections(self) -> list[str]:
        """List node labels and relationship types.

        Node labels are returned as plain strings.
        Relationship types are prefixed with "rel:" (e.g., "rel:ACTED_IN").
        """
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        # Node labels
        labels = [
            row["label"]
            for row in self._run("CALL db.labels() YIELD label RETURN label ORDER BY label")
        ]

        # Relationship types
        rel_types = [
            f"{self.REL_PREFIX}{row['relationshipType']}"
            for row in self._run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            )
        ]

        return labels + rel_types

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a node label or relationship type."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        if collection.startswith(self.REL_PREFIX):
            metadata = self._get_relationship_schema(collection)
        else:
            metadata = self._get_node_schema(collection)

        self._metadata_cache[collection] = metadata
        return metadata

    def _get_node_schema(self, label: str) -> CollectionMetadata:
        """Get schema for a node label by sampling."""
        # Sample nodes
        samples = self._run(
            f"MATCH (n:`{label}`) RETURN n LIMIT $limit",
            {"limit": self.sample_size},
        )
        docs = [self._node_to_dict(row["n"]) for row in samples]

        # Infer schema from samples
        metadata = self.infer_schema_from_samples(label, docs)

        # Get actual count
        count_result = self._run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        metadata.document_count = count_result[0]["cnt"] if count_result else 0

        # Get indexes and constraints
        indexes = []
        constraints = []
        try:
            for row in self._run("SHOW INDEXES YIELD name, labelsOrTypes, properties "
                                 "WHERE $label IN labelsOrTypes "
                                 "RETURN name, properties", {"label": label}):
                props = row.get("properties", [])
                indexes.append(f"{row['name']}({', '.join(props)})")
                # Mark indexed fields
                for prop in props:
                    for f in metadata.fields:
                        if f.name == prop:
                            f.is_indexed = True
        except Exception:
            pass

        try:
            for row in self._run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
                                 "WHERE $label IN labelsOrTypes "
                                 "RETURN name, properties", {"label": label}):
                props = row.get("properties", [])
                constraints.append(f"{row['name']}({', '.join(props)})")
                for prop in props:
                    for f in metadata.fields:
                        if f.name == prop:
                            f.is_unique = True
        except Exception:
            pass

        metadata.indexes = indexes + constraints

        # Build description with relationship context
        rel_summary = self._get_relationship_summary(label)
        desc_parts = []
        if metadata.document_count:
            desc_parts.append(f"Node label with {metadata.document_count:,} nodes.")
        if rel_summary:
            desc_parts.append(rel_summary)
        metadata.description = " ".join(desc_parts) if desc_parts else None

        return metadata

    def _get_relationship_summary(self, label: str) -> str:
        """Get outgoing/incoming relationship summary for a node label."""
        parts = []

        # Outgoing
        outgoing = self._run(
            f"MATCH (:`{label}`)-[r]->(t) "
            "RETURN type(r) AS rel_type, labels(t)[0] AS target "
            "WITH DISTINCT rel_type, target "
            "RETURN rel_type, target ORDER BY rel_type",
        )
        if outgoing:
            out_strs = [f"{r['rel_type']}\u2192{r['target']}" for r in outgoing]
            parts.append(f"Outgoing: {', '.join(out_strs)}.")

        # Incoming
        incoming = self._run(
            f"MATCH (s)-[r]->(:`{label}`) "
            "RETURN type(r) AS rel_type, labels(s)[0] AS source "
            "WITH DISTINCT rel_type, source "
            "RETURN rel_type, source ORDER BY rel_type",
        )
        if incoming:
            in_strs = [f"{r['rel_type']}\u2190{r['source']}" for r in incoming]
            parts.append(f"Incoming: {', '.join(in_strs)}.")

        return " ".join(parts)

    def _get_relationship_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a relationship type by sampling."""
        rel_type = collection[len(self.REL_PREFIX):]

        # Sample relationships with endpoint labels
        samples = self._run(
            f"MATCH (s)-[r:`{rel_type}`]->(t) "
            "RETURN r, labels(s)[0] AS source_label, labels(t)[0] AS target_label "
            "LIMIT $limit",
            {"limit": self.sample_size},
        )

        docs = []
        source_labels: set[str] = set()
        target_labels: set[str] = set()
        for row in samples:
            doc = self._rel_to_dict(row["r"])
            docs.append(doc)
            source_labels.add(row["source_label"])
            target_labels.add(row["target_label"])

        # Infer schema from relationship properties
        if docs:
            metadata = self.infer_schema_from_samples(collection, docs)
        else:
            metadata = CollectionMetadata(
                name=collection,
                database=self.name,
                nosql_type=NoSQLType.GRAPH,
                fields=[],
            )

        # Get count
        count_result = self._run(
            f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS cnt"
        )
        metadata.document_count = count_result[0]["cnt"] if count_result else 0

        # Build description with pattern
        patterns = []
        for src in sorted(source_labels):
            for tgt in sorted(target_labels):
                patterns.append(f"(:{src})-[:{rel_type}]->(:{tgt})")
        desc = f"Relationship {', '.join(patterns)}"
        if metadata.document_count:
            desc += f", {metadata.document_count:,} relationships"
        metadata.description = desc

        return metadata

    def query(self, collection: str, query_dict: dict, limit: int = 100) -> list[dict]:
        """Query nodes or relationships by property match.

        Args:
            collection: Node label or "rel:TYPE" relationship type
            query_dict: Property key-value pairs for filtering
            limit: Maximum results to return

        Returns:
            List of result dicts
        """
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        if collection.startswith(self.REL_PREFIX):
            rel_type = collection[len(self.REL_PREFIX):]
            return self._query_relationships(rel_type, query_dict, limit)

        return self._query_nodes(collection, query_dict, limit)

    def _query_nodes(self, label: str, query_dict: dict, limit: int) -> list[dict]:
        """Query nodes by property match."""
        params: dict[str, Any] = {"limit": limit}
        where_clauses = []
        for i, (key, value) in enumerate(query_dict.items()):
            param_name = f"p{i}"
            where_clauses.append(f"n.`{key}` = ${param_name}")
            params[param_name] = value

        where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher_query = f"MATCH (n:`{label}`){where} RETURN n LIMIT $limit"

        results = self._run(cypher_query, params)
        return [self._convert_record(row) for row in results]

    def _query_relationships(
        self, rel_type: str, query_dict: dict, limit: int
    ) -> list[dict]:
        """Query relationships by property match."""
        params: dict[str, Any] = {"limit": limit}
        where_clauses = []
        for i, (key, value) in enumerate(query_dict.items()):
            param_name = f"p{i}"
            where_clauses.append(f"r.`{key}` = ${param_name}")
            params[param_name] = value

        where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        cypher_query = (
            f"MATCH (s)-[r:`{rel_type}`]->(t){where} "
            "RETURN r, labels(s)[0] AS source_label, labels(t)[0] AS target_label "
            "LIMIT $limit"
        )

        results = self._run(cypher_query, params)
        converted = []
        for row in results:
            rec = self._rel_to_dict(row["r"])
            rec["_source_label"] = row["source_label"]
            rec["_target_label"] = row["target_label"]
            converted.append(rec)
        return converted

    def cypher(self, query: str, params: Optional[dict] = None) -> list[dict]:
        """Execute a raw Cypher query.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            List of result dicts
        """
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        results = self._run(query, params)
        return [self._convert_record(row) for row in results]

    def graph_schema(self) -> dict[str, Any]:
        """Return graph schema with labels, relationship types, and patterns.

        Returns:
            Dict with keys: labels, relationship_types, patterns
        """
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        labels = [
            row["label"]
            for row in self._run("CALL db.labels() YIELD label RETURN label ORDER BY label")
        ]

        rel_types = [
            row["relationshipType"]
            for row in self._run(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            )
        ]

        # Discover actual patterns
        patterns = []
        for row in self._run(
            "MATCH (s)-[r]->(t) "
            "RETURN DISTINCT labels(s)[0] AS src, type(r) AS rel, labels(t)[0] AS tgt "
            "ORDER BY src, rel, tgt"
        ):
            patterns.append(f"(:{row['src']})-[:{row['rel']}]->(:{row['tgt']})")

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "patterns": patterns,
        }

    def get_overview(self) -> str:
        """Generate compact graph overview for system prompts.

        Format:
            Graph 'movies' — Movie database (connection: db_movies, query: db_movies.cypher())
              Person(name, born) ~1,000 nodes
              Movie(title, released, tagline) ~500 nodes

            Graph patterns:
              (:Person)-[:ACTED_IN {roles}]->(:Movie) ~2,000
              (:Person)-[:DIRECTED]->(:Movie) ~200
        """
        if self._driver is None:
            raise RuntimeError("Not connected to Neo4j")

        desc_part = f" \u2014 {self.description}" if self.description else ""
        lines = [
            f"  Graph '{self.name}'{desc_part} "
            f"(connection: db_{self.name}, query: db_{self.name}.cypher())"
        ]

        # Node labels with properties and counts
        labels = [
            row["label"]
            for row in self._run("CALL db.labels() YIELD label RETURN label ORDER BY label")
        ]
        for label in labels:
            count_result = self._run(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
            count = count_result[0]["cnt"] if count_result else 0

            # Get property keys from sample
            prop_sample = self._run(
                f"MATCH (n:`{label}`) RETURN keys(n) AS props LIMIT 1"
            )
            props = prop_sample[0]["props"] if prop_sample else []
            props_str = ", ".join(sorted(props))
            lines.append(f"    {label}({props_str}) ~{count:,} nodes")

        # Relationship patterns
        pattern_rows = self._run(
            "MATCH (s)-[r]->(t) "
            "RETURN labels(s)[0] AS src, type(r) AS rel, labels(t)[0] AS tgt, "
            "count(r) AS cnt, "
            "CASE WHEN size(keys(r)) > 0 THEN keys(r)[0] ELSE null END AS sample_key "
            "ORDER BY src, rel, tgt"
        )

        if pattern_rows:
            lines.append("")
            lines.append("    Graph patterns:")
            for row in pattern_rows:
                prop_part = ""
                # Get relationship property keys
                rel_props = self._run(
                    f"MATCH ()-[r:`{row['rel']}`]->() RETURN keys(r) AS props LIMIT 1"
                )
                if rel_props and rel_props[0]["props"]:
                    prop_part = f" {{{', '.join(sorted(rel_props[0]['props']))}}}"
                lines.append(
                    f"      (:{row['src']})-[:{row['rel']}{prop_part}]->(:{row['tgt']}) "
                    f"~{row['cnt']:,}"
                )

        return "\n".join(lines)

    def _node_to_dict(self, node: Any) -> dict:
        """Convert a Neo4j Node to a plain dict."""
        type_name = type(node).__name__
        if type_name == "Node":
            return dict(node)
        return {"_raw": str(node)}

    def _rel_to_dict(self, rel: Any) -> dict:
        """Convert a Neo4j Relationship to a plain dict."""
        type_name = type(rel).__name__
        if type_name == "Relationship":
            return dict(rel)
        return {"_raw": str(rel)}

    def _convert_value(self, value: Any) -> Any:
        """Convert Neo4j types to plain Python types."""
        type_name = type(value).__name__

        if type_name == "Node":
            result = dict(value)
            result["_labels"] = list(value.labels) if hasattr(value, "labels") else []
            return result

        if type_name == "Relationship":
            result = dict(value)
            result["_type"] = value.type if hasattr(value, "type") else str(type(value))
            return result

        if type_name == "Path":
            nodes = []
            if hasattr(value, "nodes"):
                nodes = [self._convert_value(n) for n in value.nodes]
            rels = []
            if hasattr(value, "relationships"):
                rels = [self._convert_value(r) for r in value.relationships]
            return {"_nodes": nodes, "_relationships": rels}

        if isinstance(value, (list, tuple)):
            return [self._convert_value(v) for v in value]

        if isinstance(value, dict):
            return {k: self._convert_value(v) for k, v in value.items()}

        return value

    def _convert_record(self, record: dict) -> dict:
        """Convert all values in a record dict."""
        return {k: self._convert_value(v) for k, v in record.items()}
