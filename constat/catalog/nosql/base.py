# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Base class for NoSQL database connectors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class NoSQLType(Enum):
    """Type of NoSQL database."""
    DOCUMENT = "document"  # MongoDB, CouchDB
    WIDE_COLUMN = "wide_column"  # Cassandra, HBase
    KEY_VALUE = "key_value"  # Redis, DynamoDB
    GRAPH = "graph"  # Neo4j, Amazon Neptune
    SEARCH = "search"  # Elasticsearch, OpenSearch
    TIME_SERIES = "time_series"  # InfluxDB, TimescaleDB


@dataclass
class FieldInfo:
    """Information about a field in a collection/table."""
    name: str
    data_type: str  # Inferred type from samples
    nullable: bool = True
    is_indexed: bool = False
    is_unique: bool = False
    sample_values: list[Any] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class CollectionMetadata:
    """Metadata for a NoSQL collection/table/index."""
    name: str
    database: str
    nosql_type: NoSQLType
    fields: list[FieldInfo]
    document_count: int = 0
    size_bytes: int = 0
    indexes: list[str] = field(default_factory=list)
    description: Optional[str] = None

    # Type-specific metadata
    partition_key: Optional[str] = None  # Cassandra
    clustering_keys: list[str] = field(default_factory=list)  # Cassandra
    shard_key: Optional[str] = None  # MongoDB

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "database": self.database,
            "type": self.nosql_type.value,
            "fields": [
                {
                    "name": f.name,
                    "type": f.data_type,
                    "nullable": f.nullable,
                    "indexed": f.is_indexed,
                }
                for f in self.fields
            ],
            "document_count": self.document_count,
            "indexes": self.indexes,
            "description": self.description,
        }

    def to_embedding_text(self) -> str:
        """Generate text for vector embedding."""
        lines = [
            f"Collection: {self.database}.{self.name}",
            f"Type: {self.nosql_type.value}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")

        lines.append(f"Fields: {', '.join(f.name for f in self.fields)}")

        if self.indexes:
            lines.append(f"Indexes: {', '.join(self.indexes)}")

        return "\n".join(lines)


class NoSQLConnector(ABC):
    """Abstract base class for NoSQL database connectors.

    Subclasses must implement:
    - connect(): Establish connection
    - disconnect(): Close connection
    - get_collections(): List all collections/tables
    - get_collection_schema(): Get schema for a collection
    - query(): Execute a query
    - get_overview(): Token-optimized summary

    Optionally override:
    - sample_documents(): Get sample documents for schema inference
    - create_index(): Create an index
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._connected = False
        self._metadata_cache: dict[str, CollectionMetadata] = {}

    @property
    @abstractmethod
    def nosql_type(self) -> NoSQLType:
        """Return the type of NoSQL database."""
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the database."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @abstractmethod
    def get_collections(self) -> list[str]:
        """List all collections/tables in the database."""
        pass

    @abstractmethod
    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema/structure for a collection.

        For schema-less databases (MongoDB, Elasticsearch), this
        samples documents to infer the schema.
        """
        pass

    @abstractmethod
    def query(self, collection: str, query: dict, limit: int = 100) -> list[dict]:
        """Execute a query against a collection.

        Args:
            collection: Collection/table name
            query: Query in database-native format
            limit: Maximum documents to return

        Returns:
            List of documents/rows
        """
        pass

    def get_overview(self) -> str:
        """Generate token-optimized overview for system prompt."""
        collections = self.get_collections()
        total_docs = 0

        lines = [f"  {self.name} ({self.nosql_type.value}): "]

        collection_summaries = []
        for coll_name in collections[:10]:  # Limit for token efficiency
            try:
                meta = self.get_collection_schema(coll_name)
                total_docs += meta.document_count
                field_names = [f.name for f in meta.fields[:5]]
                fields_str = ", ".join(field_names)
                if len(meta.fields) > 5:
                    fields_str += f" (+{len(meta.fields) - 5} more)"
                collection_summaries.append(f"{coll_name} [{fields_str}]")
            except Exception:
                collection_summaries.append(coll_name)

        lines[0] += ", ".join(collection_summaries)
        if len(collections) > 10:
            lines[0] += f" (+{len(collections) - 10} more collections)"

        lines.append(f"    ~{total_docs:,} total documents")

        return "\n".join(lines)

    def sample_documents(self, collection: str, limit: int = 10) -> list[dict]:
        """Get sample documents for schema inference.

        Default implementation uses query with empty filter.
        Override for databases with specific sampling methods.
        """
        return self.query(collection, {}, limit=limit)

    @staticmethod
    def infer_field_type(values: list[Any]) -> str:
        """Infer field type from sample values."""
        types_seen = set()
        for val in values:
            if val is None:
                continue
            elif isinstance(val, bool):
                types_seen.add("boolean")
            elif isinstance(val, int):
                types_seen.add("integer")
            elif isinstance(val, float):
                types_seen.add("float")
            elif isinstance(val, str):
                types_seen.add("string")
            elif isinstance(val, list):
                types_seen.add("array")
            elif isinstance(val, dict):
                types_seen.add("object")
            else:
                types_seen.add("unknown")

        if not types_seen:
            return "null"
        elif len(types_seen) == 1:
            return types_seen.pop()
        else:
            return "mixed"

    def infer_schema_from_samples(
        self,
        collection: str,
        samples: list[dict],
    ) -> CollectionMetadata:
        """Infer schema from sample documents."""
        field_values: dict[str, list[Any]] = {}

        for doc in samples:
            self._extract_fields(doc, "", field_values)

        fields = []
        for field_path, values in sorted(field_values.items()):
            non_null_values = [v for v in values if v is not None]
            fields.append(FieldInfo(
                name=field_path,
                data_type=self.infer_field_type(values),
                nullable=len(non_null_values) < len(values),
                sample_values=non_null_values[:3],
            ))

        return CollectionMetadata(
            name=collection,
            database=self.name,
            nosql_type=self.nosql_type,
            fields=fields,
            document_count=len(samples),  # Will be updated with actual count
        )

    def _extract_fields(
        self,
        doc: dict,
        prefix: str,
        field_values: dict[str, list[Any]],
    ) -> None:
        """Extract field paths and values from a document."""
        for key, value in doc.items():
            path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Nested object - recurse
                self._extract_fields(value, path, field_values)
            else:
                if path not in field_values:
                    field_values[path] = []
                field_values[path].append(value)

    def get_all_metadata(self) -> list[CollectionMetadata]:
        """Get metadata for all collections."""
        return [
            self.get_collection_schema(coll)
            for coll in self.get_collections()
        ]
