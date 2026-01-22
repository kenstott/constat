# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Cassandra connector for wide-column database support."""

from typing import Optional, Any

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class CassandraConnector(NoSQLConnector):
    """Connector for Apache Cassandra and DataStax Astra.

    Usage:
        # Local Cassandra
        connector = CassandraConnector(
            hosts=["localhost"],
            keyspace="my_keyspace",
            name="cassandra_main",
        )

        # DataStax Astra (cloud)
        connector = CassandraConnector(
            keyspace="my_keyspace",
            name="astra_db",
            cloud_config={
                "secure_connect_bundle": "/path/to/secure-connect-bundle.zip"
            },
            auth_provider=("client_id", "client_secret"),
        )

        connector.connect()
        tables = connector.get_collections()
        schema = connector.get_collection_schema("users")
    """

    def __init__(
        self,
        keyspace: str,
        hosts: Optional[list[str]] = None,
        port: int = 9042,
        name: Optional[str] = None,
        description: str = "",
        cloud_config: Optional[dict] = None,
        auth_provider: Optional[tuple[str, str]] = None,
        sample_size: int = 100,
    ):
        """
        Initialize Cassandra connector.

        Args:
            keyspace: Keyspace name
            hosts: List of contact points (for local/on-prem)
            port: CQL port (default 9042)
            name: Friendly name for this connection
            description: Description of the database
            cloud_config: Cloud configuration for DataStax Astra
            auth_provider: Tuple of (username, password) for authentication
            sample_size: Number of rows to sample for type inference
        """
        super().__init__(name=name or keyspace, description=description)
        self.keyspace = keyspace
        self.hosts = hosts or ["localhost"]
        self.port = port
        self.cloud_config = cloud_config
        self.auth_provider = auth_provider
        self.sample_size = sample_size
        self._cluster = None
        self._session = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.WIDE_COLUMN

    def connect(self) -> None:
        """Connect to Cassandra cluster."""
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
        except ImportError:
            raise ImportError(
                "Cassandra connector requires cassandra-driver. "
                "Install with: pip install cassandra-driver"
            )

        auth = None
        if self.auth_provider:
            auth = PlainTextAuthProvider(
                username=self.auth_provider[0],
                password=self.auth_provider[1],
            )

        if self.cloud_config:
            # DataStax Astra cloud connection
            self._cluster = Cluster(
                cloud=self.cloud_config,
                auth_provider=auth,
            )
        else:
            # Local/on-prem connection
            self._cluster = Cluster(
                contact_points=self.hosts,
                port=self.port,
                auth_provider=auth,
            )

        self._session = self._cluster.connect(self.keyspace)
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Cassandra cluster."""
        if self._session:
            self._session.shutdown()
            self._session = None
        if self._cluster:
            self._cluster.shutdown()
            self._cluster = None
        self._connected = False

    def get_collections(self) -> list[str]:
        """List all tables in the keyspace."""
        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        query = """
            SELECT table_name
            FROM system_schema.tables
            WHERE keyspace_name = %s
        """
        rows = self._session.execute(query, [self.keyspace])
        return [row.table_name for row in rows]

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a table."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        # Get column information from system schema
        column_query = """
            SELECT column_name, type, kind, position
            FROM system_schema.columns
            WHERE keyspace_name = %s AND table_name = %s
        """
        columns = list(self._session.execute(column_query, [self.keyspace, collection]))

        fields = []
        partition_keys = []
        clustering_keys = []

        for col in columns:
            field = FieldInfo(
                name=col.column_name,
                data_type=col.type,
                nullable=col.kind == "regular",  # PK and clustering columns are not nullable
            )

            # Track partition and clustering keys
            if col.kind == "partition_key":
                partition_keys.append((col.position, col.column_name))
                field.is_indexed = True
            elif col.kind == "clustering":
                clustering_keys.append((col.position, col.column_name))
                field.is_indexed = True

            fields.append(field)

        # Sort partition and clustering keys by position
        partition_keys.sort()
        clustering_keys.sort()

        # Get table size estimate
        row_count = 0
        try:
            # This is an estimate, not exact count
            count_result = self._session.execute(
                f"SELECT COUNT(*) FROM {self.keyspace}.{collection} LIMIT 10000"
            )
            row_count = count_result.one()[0]
        except Exception:
            pass

        # Get index information
        index_query = """
            SELECT index_name, options
            FROM system_schema.indexes
            WHERE keyspace_name = %s AND table_name = %s
        """
        indexes = []
        try:
            index_rows = self._session.execute(index_query, [self.keyspace, collection])
            for idx in index_rows:
                indexes.append(idx.index_name)
                # Mark indexed fields
                target = idx.options.get("target", "")
                for field in fields:
                    if field.name == target:
                        field.is_indexed = True
        except Exception:
            pass

        metadata = CollectionMetadata(
            name=collection,
            database=self.keyspace,
            nosql_type=self.nosql_type,
            fields=fields,
            document_count=row_count,
            indexes=indexes,
            partition_key=", ".join(k[1] for k in partition_keys) or None,
            clustering_keys=[k[1] for k in clustering_keys],
        )

        self._metadata_cache[collection] = metadata
        return metadata

    def query(
        self,
        collection: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Execute a CQL query.

        Args:
            collection: Table name
            query: Query parameters as dict (key-value filters)
            limit: Maximum rows to return

        Returns:
            List of rows as dicts
        """
        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        # Build WHERE clause from query dict
        if query:
            conditions = []
            values = []
            for key, value in query.items():
                if isinstance(value, dict):
                    # Handle operators like {"$gt": 10}
                    for op, val in value.items():
                        if op == "$gt":
                            conditions.append(f"{key} > %s")
                        elif op == "$gte":
                            conditions.append(f"{key} >= %s")
                        elif op == "$lt":
                            conditions.append(f"{key} < %s")
                        elif op == "$lte":
                            conditions.append(f"{key} <= %s")
                        elif op == "$in":
                            conditions.append(f"{key} IN %s")
                        else:
                            conditions.append(f"{key} = %s")
                        values.append(val)
                else:
                    conditions.append(f"{key} = %s")
                    values.append(value)

            where_clause = " AND ".join(conditions)
            cql = f"SELECT * FROM {self.keyspace}.{collection} WHERE {where_clause} LIMIT {limit}"
            rows = self._session.execute(cql, values)
        else:
            cql = f"SELECT * FROM {self.keyspace}.{collection} LIMIT {limit}"
            rows = self._session.execute(cql)

        return [dict(row._asdict()) for row in rows]

    def execute_cql(self, cql: str, parameters: Optional[list] = None) -> list[dict]:
        """Execute raw CQL query.

        Args:
            cql: CQL query string
            parameters: Query parameters

        Returns:
            Query results as list of dicts
        """
        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        if parameters:
            rows = self._session.execute(cql, parameters)
        else:
            rows = self._session.execute(cql)

        return [dict(row._asdict()) for row in rows]

    def insert(self, collection: str, documents: list[dict]) -> int:
        """Insert rows into a table.

        Args:
            collection: Table name
            documents: Rows to insert

        Returns:
            Number of inserted rows
        """
        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        inserted = 0
        for doc in documents:
            columns = ", ".join(doc.keys())
            placeholders = ", ".join(["%s"] * len(doc))
            cql = f"INSERT INTO {self.keyspace}.{collection} ({columns}) VALUES ({placeholders})"
            self._session.execute(cql, list(doc.values()))
            inserted += 1

        return inserted

    def delete(self, collection: str, query: dict) -> int:
        """Delete rows from a table.

        Note: Cassandra requires the full primary key for DELETE.

        Args:
            collection: Table name
            query: Primary key values

        Returns:
            Number of deleted rows (always 1 for Cassandra)
        """
        if not self._session:
            raise RuntimeError("Not connected to Cassandra")

        conditions = []
        values = []
        for key, value in query.items():
            conditions.append(f"{key} = %s")
            values.append(value)

        where_clause = " AND ".join(conditions)
        cql = f"DELETE FROM {self.keyspace}.{collection} WHERE {where_clause}"
        self._session.execute(cql, values)

        return 1  # Cassandra doesn't return affected row count
