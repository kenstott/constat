# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Azure Cosmos DB connector for multi-model database support."""

from typing import Optional, Any

from .base import NoSQLConnector, NoSQLType, CollectionMetadata


class CosmosDBConnector(NoSQLConnector):
    """Connector for Azure Cosmos DB (SQL/Core API).

    Cosmos DB supports multiple APIs (SQL, MongoDB, Cassandra, Gremlin, Table).
    This connector uses the SQL (Core) API which is the native Cosmos DB API.

    For MongoDB API, use the MongoDBConnector with Cosmos DB connection string.
    For Cassandra API, use the CassandraConnector with Cosmos DB endpoint.

    Usage:
        connector = CosmosDBConnector(
            endpoint="https://myaccount.documents.azure.com:443/",
            key="your_primary_key",
            database="mydb",
            name="cosmos_main",
        )

        # Or using connection string
        connector = CosmosDBConnector(
            connection_string="AccountEndpoint=https://...;AccountKey=...",
            database="mydb",
            name="cosmos_main",
        )

        connector.connect()
        containers = connector.get_collections()
        schema = connector.get_collection_schema("users")
    """

    def __init__(
        self,
        database: str,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        connection_string: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        sample_size: int = 100,
    ):
        """
        Initialize Cosmos DB connector.

        Args:
            database: Database name
            endpoint: Cosmos DB endpoint URL
            key: Primary or secondary key
            connection_string: Full connection string (alternative to endpoint/key)
            name: Friendly name for this connection
            description: Description of the database
            sample_size: Number of documents to sample for schema inference
        """
        super().__init__(name=name or database, description=description)
        self.database_name = database
        self.endpoint = endpoint
        self.key = key
        self.connection_string = connection_string
        self.sample_size = sample_size
        self._client = None
        self._db = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.DOCUMENT

    def connect(self) -> None:
        """Connect to Cosmos DB."""
        try:
            # noinspection PyUnresolvedReferences
            from azure.cosmos import CosmosClient
        except ImportError:
            raise ImportError(
                "Cosmos DB connector requires azure-cosmos. "
                "Install with: pip install azure-cosmos"
            )

        if self.connection_string:
            self._client = CosmosClient.from_connection_string(self.connection_string)
        elif self.endpoint and self.key:
            self._client = CosmosClient(self.endpoint, credential=self.key)
        else:
            raise ValueError(
                "Either connection_string or both endpoint and key must be provided"
            )

        self._db = self._client.get_database_client(self.database_name)
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Cosmos DB."""
        self._client = None
        self._db = None
        self._connected = False

    def get_collections(self) -> list[str]:
        """List all containers in the database."""
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        containers = self._db.list_containers()
        return [container["id"] for container in containers]

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a container by sampling documents."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)

        # Get container properties
        props = container.read()
        partition_key_paths = props.get("partitionKey", {}).get("paths", [])
        partition_key = partition_key_paths[0].lstrip("/") if partition_key_paths else None

        # Sample documents
        query = f"SELECT TOP {self.sample_size} * FROM c"
        samples = list(container.query_items(query=query, enable_cross_partition_query=True))

        # Infer schema
        metadata = self.infer_schema_from_samples(collection, samples)

        # Get approximate count (Cosmos DB doesn't have exact count without scanning)
        count_query = "SELECT VALUE COUNT(1) FROM c"
        try:
            count_result = list(container.query_items(
                query=count_query,
                enable_cross_partition_query=True,
            ))
            metadata.document_count = count_result[0] if count_result else 0
        except Exception:
            metadata.document_count = len(samples)

        # Set partition key
        metadata.partition_key = partition_key

        # Mark partition key as indexed
        if partition_key:
            for field in metadata.fields:
                if field.name == partition_key:
                    field.is_indexed = True

        # Get indexing policy
        indexing_policy = props.get("indexingPolicy", {})
        included_paths = indexing_policy.get("includedPaths", [])

        indexes = []
        for path in included_paths:
            path_str = path.get("path", "")
            if path_str != "/*":
                indexes.append(path_str)

        metadata.indexes = indexes

        self._metadata_cache[collection] = metadata
        return metadata

    def query(
        self,
        collection: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Execute a query using Cosmos DB SQL syntax.

        Args:
            collection: Container name
            query: Query as dict with optional "sql" key for raw SQL,
                   or key-value pairs for simple equality filters
            limit: Maximum documents to return

        Returns:
            List of documents
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)

        # Build query
        if "sql" in query:
            # Raw SQL query
            sql = query["sql"]
            parameters = query.get("parameters", [])
        elif query:
            # Build WHERE clause from dict
            conditions = []
            parameters = []
            for i, (key, value) in enumerate(query.items()):
                param_name = f"@p{i}"
                conditions.append(f"c.{key} = {param_name}")
                parameters.append({"name": param_name, "value": value})

            where_clause = " AND ".join(conditions)
            sql = f"SELECT TOP {limit} * FROM c WHERE {where_clause}"
        else:
            sql = f"SELECT TOP {limit} * FROM c"
            parameters = []

        results = list(container.query_items(
            query=sql,
            parameters=parameters if parameters else None,
            enable_cross_partition_query=True,
        ))

        # Remove system properties for cleaner output
        cleaned = []
        for doc in results:
            cleaned_doc = {
                k: v for k, v in doc.items()
                if not k.startswith("_")
            }
            cleaned.append(cleaned_doc)

        return cleaned

    def query_sql(
        self,
        collection: str,
        sql: str,
        parameters: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Execute a raw SQL query.

        Args:
            collection: Container name
            sql: SQL query string
            parameters: Query parameters as list of {"name": "@param", "value": val}

        Returns:
            Query results
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)

        results = list(container.query_items(
            query=sql,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))

        return results

    def insert(self, collection: str, documents: list[dict]) -> list[str]:
        """Insert documents into a container.

        Args:
            collection: Container name
            documents: Documents to insert

        Returns:
            List of document IDs
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)
        ids = []

        for doc in documents:
            # Cosmos DB requires an 'id' field
            if "id" not in doc:
                import uuid
                doc["id"] = str(uuid.uuid4())

            result = container.create_item(body=doc)
            ids.append(result["id"])

        return ids

    def upsert(self, collection: str, documents: list[dict]) -> list[str]:
        """Upsert documents (insert or update).

        Args:
            collection: Container name
            documents: Documents to upsert

        Returns:
            List of document IDs
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)
        ids = []

        for doc in documents:
            if "id" not in doc:
                import uuid
                doc["id"] = str(uuid.uuid4())

            result = container.upsert_item(body=doc)
            ids.append(result["id"])

        return ids

    def update(
        self,
        collection: str,
        doc_id: str,
        partition_key: Any,
        updates: dict,
    ) -> dict:
        """Update a document.

        Args:
            collection: Container name
            doc_id: Document ID
            partition_key: Partition key value
            updates: Fields to update

        Returns:
            Updated document
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)

        # Read current document
        doc = container.read_item(item=doc_id, partition_key=partition_key)

        # Apply updates
        doc.update(updates)

        # Replace document
        result = container.replace_item(item=doc_id, body=doc)
        return result

    def delete(self, collection: str, query: dict) -> int:
        """Delete a document.

        Args:
            collection: Container name
            query: Must contain 'id' and partition key value

        Returns:
            Number of deleted documents (always 1)
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        container = self._db.get_container_client(collection)

        doc_id = query.get("id")
        partition_key = query.get("partition_key")

        if not doc_id:
            raise ValueError("Query must contain 'id' field")

        container.delete_item(item=doc_id, partition_key=partition_key)
        return 1

    def create_container(
        self,
        collection: str,
        partition_key: str,
        throughput: Optional[int] = None,
    ) -> None:
        """Create a new container.

        Args:
            collection: Container name
            partition_key: Partition key path (e.g., "/userId")
            throughput: Provisioned throughput (RU/s), None for serverless
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        # noinspection PyUnresolvedReferences
        from azure.cosmos import PartitionKey

        container_kwargs = {
            "id": collection,
            "partition_key": PartitionKey(path=partition_key),
        }

        if throughput:
            container_kwargs["offer_throughput"] = throughput

        self._db.create_container(**container_kwargs)

    def delete_container(self, collection: str) -> None:
        """Delete a container.

        Args:
            collection: Container name
        """
        if not self._db:
            raise RuntimeError("Not connected to Cosmos DB")

        self._db.delete_container(collection)

    def get_overview(self) -> str:
        """Generate token-optimized overview for system prompt."""
        collections = self.get_collections()

        lines = [f"  {self.name} (Cosmos DB): "]

        collection_summaries = []
        total_docs = 0

        for coll_name in collections[:10]:
            try:
                meta = self.get_collection_schema(coll_name)
                total_docs += meta.document_count
                field_names = [f.name for f in meta.fields[:5]]
                fields_str = ", ".join(field_names)
                if len(meta.fields) > 5:
                    fields_str += f" (+{len(meta.fields) - 5} more)"

                pk_info = f" PK:{meta.partition_key}" if meta.partition_key else ""
                collection_summaries.append(f"{coll_name}{pk_info} [{fields_str}]")
            except Exception:
                collection_summaries.append(coll_name)

        lines[0] += ", ".join(collection_summaries)
        if len(collections) > 10:
            lines[0] += f" (+{len(collections) - 10} more containers)"

        lines.append(f"    ~{total_docs:,} total documents")

        return "\n".join(lines)
