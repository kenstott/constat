# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Elasticsearch connector for search engine support."""

from typing import Optional

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class ElasticsearchConnector(NoSQLConnector):
    """Connector for Elasticsearch and OpenSearch.

    Works with:
    - Self-hosted Elasticsearch
    - Elastic Cloud
    - Amazon OpenSearch Service
    - Self-hosted OpenSearch

    Usage:
        # Local Elasticsearch
        connector = ElasticsearchConnector(
            hosts=["http://localhost:9200"],
            name="es_main",
        )

        # Elastic Cloud
        connector = ElasticsearchConnector(
            cloud_id="deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJA...",
            api_key="api_key_here",
            name="elastic_cloud",
        )

        # AWS OpenSearch
        connector = ElasticsearchConnector(
            hosts=["https://search-domain.us-east-1.es.amazonaws.com"],
            name="opensearch",
            use_ssl=True,
        )

        connector.connect()
        indices = connector.get_collections()
        mapping = connector.get_collection_schema("products")
    """

    def __init__(
        self,
        hosts: Optional[list[str]] = None,
        name: Optional[str] = None,
        description: str = "",
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        basic_auth: Optional[tuple[str, str]] = None,
        use_ssl: bool = False,
        verify_certs: bool = True,
        sample_size: int = 100,
    ):
        """
        Initialize Elasticsearch connector.

        Args:
            hosts: List of Elasticsearch hosts
            name: Friendly name for this connection
            description: Description of the cluster
            cloud_id: Elastic Cloud deployment ID
            api_key: API key for authentication
            basic_auth: Tuple of (username, password)
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify SSL certificates
            sample_size: Number of documents to sample for type verification
        """
        super().__init__(name=name or "elasticsearch", description=description)
        self.hosts = hosts or ["http://localhost:9200"]
        self.cloud_id = cloud_id
        self.api_key = api_key
        self.basic_auth = basic_auth
        self.use_ssl = use_ssl
        self.verify_certs = verify_certs
        self.sample_size = sample_size
        self._client = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.SEARCH

    def connect(self) -> None:
        """Connect to Elasticsearch cluster."""
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            raise ImportError(
                "Elasticsearch connector requires elasticsearch. "
                "Install with: pip install elasticsearch"
            )

        if self.cloud_id:
            # Elastic Cloud connection
            self._client = Elasticsearch(
                cloud_id=self.cloud_id,
                api_key=self.api_key,
            )
        elif self.api_key:
            self._client = Elasticsearch(
                hosts=self.hosts,
                api_key=self.api_key,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
            )
        elif self.basic_auth:
            self._client = Elasticsearch(
                hosts=self.hosts,
                basic_auth=self.basic_auth,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
            )
        else:
            self._client = Elasticsearch(
                hosts=self.hosts,
                use_ssl=self.use_ssl,
                verify_certs=self.verify_certs,
            )

        # Verify connection
        self._client.info()
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Elasticsearch cluster."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False

    def get_collections(self) -> list[str]:
        """List all indices in the cluster."""
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        # Get all indices, excluding system indices
        indices = self._client.indices.get_alias(index="*")
        return [idx for idx in indices.keys() if not idx.startswith(".")]

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get mapping for an index."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        # Get index mapping
        mapping = self._client.indices.get_mapping(index=collection)
        properties = mapping[collection]["mappings"].get("properties", {})

        fields = self._parse_mapping_properties(properties)

        # Get document count
        count_response = self._client.count(index=collection)
        doc_count = count_response["count"]

        # Get index settings for shard info
        settings = self._client.indices.get_settings(index=collection)
        index_settings = settings[collection]["settings"]["index"]

        metadata = CollectionMetadata(
            name=collection,
            database="elasticsearch",
            nosql_type=self.nosql_type,
            fields=fields,
            document_count=doc_count,
            description=f"Shards: {index_settings.get('number_of_shards', 'N/A')}, "
                       f"Replicas: {index_settings.get('number_of_replicas', 'N/A')}",
        )

        self._metadata_cache[collection] = metadata
        return metadata

    def _parse_mapping_properties(
        self,
        properties: dict,
        prefix: str = "",
    ) -> list[FieldInfo]:
        """Parse Elasticsearch mapping properties into FieldInfo list."""
        fields = []

        for field_name, field_props in properties.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            field_type = field_props.get("type", "object")

            # Handle nested objects
            if "properties" in field_props:
                fields.extend(
                    self._parse_mapping_properties(field_props["properties"], full_name)
                )
            else:
                field = FieldInfo(
                    name=full_name,
                    data_type=field_type,
                    nullable=True,  # ES fields are nullable by default
                    is_indexed=field_props.get("index", True),
                )

                # Check for keyword field (exact match)
                if "fields" in field_props:
                    for sub_field, sub_props in field_props["fields"].items():
                        if sub_props.get("type") == "keyword":
                            field.description = "Has keyword sub-field for exact matching"

                fields.append(field)

        return fields

    def query(
        self,
        collection: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Execute an Elasticsearch query.

        Args:
            collection: Index name
            query: Elasticsearch query DSL
            limit: Maximum documents to return

        Returns:
            List of documents (without metadata)
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        # If query is empty, match all
        if not query:
            query = {"match_all": {}}

        response = self._client.search(
            index=collection,
            query=query,
            size=limit,
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]

    def search(
        self,
        collection: str,
        query_string: str,
        fields: Optional[list[str]] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Execute a simple text search.

        Args:
            collection: Index name
            query_string: Search query text
            fields: Fields to search (default: all)
            limit: Maximum documents to return

        Returns:
            List of documents with scores
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        if fields:
            query = {
                "multi_match": {
                    "query": query_string,
                    "fields": fields,
                }
            }
        else:
            query = {
                "query_string": {
                    "query": query_string,
                }
            }

        response = self._client.search(
            index=collection,
            query=query,
            size=limit,
        )

        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"].copy()
            doc["_score"] = hit["_score"]
            doc["_id"] = hit["_id"]
            results.append(doc)

        return results

    def aggregate(
        self,
        collection: str,
        aggs: dict,
        query: Optional[dict] = None,
    ) -> dict:
        """Execute an aggregation query.

        Args:
            collection: Index name
            aggs: Aggregation definition
            query: Optional filter query

        Returns:
            Aggregation results
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        body = {"aggs": aggs, "size": 0}
        if query:
            body["query"] = query

        response = self._client.search(index=collection, **body)
        return response["aggregations"]

    def insert(self, collection: str, documents: list[dict]) -> list[str]:
        """Index documents.

        Args:
            collection: Index name
            documents: Documents to index

        Returns:
            List of document IDs
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        ids = []
        for doc in documents:
            # Use _id from doc if present, otherwise auto-generate
            doc_id = doc.pop("_id", None)
            if doc_id:
                response = self._client.index(index=collection, id=doc_id, document=doc)
            else:
                response = self._client.index(index=collection, document=doc)
            ids.append(response["_id"])

        return ids

    def bulk_insert(self, collection: str, documents: list[dict]) -> dict:
        """Bulk index documents for better performance.

        Args:
            collection: Index name
            documents: Documents to index

        Returns:
            Bulk response summary
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": collection,
                "_source": doc,
            }
            for doc in documents
        ]

        success, errors = bulk(self._client, actions)
        return {"success": success, "errors": errors}

    def delete(self, collection: str, query: dict) -> int:
        """Delete documents matching a query.

        Args:
            collection: Index name
            query: Query to match documents for deletion

        Returns:
            Number of deleted documents
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        response = self._client.delete_by_query(
            index=collection,
            query=query,
        )
        return response["deleted"]

    def create_index(
        self,
        collection: str,
        mappings: Optional[dict] = None,
        settings: Optional[dict] = None,
    ) -> None:
        """Create a new index.

        Args:
            collection: Index name
            mappings: Field mappings
            settings: Index settings (shards, replicas, etc.)
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        body = {}
        if mappings:
            body["mappings"] = mappings
        if settings:
            body["settings"] = settings

        self._client.indices.create(index=collection, body=body if body else None)

    def delete_index(self, collection: str) -> None:
        """Delete an index.

        Args:
            collection: Index name
        """
        if not self._client:
            raise RuntimeError("Not connected to Elasticsearch")

        self._client.indices.delete(index=collection)
