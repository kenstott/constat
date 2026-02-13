# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MongoDB connector for document database support."""

from typing import Optional

from .base import NoSQLConnector, NoSQLType, CollectionMetadata


class MongoDBConnector(NoSQLConnector):
    """Connector for MongoDB document databases.

    Usage:
        connector = MongoDBConnector(
            uri="mongodb://localhost:27017",
            database="mydb",
            name="mongo_main",
        )
        connector.connect()

        # List collections
        collections = connector.get_collections()

        # Get schema (inferred from samples)
        schema = connector.get_collection_schema("users")

        # Query
        results = connector.query("users", {"age": {"$gt": 21}})
    """

    def __init__(
        self,
        uri: str,
        database: str,
        name: Optional[str] = None,
        description: str = "",
        sample_size: int = 100,
    ):
        """
        Initialize MongoDB connector.

        Args:
            uri: MongoDB connection URI
            database: Database name
            name: Friendly name for this connection
            description: Description of the database
            sample_size: Number of documents to sample for schema inference
        """
        super().__init__(name=name or database, description=description)
        self.uri = uri
        self.database_name = database
        self.sample_size = sample_size
        self._client = None
        self._db = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.DOCUMENT

    @property
    def db(self):
        """Get the underlying pymongo database object for direct access."""
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")
        return self._db

    def __getitem__(self, collection_name: str):
        """Allow subscript access like db_ecommerce['customers']."""
        return self.db[collection_name]

    def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "MongoDB connector requires pymongo. "
                "Install with: pip install pymongo"
            )

        self._client = MongoClient(self.uri)
        self._db = self._client[self.database_name]
        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._connected = False

    def get_collections(self) -> list[str]:
        """List all collections in the database."""
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")
        return self._db.list_collection_names()

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a collection by sampling documents."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]

        # Sample documents
        samples = list(coll.find().limit(self.sample_size))

        # Infer schema
        metadata = self.infer_schema_from_samples(collection, samples)

        # Get actual document count
        metadata.document_count = coll.estimated_document_count()

        # Get index information
        metadata.indexes = [
            idx["name"]
            for idx in coll.list_indexes()
            if idx["name"] != "_id_"
        ]

        # Check for shard key (if sharded)
        try:
            stats = self._db.command("collStats", collection)
            if "shards" in stats:
                metadata.shard_key = str(stats.get("shardKey"))
        except Exception:
            pass

        # Mark indexed fields
        for idx_info in coll.list_indexes():
            for key in idx_info.get("key", {}).keys():
                for field in metadata.fields:
                    if field.name == key or field.name.startswith(f"{key}."):
                        field.is_indexed = True
                        if idx_info.get("unique"):
                            field.is_unique = True

        self._metadata_cache[collection] = metadata
        return metadata

    def query(self, collection: str, query: dict, limit: int = 100) -> list[dict]:
        """Execute a MongoDB query.

        Args:
            collection: Collection name
            query: MongoDB query filter
            limit: Maximum documents to return

        Returns:
            List of documents (as dicts)
        """
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]
        cursor = coll.find(query).limit(limit)

        # Convert ObjectId to string for serialization
        results = []
        for doc in cursor:
            doc = dict(doc)
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            results.append(doc)

        return results

    def aggregate(self, collection: str, pipeline: list[dict]) -> list[dict]:
        """Execute a MongoDB aggregation pipeline.

        Args:
            collection: Collection name
            pipeline: Aggregation pipeline stages

        Returns:
            Aggregation results
        """
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]
        results = []

        for doc in coll.aggregate(pipeline):
            doc = dict(doc)
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            results.append(doc)

        return results

    def insert(self, collection: str, documents: list[dict]) -> list[str]:
        """Insert documents into a collection.

        Args:
            collection: Collection name
            documents: Documents to insert

        Returns:
            List of inserted document IDs
        """
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]

        if len(documents) == 1:
            result = coll.insert_one(documents[0])
            return [str(result.inserted_id)]
        else:
            result = coll.insert_many(documents)
            return [str(doc_id) for doc_id in result.inserted_ids]

    def update(
        self,
        collection: str,
        query: dict,
        update: dict,
        upsert: bool = False,
    ) -> int:
        """Update documents in a collection.

        Args:
            collection: Collection name
            query: Filter query
            update: Update operations
            upsert: Insert if not found

        Returns:
            Number of modified documents
        """
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]
        result = coll.update_many(query, update, upsert=upsert)
        return result.modified_count

    def delete(self, collection: str, query: dict) -> int:
        """Delete documents from a collection.

        Args:
            collection: Collection name
            query: Filter query

        Returns:
            Number of deleted documents
        """
        if self._db is None:
            raise RuntimeError("Not connected to MongoDB")

        coll = self._db[collection]
        result = coll.delete_many(query)
        return result.deleted_count
