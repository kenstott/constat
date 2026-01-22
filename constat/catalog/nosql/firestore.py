# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Google Cloud Firestore connector for document database support."""

from typing import Optional, Any

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class FirestoreConnector(NoSQLConnector):
    """Connector for Google Cloud Firestore.

    Supports both Firestore in Native mode and Firestore in Datastore mode.
    This connector uses the Firestore Native API.

    Usage:
        # Using application default credentials
        connector = FirestoreConnector(
            project="my-gcp-project",
            name="firestore_main",
        )

        # Using service account key file
        connector = FirestoreConnector(
            project="my-gcp-project",
            credentials_path="/path/to/service-account.json",
            name="firestore_main",
        )

        # Using specific database (multi-database support)
        connector = FirestoreConnector(
            project="my-gcp-project",
            database="my-database",  # default is "(default)"
            name="firestore_main",
        )

        connector.connect()
        collections = connector.get_collections()
        schema = connector.get_collection_schema("users")
    """

    def __init__(
        self,
        project: str,
        database: str = "(default)",
        credentials_path: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        sample_size: int = 100,
    ):
        """
        Initialize Firestore connector.

        Args:
            project: GCP project ID
            database: Firestore database ID (default is "(default)")
            credentials_path: Path to service account JSON key file
            name: Friendly name for this connection
            description: Description of the database
            sample_size: Number of documents to sample for schema inference
        """
        super().__init__(name=name or project, description=description)
        self.project = project
        self.database_id = database
        self.credentials_path = credentials_path
        self.sample_size = sample_size
        self._client = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.DOCUMENT

    def connect(self) -> None:
        """Connect to Firestore."""
        try:
            from google.cloud import firestore
        except ImportError:
            raise ImportError(
                "Firestore connector requires google-cloud-firestore. "
                "Install with: pip install google-cloud-firestore"
            )

        if self.credentials_path:
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            self._client = firestore.Client(
                project=self.project,
                credentials=credentials,
                database=self.database_id,
            )
        else:
            # Use application default credentials
            self._client = firestore.Client(
                project=self.project,
                database=self.database_id,
            )

        self._connected = True

    def disconnect(self) -> None:
        """Disconnect from Firestore."""
        if self._client:
            self._client.close()
            self._client = None
        self._connected = False

    def get_collections(self) -> list[str]:
        """List all top-level collections."""
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        return [coll.id for coll in self._client.collections()]

    def get_subcollections(self, document_path: str) -> list[str]:
        """List subcollections of a document.

        Args:
            document_path: Path to document (e.g., "users/user123")

        Returns:
            List of subcollection names
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        doc_ref = self._client.document(document_path)
        return [coll.id for coll in doc_ref.collections()]

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Get schema for a collection by sampling documents."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        coll_ref = self._client.collection(collection)

        # Sample documents
        samples = []
        for doc in coll_ref.limit(self.sample_size).stream():
            doc_dict = doc.to_dict()
            doc_dict["_id"] = doc.id  # Include document ID
            samples.append(doc_dict)

        # Infer schema
        metadata = self.infer_schema_from_samples(collection, samples)

        # Firestore doesn't have a direct count API without reading all docs
        # Use the sample count as estimate
        metadata.document_count = len(samples)

        self._metadata_cache[collection] = metadata
        return metadata

    def query(
        self,
        collection: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Execute a query on a collection.

        Args:
            collection: Collection path
            query: Query filters as dict (supports basic operators)
            limit: Maximum documents to return

        Returns:
            List of documents
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        coll_ref = self._client.collection(collection)
        query_ref = coll_ref

        # Apply filters
        for field, condition in query.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        query_ref = query_ref.where(field, "==", value)
                    elif op == "$ne":
                        query_ref = query_ref.where(field, "!=", value)
                    elif op == "$gt":
                        query_ref = query_ref.where(field, ">", value)
                    elif op == "$gte":
                        query_ref = query_ref.where(field, ">=", value)
                    elif op == "$lt":
                        query_ref = query_ref.where(field, "<", value)
                    elif op == "$lte":
                        query_ref = query_ref.where(field, "<=", value)
                    elif op == "$in":
                        query_ref = query_ref.where(field, "in", value)
                    elif op == "$not_in":
                        query_ref = query_ref.where(field, "not-in", value)
                    elif op == "$contains":
                        query_ref = query_ref.where(field, "array-contains", value)
                    elif op == "$contains_any":
                        query_ref = query_ref.where(field, "array-contains-any", value)
            else:
                # Simple equality
                query_ref = query_ref.where(field, "==", condition)

        # Apply limit
        query_ref = query_ref.limit(limit)

        results = []
        for doc in query_ref.stream():
            doc_dict = doc.to_dict()
            doc_dict["_id"] = doc.id
            results.append(doc_dict)

        return results

    def get_document(self, document_path: str) -> Optional[dict]:
        """Get a single document by path.

        Args:
            document_path: Full document path (e.g., "users/user123")

        Returns:
            Document data or None if not found
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        doc_ref = self._client.document(document_path)
        doc = doc_ref.get()

        if doc.exists:
            result = doc.to_dict()
            result["_id"] = doc.id
            return result
        return None

    def insert(self, collection: str, documents: list[dict]) -> list[str]:
        """Add documents to a collection.

        Args:
            collection: Collection path
            documents: Documents to add

        Returns:
            List of document IDs
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        coll_ref = self._client.collection(collection)
        ids = []

        for doc in documents:
            # Use _id if provided, otherwise auto-generate
            doc_id = doc.pop("_id", None)

            if doc_id:
                doc_ref = coll_ref.document(doc_id)
                doc_ref.set(doc)
                ids.append(doc_id)
            else:
                _, doc_ref = coll_ref.add(doc)
                ids.append(doc_ref.id)

        return ids

    def set_document(
        self,
        document_path: str,
        data: dict,
        merge: bool = False,
    ) -> str:
        """Set a document (create or overwrite).

        Args:
            document_path: Full document path
            data: Document data
            merge: If True, merge with existing data

        Returns:
            Document ID
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        doc_ref = self._client.document(document_path)
        doc_ref.set(data, merge=merge)
        return doc_ref.id

    def update(
        self,
        document_path: str,
        updates: dict,
    ) -> None:
        """Update specific fields of a document.

        Args:
            document_path: Full document path
            updates: Fields to update
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        doc_ref = self._client.document(document_path)
        doc_ref.update(updates)

    def delete(self, collection: str, query: dict) -> int:
        """Delete documents matching a query.

        Args:
            collection: Collection path
            query: Query to match documents, or {"_id": "doc_id"} for single doc

        Returns:
            Number of deleted documents
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        # Single document deletion
        if "_id" in query and len(query) == 1:
            doc_ref = self._client.collection(collection).document(query["_id"])
            doc_ref.delete()
            return 1

        # Query-based deletion
        docs_to_delete = self.query(collection, query, limit=500)
        coll_ref = self._client.collection(collection)

        batch = self._client.batch()
        count = 0

        for doc in docs_to_delete:
            doc_ref = coll_ref.document(doc["_id"])
            batch.delete(doc_ref)
            count += 1

        batch.commit()
        return count

    def batch_write(
        self,
        operations: list[dict],
    ) -> None:
        """Execute multiple write operations atomically.

        Args:
            operations: List of operations, each with:
                - "type": "set", "update", or "delete"
                - "path": document path
                - "data": document data (for set/update)
                - "merge": bool (for set, optional)
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        batch = self._client.batch()

        for op in operations:
            doc_ref = self._client.document(op["path"])

            if op["type"] == "set":
                batch.set(doc_ref, op["data"], merge=op.get("merge", False))
            elif op["type"] == "update":
                batch.update(doc_ref, op["data"])
            elif op["type"] == "delete":
                batch.delete(doc_ref)

        batch.commit()

    def run_transaction(
        self,
        transaction_fn,
    ) -> Any:
        """Run a transaction.

        Args:
            transaction_fn: Function that takes transaction and client,
                           returns result

        Returns:
            Transaction result
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        from google.cloud.firestore import transactional

        @transactional
        def run_in_transaction(transaction):
            return transaction_fn(transaction, self._client)

        return run_in_transaction(self._client.transaction())

    def collection_group_query(
        self,
        collection_id: str,
        query: dict,
        limit: int = 100,
    ) -> list[dict]:
        """Query across all collections with the same ID.

        Args:
            collection_id: Collection ID to query across
            query: Query filters
            limit: Maximum documents to return

        Returns:
            List of documents with their paths
        """
        if not self._client:
            raise RuntimeError("Not connected to Firestore")

        coll_group = self._client.collection_group(collection_id)
        query_ref = coll_group

        # Apply filters
        for field, condition in query.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op == "$eq":
                        query_ref = query_ref.where(field, "==", value)
                    elif op == "$gt":
                        query_ref = query_ref.where(field, ">", value)
                    elif op == "$gte":
                        query_ref = query_ref.where(field, ">=", value)
                    elif op == "$lt":
                        query_ref = query_ref.where(field, "<", value)
                    elif op == "$lte":
                        query_ref = query_ref.where(field, "<=", value)
            else:
                query_ref = query_ref.where(field, "==", condition)

        query_ref = query_ref.limit(limit)

        results = []
        for doc in query_ref.stream():
            doc_dict = doc.to_dict()
            doc_dict["_id"] = doc.id
            doc_dict["_path"] = doc.reference.path
            results.append(doc_dict)

        return results
