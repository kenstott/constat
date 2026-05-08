# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""NoSQL database connectors.

Provides a unified interface for NoSQL databases that aren't
supported by SQLAlchemy. Each connector provides:
- Connection management
- Schema/structure introspection
- Query execution
- Embedding text generation for vector search

Supported databases:

On-Premise / Self-Hosted:
- MongoDB: Document store (collections, documents)
- Cassandra: Wide-column store (keyspaces, tables) - also DataStax Astra
- Elasticsearch: Search engine (indices, documents) - also Elastic Cloud, OpenSearch

Cloud-Native:
- DynamoDB: AWS key-value/document store
- Cosmos DB: Azure multi-model database (SQL/Core API)
- Firestore: Google Cloud document database

Usage:
    from constat.catalog.nosql import MongoDBConnector

    connector = MongoDBConnector(uri="mongodb://localhost:27017", database="mydb")
    connector.connect()

    # Get schema overview
    overview = connector.get_overview()

    # Get collection schema
    schema = connector.get_collection_schema("users")

    # Query
    results = connector.query("users", {"age": {"$gt": 21}})

Cloud Examples:

    # AWS DynamoDB
    from constat.catalog.nosql import DynamoDBConnector
    connector = DynamoDBConnector(region="us-east-1")
    connector.connect()

    # Azure Cosmos DB
    from constat.catalog.nosql import CosmosDBConnector
    connector = CosmosDBConnector(
        endpoint="https://myaccount.documents.azure.com:443/",
        key="...",
        database="mydb",
    )
    connector.connect()

    # Google Firestore
    from constat.catalog.nosql import FirestoreConnector
    connector = FirestoreConnector(project="my-gcp-project")
    connector.connect()
"""

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo
from .cassandra import CassandraConnector
from .cosmosdb import CosmosDBConnector
from .dynamodb import DynamoDBConnector
from .elasticsearch import ElasticsearchConnector
from .firestore import FirestoreConnector
from .mongodb import MongoDBConnector

__all__ = [
    # Base classes
    "NoSQLConnector",
    "NoSQLType",
    "CollectionMetadata",
    "FieldInfo",
    # On-premise connectors
    "MongoDBConnector",
    "CassandraConnector",
    "ElasticsearchConnector",
    # Cloud connectors
    "DynamoDBConnector",
    "CosmosDBConnector",
    "FirestoreConnector",
]
