# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Database schema introspection, caching, and vector search."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from constat.discovery.doc_tools import DocumentDiscoveryTools

import numpy as np
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from constat.core.config import Config, DatabaseConfig, DatabaseCredentials
from constat.embedding_loader import EmbeddingModelLoader
from constat.catalog.nosql.base import NoSQLConnector
from constat.catalog.file.connector import FileConnector, FileType


@dataclass
class ColumnMetadata:
    """Metadata for a single column."""
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    comment: Optional[str] = None  # Column comment/description from DB
    # For enum-like columns, sample distinct values
    sample_values: Optional[list[str]] = None


@dataclass
class ForeignKey:
    """Foreign key relationship."""
    from_column: str
    to_table: str
    to_column: str
    # FK constraint comment if available
    comment: Optional[str] = None


@dataclass
class TableMetadata:
    """Full metadata for a table."""
    database: str
    name: str
    comment: Optional[str] = None  # Table comment/description from DB
    columns: list[ColumnMetadata] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)
    row_count: int = 0
    # Tables that reference this table
    referenced_by: list[str] = field(default_factory=list)
     # Database type - critical for LLM to know query semantics (sql vs nosql)
    database_type: str = ""  # e.g., "postgresql", "mysql", "mongodb", "elasticsearch"

    @property
    def full_name(self) -> str:
        return f"{self.database}.{self.name}"

    def to_dict(self) -> dict:
        """Convert to dict for LLM tool response."""
        result = {
            "database": self.database,
            "table": self.name,
            "columns": [
                {
                    "name": c.name,
                    "type": c.type,
                    "nullable": c.nullable,
                    "primary_key": c.primary_key,
                    **({"comment": c.comment} if c.comment else {}),
                    **({"sample_values": c.sample_values} if c.sample_values else {}),
                }
                for c in self.columns
            ],
            "primary_keys": self.primary_keys,
            "foreign_keys": [
                {"from": fk.from_column, "to": f"{fk.to_table}.{fk.to_column}"}
                for fk in self.foreign_keys
            ],
            "referenced_by": self.referenced_by,
            "row_count": self.row_count,
        }
        if self.comment:
            result["comment"] = self.comment
        return result

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding.

        Includes table comment, column names, types, and column comments
        to enable semantic search based on business meaning.
        """
        parts = [f"Table: {self.full_name}"]

        # Table comment is highly valuable for semantic search
        if self.comment:
            parts.append(f"Description: {self.comment}")

        parts.append(f"Columns: {', '.join(c.name for c in self.columns)}")

        # Add column types for context
        col_details = [f"{c.name}:{c.type}" for c in self.columns]
        parts.append(f"Schema: {', '.join(col_details)}")

        # Include column comments - these often contain business meaning
        col_comments = [
            f"{c.name}: {c.comment}"
            for c in self.columns
            if c.comment
        ]
        if col_comments:
            parts.append(f"Column descriptions: {'; '.join(col_comments)}")

        # Add relationships
        if self.foreign_keys:
            fk_strs = [f"{fk.from_column}→{fk.to_table}" for fk in self.foreign_keys]
            parts.append(f"References: {', '.join(fk_strs)}")

        if self.referenced_by:
            parts.append(f"Referenced by: {', '.join(self.referenced_by)}")

        return "\n".join(parts)


@dataclass
class TableMatch:
    """Result from vector search."""
    table: str
    database: str
    relevance: float
    summary: str


class SchemaManager:
    """
    Manages database connections, schema introspection, and vector search.

    Supports both SQL databases (via SQLAlchemy) and NoSQL databases
    (MongoDB, Cassandra, Elasticsearch, DynamoDB, CosmosDB, Firestore).

    On initialization:
    1. Connects to all configured databases
    2. Introspects schemas and caches metadata
    3. Builds vector index for semantic search
    4. Generates token-optimized overview for system prompt
    """

    # High-quality embedding model for semantic search (~1.3GB, 1024 dims)
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"

    def __init__(self, config: Config):
        self.config = config
        self.connections: dict[str, Engine] = {}  # SQL connections
        self.nosql_connections: dict[str, NoSQLConnector] = {}  # NoSQL connections
        self.file_connections: dict[str, FileConnector] = {}  # File data sources
        self.metadata_cache: dict[str, TableMetadata] = {}  # key: "db.table"

        # Vector store for embeddings (shared DuckDB)
        from constat.discovery.vector_store import DuckDBVectorStore
        self._vector_store: Optional[DuckDBVectorStore] = None
        self._model: Optional[SentenceTransformer] = None

        # Cached overview string
        self._overview: Optional[str] = None

        # Progress callback (set during initialize)
        self._progress_callback: Optional[Callable[[str, int, int, str], None]] = None

    def initialize(self, progress_callback: Optional[Callable[[str, int, int, str], None]] = None) -> None:
        """Connect to databases, introspect schemas, build vector index.

        Uses cached schema metadata when available to avoid expensive
        database introspection on every startup. Cache is invalidated
        when config.databases changes.

        Args:
            progress_callback: Optional callback for progress updates.
                Called with (stage, current, total, detail) where:
                - stage: 'connecting', 'introspecting', 'indexing', 'done'
                - current: current item number (1-based)
                - total: total items in this stage
                - detail: description of current item
        """
        self._progress_callback = progress_callback

        # Compute config hash for cache validation
        config_hash = self._compute_config_hash()

        # Try to load schema from cache first
        if self._load_schema_cache(config_hash):
            # Cache hit - still need connections for query execution
            self._connect_all()
            # Reverse references are stored in cache, no need to recompute
        else:
            # Cache miss - full introspection required
            self._connect_all()
            self._introspect_all()
            self._resolve_reverse_references()
            # Save to cache for next time
            self._save_schema_cache(config_hash)

        self._build_vector_index()
        self._generate_overview()
        self._progress_callback = None

    def add_database_dynamic(self, db_name: str, db_config: DatabaseConfig) -> bool:
        """Dynamically add and introspect a database after initialization.

        This allows adding project databases at runtime without reinitializing
        the entire schema manager.

        Args:
            db_name: Name for the database
            db_config: Database configuration

        Returns:
            True if successfully added
        """
        try:
            # Connect based on type
            source_type = db_config.type or "sql"
            logger.info(f"add_database_dynamic: {db_name}, type={source_type}, uri={db_config.uri}")
            logger.info(f"  is_file_source={db_config.is_file_source()}, is_nosql={db_config.is_nosql()}")

            if db_config.is_file_source():
                logger.info(f"  Connecting as file source")
                self._connect_file(db_name, db_config)
                # Introspect file source
                connector = self.file_connections.get(db_name)
                if connector:
                    table_meta = TableMetadata(
                        database=db_name,
                        name=db_name,
                        comment=db_config.description,
                        database_type=source_type,
                    )
                    # Get columns from file
                    try:
                        columns = connector.get_columns()
                        table_meta.columns = [
                            ColumnMetadata(name=c["name"], type=c.get("type", "unknown"))
                            for c in columns
                        ]
                        logger.info(f"  File source has {len(table_meta.columns)} columns")
                    except Exception as e:
                        logger.warning(f"  Failed to get columns: {e}")
                    self.metadata_cache[f"{db_name}.{db_name}"] = table_meta
                    logger.info(f"  Added to metadata_cache: {db_name}.{db_name}")
            elif db_config.is_nosql():
                logger.info(f"  Connecting as NoSQL")
                self._connect_nosql(db_name, db_config)
                # Introspect NoSQL
                connector = self.nosql_connections.get(db_name)
                if connector:
                    collections = connector.list_collections()
                    logger.info(f"  NoSQL has {len(collections)} collections")
                    for coll_name in collections:
                        schema = connector.infer_schema(coll_name)
                        table_meta = TableMetadata(
                            database=db_name,
                            name=coll_name,
                            comment=db_config.description,
                            database_type=source_type,
                        )
                        if schema:
                            table_meta.columns = [
                                ColumnMetadata(name=f["name"], type=f.get("type", "unknown"))
                                for f in schema.get("fields", [])
                            ]
                        self.metadata_cache[f"{db_name}.{coll_name}"] = table_meta
            else:
                # SQL database
                logger.info(f"  Connecting as SQL database")
                self._connect_sql(db_name, db_config)
                engine = self.connections.get(db_name)
                if engine:
                    inspector = inspect(engine)
                    table_names = inspector.get_table_names()
                    logger.info(f"  SQL database has {len(table_names)} tables: {table_names}")
                    for table_name in table_names:
                        table_meta = self._introspect_table(db_name, engine, inspector, table_name)
                        self.metadata_cache[table_meta.full_name] = table_meta
                        logger.info(f"  Introspected table: {db_name}.{table_name}")
                else:
                    logger.warning(f"  No engine created for {db_name}")

            # Rebuild vector index to include new entities
            logger.info(f"  Rebuilding vector index...")
            self._build_vector_index()
            logger.info(f"  metadata_cache now has {len(self.metadata_cache)} entries")

            logger.info(f"Dynamically added database: {db_name} ({source_type})")
            return True
        except Exception as e:
            logger.exception(f"Failed to dynamically add database {db_name}: {e}")
            return False

    def refresh(self, progress_callback: Optional[Callable[[str, int, int, str], None]] = None) -> None:
        """Clear caches and re-introspect all schemas.

        Use this when database schemas have changed and you need fresh metadata.
        """
        # Clear all caches (memory and disk)
        self.metadata_cache.clear()
        self._overview = None

        # Clear schema entities from vector store
        if self._vector_store:
            self._vector_store.clear_catalog_entities('schema')

        # Delete disk caches to force fresh introspection
        schema_cache = self._get_schema_cache_path()
        if schema_cache.exists():
            schema_cache.unlink()

        # Re-initialize (connections are preserved)
        self._progress_callback = progress_callback
        self._introspect_all()
        self._resolve_reverse_references()
        config_hash = self._compute_config_hash()
        self._save_schema_cache(config_hash)
        self._build_vector_index()
        self._generate_overview()
        self._progress_callback = None

    def _emit_progress(self, stage: str, current: int, total: int, detail: str) -> None:
        """Emit progress update if callback is registered."""
        if self._progress_callback:
            self._progress_callback(stage, current, total, detail)

    def _connect_all(self) -> None:
        """Establish connections to all configured databases."""
        db_items = list(self.config.databases.items())
        total = len(db_items)
        for i, (db_name, db_config) in enumerate(db_items, 1):
            source_type = db_config.type or "sql"
            self._emit_progress("connecting", i, total, f"{db_name} ({source_type})")
            if db_config.is_file_source():
                self._connect_file(db_name, db_config)
            elif db_config.is_nosql():
                self._connect_nosql(db_name, db_config)
            else:
                self._connect_sql(db_name, db_config)

    def _connect_file(self, db_name: str, db_config: DatabaseConfig) -> None:
        """Connect to a file-based data source."""
        config_dir = self.config.config_dir if self.config else None
        connector = FileConnector.from_config(db_name, db_config, config_dir=config_dir)
        self.file_connections[db_name] = connector

    def _connect_sql(self, db_name: str, db_config: DatabaseConfig) -> None:
        """Connect to a SQL database via SQLAlchemy."""
        config_dir = self.config.config_dir if self.config else None
        connection_uri = db_config.get_connection_uri(config_dir)
        engine = create_engine(connection_uri)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        self.connections[db_name] = engine

    def _connect_nosql(self, db_name: str, db_config: DatabaseConfig) -> None:
        """Connect to a NoSQL database using the appropriate connector."""
        connector = self._create_nosql_connector(db_name, db_config)
        if connector:
            connector.connect()
            self.nosql_connections[db_name] = connector

    def _create_nosql_connector(self, db_name: str, db_config: DatabaseConfig) -> Optional[NoSQLConnector]:
        """Create the appropriate NoSQL connector based on database type."""
        db_type = db_config.type

        if db_type == "mongodb":
            from constat.catalog.nosql.mongodb import MongoDBConnector
            return MongoDBConnector(
                uri=db_config.uri or "mongodb://localhost:27017",
                database=db_config.database or db_name,
                name=db_name,
                description=db_config.description,
                sample_size=db_config.sample_size,
            )

        elif db_type == "cassandra":
            from constat.catalog.nosql.cassandra import CassandraConnector
            auth_provider = None
            if db_config.username and db_config.password:
                auth_provider = (db_config.username, db_config.password)
            return CassandraConnector(
                keyspace=db_config.keyspace or db_name,
                hosts=db_config.hosts,
                port=db_config.port or 9042,
                name=db_name,
                description=db_config.description,
                cloud_config=db_config.cloud_config,
                auth_provider=auth_provider,
                sample_size=db_config.sample_size,
            )

        elif db_type == "elasticsearch":
            from constat.catalog.nosql.elasticsearch import ElasticsearchConnector
            return ElasticsearchConnector(
                hosts=db_config.hosts or ["http://localhost:9200"],
                name=db_name,
                description=db_config.description,
                api_key=db_config.api_key,
                username=db_config.username,
                password=db_config.password,
                sample_size=db_config.sample_size,
            )

        elif db_type == "dynamodb":
            from constat.catalog.nosql.dynamodb import DynamoDBConnector
            return DynamoDBConnector(
                region=db_config.region,
                name=db_name,
                description=db_config.description,
                endpoint_url=db_config.endpoint_url,
                aws_access_key_id=db_config.aws_access_key_id,
                aws_secret_access_key=db_config.aws_secret_access_key,
                aws_session_token=db_config.aws_session_token,
                profile_name=db_config.profile_name,
                sample_size=db_config.sample_size,
            )

        elif db_type == "cosmosdb":
            from constat.catalog.nosql.cosmosdb import CosmosDBConnector
            return CosmosDBConnector(
                endpoint=db_config.endpoint or "",
                key=db_config.key or "",
                database=db_config.database or db_name,
                container=db_config.container or "",
                name=db_name,
                description=db_config.description,
                sample_size=db_config.sample_size,
            )

        elif db_type == "firestore":
            from constat.catalog.nosql.firestore import FirestoreConnector
            return FirestoreConnector(
                project=db_config.project or "",
                collection=db_config.collection or "",
                name=db_name,
                description=db_config.description,
                credentials_path=db_config.credentials_path,
                sample_size=db_config.sample_size,
            )

        return None

    def _introspect_all(self) -> None:
        """Introspect all tables/collections in all databases."""
        # Count total items for progress
        total_items = len(self.file_connections)  # Files are 1:1
        for db_name, engine in self.connections.items():
            inspector = inspect(engine)
            total_items += len(inspector.get_table_names())
        for db_name, connector in self.nosql_connections.items():
            total_items += len(connector.get_collections())

        current = 0

        # Introspect SQL databases
        for db_name, engine in self.connections.items():
            inspector = inspect(engine)

            for table_name in inspector.get_table_names():
                current += 1
                self._emit_progress("introspecting", current, total_items, f"{db_name}.{table_name}")
                table_meta = self._introspect_table(db_name, engine, inspector, table_name)
                self.metadata_cache[table_meta.full_name] = table_meta

        # Introspect NoSQL databases
        for db_name, connector in self.nosql_connections.items():
            for collection_name in connector.get_collections():
                current += 1
                self._emit_progress("introspecting", current, total_items, f"{db_name}.{collection_name}")
                # Get schema for the collection
                collection_meta = connector.get_collection_schema(collection_name)
                # Convert NoSQL CollectionMetadata to TableMetadata
                table_meta = self._convert_nosql_metadata(db_name, connector, collection_meta)
                self.metadata_cache[table_meta.full_name] = table_meta

        # Introspect file-based data sources
        for db_name, connector in self.file_connections.items():
            current += 1
            self._emit_progress("introspecting", current, total_items, f"{db_name} (file)")
            file_meta = connector.get_metadata()
            table_meta = self._convert_file_metadata(db_name, connector, file_meta)
            self.metadata_cache[table_meta.full_name] = table_meta

    def _convert_nosql_metadata(self, db_name: str, connector: NoSQLConnector, collection_meta) -> TableMetadata:
        """Convert NoSQL CollectionMetadata to TableMetadata for unified handling."""
        # collection_meta is already the schema from get_collection_schema()

        # Convert fields to columns
        columns = []
        for field_info in collection_meta.fields:
            columns.append(ColumnMetadata(
                name=field_info.name,
                type=field_info.data_type,  # FieldInfo uses data_type not type
                nullable=field_info.nullable,
                primary_key=field_info.is_indexed and field_info.is_unique,
                comment=field_info.description if hasattr(field_info, 'description') else None,
            ))

        # Get the database type from config
        db_type = ""
        if db_name in self.config.databases:
            db_type = self.config.databases[db_name].type

        # Get key fields (fields that are indexed and unique)
        key_fields = [f.name for f in collection_meta.fields if f.is_indexed and f.is_unique]

        return TableMetadata(
            database=db_name,
            name=collection_meta.name,
            comment=collection_meta.description,
            columns=columns,
            primary_keys=key_fields,
            foreign_keys=[],  # NoSQL typically doesn't have FK constraints
            row_count=collection_meta.document_count,
            referenced_by=[],
            database_type=db_type,
        )

    def _convert_file_metadata(self, db_name: str, connector: FileConnector, file_meta) -> TableMetadata:
        """Convert FileMetadata to TableMetadata for unified handling."""
        # Convert columns
        columns = []
        for col_info in file_meta.columns:
            columns.append(ColumnMetadata(
                name=col_info.name,
                type=col_info.data_type,
                nullable=col_info.nullable,
                primary_key=False,
                comment=col_info.description,
                sample_values=col_info.sample_values[:5] if col_info.sample_values else None,
            ))

        # Include file path in the comment for LLM awareness
        comment = file_meta.description or ""
        if file_meta.path:
            comment = f"{comment}\nPath: {file_meta.path}" if comment else f"Path: {file_meta.path}"

        return TableMetadata(
            database=db_name,
            name=file_meta.name,  # Use logical name
            comment=comment,
            columns=columns,
            primary_keys=[],
            foreign_keys=[],
            row_count=file_meta.row_count,
            referenced_by=[],
            database_type=file_meta.file_type.value,  # e.g., "csv", "parquet"
        )

    def _introspect_table(
        self, db_name: str, engine: Engine, inspector, table_name: str
    ) -> TableMetadata:
        """Introspect a single table including comments."""
        # Get table comment (if supported by the database)
        table_comment = None
        try:
            comment_info = inspector.get_table_comment(table_name)
            if comment_info and comment_info.get("text"):
                table_comment = comment_info["text"]
        except (NotImplementedError, Exception):
            # Some databases/dialects don't support table comments
            pass

        # Get columns with comments
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append(
                ColumnMetadata(
                    name=col["name"],
                    type=self._simplify_type(str(col["type"])),
                    nullable=col.get("nullable", True),
                    comment=col.get("comment"),  # Column comment if available
                )
            )

        # Get primary keys
        pk_constraint = inspector.get_pk_constraint(table_name)
        primary_keys = pk_constraint.get("constrained_columns", []) if pk_constraint else []

        # Mark PK columns
        for col in columns:
            if col.name in primary_keys:
                col.primary_key = True

        # Get foreign keys (with optional comment)
        foreign_keys = []
        for fk in inspector.get_foreign_keys(table_name):
            # Handle composite FKs by taking first column
            if fk["constrained_columns"] and fk["referred_columns"]:
                foreign_keys.append(
                    ForeignKey(
                        from_column=fk["constrained_columns"][0],
                        to_table=fk["referred_table"],
                        to_column=fk["referred_columns"][0],
                        comment=fk.get("comment"),  # FK comment if available
                    )
                )

        # Get approximate row count
        row_count = 0
        try:
            with engine.connect() as conn:
                result = conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
                row_count = result.scalar() or 0
        except Exception:
            pass  # Skip if count fails

        # Get the database type from config
        db_type = ""
        if db_name in self.config.databases:
            db_type = self.config.databases[db_name].type

        return TableMetadata(
            database=db_name,
            name=table_name,
            comment=table_comment,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            row_count=row_count,
            database_type=db_type,
        )

    def _simplify_type(self, type_str: str) -> str:
        """Simplify verbose SQL types for token efficiency."""
        type_lower = type_str.lower()

        if "int" in type_lower:
            return "int"
        if "varchar" in type_lower or "text" in type_lower or "char" in type_lower:
            return "str"
        if "float" in type_lower or "double" in type_lower or "real" in type_lower:
            return "float"
        if "decimal" in type_lower or "numeric" in type_lower:
            return "decimal"
        if "datetime" in type_lower or "timestamp" in type_lower:
            return "datetime"
        if "date" in type_lower:
            return "date"
        if "time" in type_lower:
            return "time"
        if "bool" in type_lower:
            return "bool"
        if "blob" in type_lower or "binary" in type_lower:
            return "bytes"

        return type_str

    def _resolve_reverse_references(self) -> None:
        """Populate referenced_by for each table based on foreign keys."""
        for table_meta in self.metadata_cache.values():
            for fk in table_meta.foreign_keys:
                # Find the referenced table
                ref_key = f"{table_meta.database}.{fk.to_table}"
                if ref_key in self.metadata_cache:
                    ref_table = self.metadata_cache[ref_key]
                    ref_str = f"{table_meta.name}.{fk.from_column}"
                    if ref_str not in ref_table.referenced_by:
                        ref_table.referenced_by.append(ref_str)

    def _compute_config_hash(self) -> str:
        """Compute a deterministic hash of the databases config.

        The hash changes when databases are added, removed, or their
        connection parameters change. This triggers re-introspection
        and re-embedding.
        """
        # Build a canonical representation of the databases config
        # Include only the keys that affect schema structure
        config_data = {}
        for db_name, db_config in sorted(self.config.databases.items()):
            config_data[db_name] = {
                "type": db_config.type or "",
                "uri": db_config.uri or "",
                "database": db_config.database or "",
                "hosts": sorted(db_config.hosts) if db_config.hosts else [],
                "port": db_config.port or 0,
                "path": db_config.path or "",
                "keyspace": db_config.keyspace or "",
            }

        # Create deterministic JSON and hash it
        config_json = json.dumps(config_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(config_json.encode()).hexdigest()[:16]

    def _get_cache_dir(self) -> Path:
        """Get the .constat cache directory."""
        source_path = getattr(self.config, '_source_path', None)
        if source_path:
            cache_dir = source_path.parent / ".constat"
        else:
            cache_dir = Path.cwd() / ".constat"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _get_schema_cache_path(self) -> Path:
        """Get the path to the schema cache JSON file."""
        return self._get_cache_dir() / "schema_cache.json"

    def _load_schema_cache(self, expected_hash: str) -> bool:
        """Load schema metadata from cache if hash matches.

        Args:
            expected_hash: The current config hash

        Returns:
            True if cache was loaded successfully, False otherwise
        """
        cache_path = self._get_schema_cache_path()
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)

            # Check hash matches
            if cache_data.get("config_hash") != expected_hash:
                return False

            # Rebuild metadata_cache from cached data
            tables = cache_data.get("tables", {})
            for full_name, table_dict in tables.items():
                # Rebuild ColumnMetadata objects
                columns = [
                    ColumnMetadata(
                        name=c["name"],
                        type=c["type"],
                        nullable=c.get("nullable", True),
                        primary_key=c.get("primary_key", False),
                        comment=c.get("comment"),
                        sample_values=c.get("sample_values"),
                    )
                    for c in table_dict.get("columns", [])
                ]

                # Rebuild ForeignKey objects
                foreign_keys = [
                    ForeignKey(
                        from_column=fk["from_column"],
                        to_table=fk["to_table"],
                        to_column=fk["to_column"],
                        comment=fk.get("comment"),
                    )
                    for fk in table_dict.get("foreign_keys", [])
                ]

                self.metadata_cache[full_name] = TableMetadata(
                    database=table_dict["database"],
                    name=table_dict["name"],
                    comment=table_dict.get("comment"),
                    columns=columns,
                    primary_keys=table_dict.get("primary_keys", []),
                    foreign_keys=foreign_keys,
                    row_count=table_dict.get("row_count", 0),
                    referenced_by=table_dict.get("referenced_by", []),
                    database_type=table_dict.get("database_type", ""),
                )

            return True
        except Exception:
            return False

    def _save_schema_cache(self, config_hash: str) -> None:
        """Save schema metadata to cache file."""
        cache_path = self._get_schema_cache_path()

        try:
            tables = {}
            for full_name, meta in self.metadata_cache.items():
                tables[full_name] = {
                    "database": meta.database,
                    "name": meta.name,
                    "comment": meta.comment,
                    "columns": [
                        {
                            "name": c.name,
                            "type": c.type,
                            "nullable": c.nullable,
                            "primary_key": c.primary_key,
                            "comment": c.comment,
                            "sample_values": c.sample_values,
                        }
                        for c in meta.columns
                    ],
                    "primary_keys": meta.primary_keys,
                    "foreign_keys": [
                        {
                            "from_column": fk.from_column,
                            "to_table": fk.to_table,
                            "to_column": fk.to_column,
                            "comment": fk.comment,
                        }
                        for fk in meta.foreign_keys
                    ],
                    "row_count": meta.row_count,
                    "referenced_by": meta.referenced_by,
                    "database_type": meta.database_type,
                }

            cache_data = {
                "config_hash": config_hash,
                "tables": tables,
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            # Silently ignore cache save failures
            pass

    def _build_vector_index(self) -> None:
        """Build vector embeddings for all tables.

        Uses caching based on schema hash - embeddings are only recomputed
        when the schema structure changes (new tables, columns, or types).
        Embeddings are stored in the shared vectors.duckdb.
        """
        if not self.metadata_cache:
            return

        # Initialize vector store if needed
        if self._vector_store is None:
            from constat.discovery.vector_store import DuckDBVectorStore
            self._vector_store = DuckDBVectorStore()

        # Compute hash of config.databases
        config_hash = self._compute_config_hash()

        # Check if cache is valid
        cached_hash = self._vector_store.get_catalog_config_hash('schema')
        if cached_hash == config_hash and self._vector_store.count_catalog_entities(source='schema') > 0:
            # Cache hit - embeddings already in DuckDB
            return

        # Cache miss - need to compute embeddings
        self._emit_progress("indexing", 1, 2, "loading embedding model")
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Generate entity records for unified catalog
        # Each table becomes an entity, and each column becomes a child entity
        entities = []
        texts = []

        for full_name in sorted(self.metadata_cache.keys()):
            table_meta = self.metadata_cache[full_name]
            embedding_text = table_meta.to_embedding_text()
            texts.append(embedding_text)

            # Table entity - use normalized name for display, keep original in metadata
            from constat.discovery.models import normalize_entity_name
            entities.append({
                "id": full_name,
                "name": normalize_entity_name(table_meta.name),
                "type": "table",
                "parent_id": None,
                "metadata": {
                    "database": table_meta.database,
                    "database_type": table_meta.database_type,
                    "row_count": table_meta.row_count,
                    "comment": table_meta.comment,
                    "original_name": table_meta.name,
                    "columns": [c.name for c in table_meta.columns],
                    "primary_keys": table_meta.primary_keys,
                    "foreign_keys": [
                        {"from": fk.from_column, "to_table": fk.to_table, "to_column": fk.to_column}
                        for fk in table_meta.foreign_keys
                    ],
                },
            })

        # Generate embeddings for tables
        self._emit_progress("indexing", 2, 2, f"vectorizing {len(texts)} tables")
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Clear old and save to DuckDB
        self._vector_store.clear_catalog_entities('schema')
        self._vector_store.add_catalog_entities(entities, embeddings, 'schema', config_hash)

        # Extract entities from table/column descriptions
        self._extract_entities_from_descriptions()

    def _extract_entities_from_descriptions(self) -> None:
        """Extract entities from table and column descriptions using spaCy NER.

        This enables finding relationships between schema elements and concepts
        mentioned in their descriptions/comments.

        Steps:
        1. Collect all description chunks
        2. Generate embeddings and store chunks in vector store
        3. Extract entities and chunk links
        4. Store entities and links for proper reference tracking
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return

        # Lazy imports to avoid circular dependency
        from constat.discovery.models import DocumentChunk, ChunkEntity
        from constat.discovery.entity_extractor import EntityExtractor, ExtractionConfig

        # Collect all chunks first
        chunks: list[DocumentChunk] = []
        for full_name, table_meta in self.metadata_cache.items():
            # Table comment
            if table_meta.comment:
                chunks.append(DocumentChunk(
                    document_name=f"schema:{full_name}",
                    content=table_meta.comment,
                    section="table_description",
                    chunk_index=0,
                ))

            # Column comments
            for i, col in enumerate(table_meta.columns):
                if col.comment:
                    chunks.append(DocumentChunk(
                        document_name=f"schema:{full_name}.{col.name}",
                        content=col.comment,
                        section="column_description",
                        chunk_index=i,
                    ))

        if not chunks:
            logger.debug("No schema descriptions to extract entities from")
            return

        # Step 1: Generate embeddings and store chunks
        if self._model is not None:
            try:
                texts = [c.content for c in chunks]
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                self._vector_store.add_chunks(chunks, embeddings)
                logger.debug(f"Stored {len(chunks)} schema description chunks")
            except Exception as e:
                logger.warning(f"Failed to store schema description chunks: {e}")

        # Step 2: Configure extractor with NER only
        config = ExtractionConfig(
            extract_schema=False,
            extract_ner=True,
        )
        extractor = EntityExtractor(config)

        # Step 3: Extract entities and collect links
        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        # Step 4: Store entities
        entities = extractor.get_all_entities()
        if entities:
            logger.debug(f"Extracted {len(entities)} entities from schema descriptions")
            self._vector_store.add_entities(entities, source="schema")

        # Step 5: Store chunk-entity links
        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links: dict[tuple[str, str], ChunkEntity] = {}
            for link in all_links:
                key = (link.chunk_id, link.entity_id)
                if key not in unique_links:
                    unique_links[key] = link
                else:
                    existing = unique_links[key]
                    unique_links[key] = ChunkEntity(
                        chunk_id=link.chunk_id,
                        entity_id=link.entity_id,
                        mention_count=existing.mention_count + link.mention_count,
                        confidence=max(existing.confidence, link.confidence),
                        mention_text=existing.mention_text or link.mention_text,
                    )
            self._vector_store.link_chunk_entities(list(unique_links.values()))
            logger.debug(f"Created {len(unique_links)} chunk-entity links for schema descriptions")

    def _generate_overview(self) -> None:
        """Generate token-optimized overview for system prompt.

        Includes column names to reduce LLM tool calls for schema exploration.
        """
        lines = []

        # Add global databases description if provided
        if self.config.databases_description:
            lines.append(self.config.databases_description)
            lines.append("")

        lines.append("Available databases:")

        # Build a lookup for database descriptions
        db_descriptions = {
            db_name: db_config.description
            for db_name, db_config in self.config.databases.items()
            if db_config.description
        }

        # Group tables by database
        by_db: dict[str, list[TableMetadata]] = {}
        for table_meta in self.metadata_cache.values():
            by_db.setdefault(table_meta.database, []).append(table_meta)

        for db_name, tables in sorted(by_db.items()):
            total_rows = sum(t.row_count for t in tables)

            # Include database description if available
            if db_name in db_descriptions:
                lines.append(f"  {db_name}: {db_descriptions[db_name]}")
            else:
                lines.append(f"  {db_name}: ({len(tables)} tables, ~{total_rows:,} rows)")

            # Include compact schema for each table (table_name(col1, col2, ...))
            for table in sorted(tables, key=lambda t: t.name):
                col_names = ", ".join(c.name for c in table.columns[:15])  # Limit columns shown
                if len(table.columns) > 15:
                    col_names += f", ... (+{len(table.columns) - 15} more)"
                pk_cols = [c.name for c in table.columns if c.primary_key]
                pk_info = f" [PK: {', '.join(pk_cols)}]" if pk_cols else ""
                lines.append(f"    {table.name}({col_names}){pk_info} ~{table.row_count} rows")

        # Add key relationships
        all_fks = []
        for table_meta in self.metadata_cache.values():
            for fk in table_meta.foreign_keys:
                all_fks.append(
                    f"{table_meta.name}.{fk.from_column} → {fk.to_table}.{fk.to_column}"
                )

        if all_fks:
            lines.append("\nKey relationships:")
            for fk_str in sorted(set(all_fks)):
                lines.append(f"  {fk_str}")

        self._overview = "\n".join(lines)

    def get_overview(self) -> str:
        """Return token-optimized schema overview for system prompt.

        WARNING: This can be very large for databases with many tables.
        Consider using get_brief_summary() + discovery tools instead.
        """
        if self._overview is None:
            self._generate_overview()
        return self._overview or ""

    def get_brief_summary(self, allowed_databases: Optional[set[str]] = None) -> str:
        """Return a brief summary of databases without listing all tables.

        Use this instead of get_overview() when table count is potentially large.
        The LLM can use discovery tools (find_relevant_tables, get_table_schema)
        to explore specific tables as needed.

        Args:
            allowed_databases: Set of allowed database names. If None, all databases
                are included. If empty set, no databases are included.
        """
        lines = ["Available databases:"]

        # Build a lookup for database descriptions
        db_descriptions = {
            db_name: db_config.description
            for db_name, db_config in self.config.databases.items()
            if db_config.description
        }

        # Group tables by database
        by_db: dict[str, list[TableMetadata]] = {}
        for table_meta in self.metadata_cache.values():
            by_db.setdefault(table_meta.database, []).append(table_meta)

        for db_name, tables in sorted(by_db.items()):
            # Skip databases not allowed by permissions
            if allowed_databases is not None and db_name not in allowed_databases:
                continue

            total_rows = sum(t.row_count for t in tables)

            if db_name in db_descriptions:
                lines.append(f"  {db_name}: {db_descriptions[db_name]} ({len(tables)} tables, ~{total_rows:,} rows)")
            else:
                lines.append(f"  {db_name}: {len(tables)} tables, ~{total_rows:,} rows")

        lines.append("\nUse discovery tools to explore: find_relevant_tables(query), get_table_schema(table)")
        return "\n".join(lines)

    def get_table_schema(self, table: str) -> dict:
        """
        Return full schema for one table.

        Args:
            table: Table name as "database.table" or just "table" (if unambiguous)

        Returns:
            Dict with columns, types, relationships, row count
        """
        # Try exact match first
        if table in self.metadata_cache:
            return self.metadata_cache[table].to_dict()

        # Try finding by table name only (if unambiguous)
        matches = [
            meta for meta in self.metadata_cache.values() if meta.name == table
        ]
        if len(matches) == 1:
            return matches[0].to_dict()
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous table name '{table}'. "
                f"Matches: {[m.full_name for m in matches]}"
            )

        raise KeyError(f"Table not found: {table}")

    def find_relevant_tables(
        self,
        query: str,
        top_k: int = 5,
        doc_tools: Optional["DocumentDiscoveryTools"] = None,
        doc_limit: int = 2,
    ) -> list[dict]:
        """
        Find tables relevant to a natural language query using vector search.

        Queries embeddings directly from DuckDB using array_cosine_similarity.

        Args:
            query: Natural language description of what data is needed
            top_k: Maximum number of results to return
            doc_tools: Optional DocumentDiscoveryTools instance for enriching
                      results with relevant documentation excerpts
            doc_limit: Maximum number of doc excerpts per table (default 2)

        Returns:
            List of dicts with table, database, relevance score, summary,
            and optionally documentation excerpts mentioning the table
        """
        if self._vector_store is None or self._vector_store.count_catalog_entities(source='schema') == 0:
            return []

        # Lazy load the model only when needed for queries (uses shared loader)
        if self._model is None:
            self._model = EmbeddingModelLoader.get_instance().get_model()

        # Embed the query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Search unified catalog entities for schema tables
        search_results = self._vector_store.search_catalog_entities(
            query_embedding, source='schema', entity_type='table', limit=top_k
        )

        results = []
        for entity in search_results:
            full_name = entity["id"]
            relevance = entity["similarity"]
            # Vector store normalizes IDs to lowercase, but metadata_cache uses original case
            # Look up using case-insensitive match
            table_meta = self.metadata_cache.get(full_name)
            if not table_meta:
                # Try case-insensitive lookup
                for key, meta in self.metadata_cache.items():
                    if key.lower() == full_name.lower():
                        table_meta = meta
                        full_name = key  # Use original case
                        break
            if not table_meta:
                continue

            # Generate a brief summary
            col_names = [c.name for c in table_meta.columns[:5]]
            if len(table_meta.columns) > 5:
                col_names.append("...")
            summary = f"{table_meta.name}: {', '.join(col_names)} ({table_meta.row_count:,} rows)"

            result = {
                "table": table_meta.name,
                "database": table_meta.database,
                "full_name": full_name,
                "relevance": round(relevance, 3),
                "summary": summary,
                # Database type - tells LLM what query semantics to use
                "database_type": table_meta.database_type,  # e.g., "postgresql", "mongodb"
                "is_nosql": table_meta.database_type in ("mongodb", "elasticsearch", "dynamodb", "cosmosdb", "firestore", "cassandra"),
            }

            # Enrich with document context if doc_tools provided
            if doc_tools:
                doc_context = doc_tools.explore_entity(table_meta.name, limit=doc_limit)
                if doc_context:
                    result["documentation"] = [
                        {
                            "document": d["document"],
                            "excerpt": d["excerpt"],
                            "section": d.get("section"),
                        }
                        for d in doc_context
                    ]

            results.append(result)

        return results

    def list_tables(self) -> list[str]:
        """Return list of all table full names."""
        return list(self.metadata_cache.keys())

    def get_description_text(self) -> list[tuple[str, str]]:
        """Return all metadata text from schema for NER processing.

        Includes table names, column names, and their descriptions.

        Returns:
            List of (source_name, text) tuples for NER extraction
        """
        from constat.discovery.models import normalize_entity_name

        results = []

        for table_meta in self.metadata_cache.values():
            # Table name (normalized for NER)
            table_name_normalized = normalize_entity_name(table_meta.name)
            results.append((f"table:{table_meta.full_name}", table_name_normalized))

            # Table description
            if table_meta.comment:
                results.append((f"table:{table_meta.full_name}:desc", table_meta.comment))

            # Column names and descriptions
            for col in table_meta.columns:
                col_name_normalized = normalize_entity_name(col.name)
                results.append((f"column:{table_meta.full_name}.{col.name}", col_name_normalized))
                if col.comment:
                    results.append((f"column:{table_meta.full_name}.{col.name}:desc", col.comment))

        return results

    def get_entity_names(self, include_columns: bool = False, normalize: bool = True) -> list[str]:
        """Return table names (and optionally column names) for entity extraction.

        By default, only returns table names since column names are often too
        generic ("date", "name", "status") and cause false matches in documents.

        Args:
            include_columns: If True, also include column names (default False)
            normalize: If True, normalize names for NER (e.g., "performance_reviews" -> "performance review")

        Returns:
            List of unique entity names
        """
        from constat.discovery.models import normalize_entity_name

        entities = set()

        for table_meta in self.metadata_cache.values():
            # Add table name (without database prefix for matching)
            name = table_meta.name
            if normalize:
                name = normalize_entity_name(name)
            entities.add(name)

            # Optionally add column names
            if include_columns:
                for col in table_meta.columns:
                    col_name = col.name
                    if normalize:
                        col_name = normalize_entity_name(col_name)
                    entities.add(col_name)

        return list(entities)

    def get_table_metadata(self, database: str, table_name: str) -> Optional[TableMetadata]:
        """Get metadata for a specific table.

        Args:
            database: Database name
            table_name: Table name (case-insensitive)

        Returns:
            TableMetadata if found, None otherwise
        """
        # Try exact match first
        full_name = f"{database}.{table_name}"
        if full_name in self.metadata_cache:
            return self.metadata_cache[full_name]

        # Try case-insensitive match
        table_lower = table_name.lower()
        for key, meta in self.metadata_cache.items():
            if meta.database == database and meta.name.lower() == table_lower:
                return meta

        # Try matching just by table name (any database)
        for key, meta in self.metadata_cache.items():
            if meta.name.lower() == table_lower:
                return meta

        return None

    def get_connection(self, database: str) -> Union[Engine, NoSQLConnector, FileConnector]:
        """Get connection for a database (SQL Engine, NoSQL Connector, or File Connector)."""
        if database in self.connections:
            return self.connections[database]
        if database in self.nosql_connections:
            return self.nosql_connections[database]
        if database in self.file_connections:
            return self.file_connections[database]
        raise KeyError(f"Database not found: {database}")

    def get_sql_connection(self, database: str) -> Engine:
        """Get SQLAlchemy engine for a SQL database."""
        if database not in self.connections:
            raise KeyError(f"SQL database not found: {database}")
        return self.connections[database]

    def get_nosql_connection(self, database: str) -> NoSQLConnector:
        """Get NoSQL connector for a database."""
        if database not in self.nosql_connections:
            raise KeyError(f"NoSQL database not found: {database}")
        return self.nosql_connections[database]

    def is_nosql(self, database: str) -> bool:
        """Check if a database is NoSQL."""
        return database in self.nosql_connections

    def is_file_source(self, database: str) -> bool:
        """Check if a database is a file-based data source."""
        return database in self.file_connections

    def get_file_connection(self, database: str) -> FileConnector:
        """Get FileConnector for a file-based data source."""
        if database not in self.file_connections:
            raise KeyError(f"File data source not found: {database}")
        return self.file_connections[database]
