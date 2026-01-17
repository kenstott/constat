"""Database schema introspection, caching, and vector search."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from constat.core.config import Config, DatabaseConfig, DatabaseCredentials
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

    # Lightweight, fast embedding model (~80MB)
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, config: Config):
        self.config = config
        self.connections: dict[str, Engine] = {}  # SQL connections
        self.nosql_connections: dict[str, NoSQLConnector] = {}  # NoSQL connections
        self.file_connections: dict[str, FileConnector] = {}  # File data sources
        self.metadata_cache: dict[str, TableMetadata] = {}  # key: "db.table"

        # Vector index components
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_keys: list[str] = []  # Maps index → table full_name
        self._model: Optional[SentenceTransformer] = None

        # Cached overview string
        self._overview: Optional[str] = None

        # Progress callback (set during initialize)
        self._progress_callback: Optional[Callable[[str, int, int, str], None]] = None

    def initialize(self, progress_callback: Optional[Callable[[str, int, int, str], None]] = None) -> None:
        """Connect to databases, introspect schemas, build vector index.

        Args:
            progress_callback: Optional callback for progress updates.
                Called with (stage, current, total, detail) where:
                - stage: 'connecting', 'introspecting', 'indexing', 'done'
                - current: current item number (1-based)
                - total: total items in this stage
                - detail: description of current item
        """
        self._progress_callback = progress_callback
        self._connect_all()
        self._introspect_all()
        self._resolve_reverse_references()
        self._build_vector_index()
        self._generate_overview()
        self._progress_callback = None

    def refresh(self, progress_callback: Optional[Callable[[str, int, int, str], None]] = None) -> None:
        """Clear caches and re-introspect all schemas.

        Use this when database schemas have changed and you need fresh metadata.
        """
        # Clear all caches
        self.metadata_cache.clear()
        self._embeddings = None
        self._embedding_keys = []
        self._overview = None

        # Re-initialize (connections are preserved)
        self._progress_callback = progress_callback
        self._introspect_all()
        self._resolve_reverse_references()
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
        connector = FileConnector.from_config(db_name, db_config)
        self.file_connections[db_name] = connector

    def _connect_sql(self, db_name: str, db_config: DatabaseConfig) -> None:
        """Connect to a SQL database via SQLAlchemy."""
        connection_uri = db_config.get_connection_uri()
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

    def _build_vector_index(self) -> None:
        """Build vector embeddings for all tables."""
        if not self.metadata_cache:
            return

        # Load embedding model
        self._emit_progress("indexing", 1, 2, "loading embedding model")
        self._model = SentenceTransformer(self.EMBEDDING_MODEL)

        # Generate texts for embedding
        texts = []
        self._embedding_keys = []

        for full_name, table_meta in self.metadata_cache.items():
            texts.append(table_meta.to_embedding_text())
            self._embedding_keys.append(full_name)

        # Generate embeddings
        self._emit_progress("indexing", 2, 2, f"vectorizing {len(texts)} tables")
        self._embeddings = self._model.encode(texts, convert_to_numpy=True)

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

    def get_brief_summary(self) -> str:
        """Return a brief summary of databases without listing all tables.

        Use this instead of get_overview() when table count is potentially large.
        The LLM can use discovery tools (find_relevant_tables, get_table_schema)
        to explore specific tables as needed.
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

    def find_relevant_tables(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Find tables relevant to a natural language query using vector search.

        Args:
            query: Natural language description of what data is needed
            top_k: Maximum number of results to return

        Returns:
            List of dicts with table, database, relevance score, and summary
        """
        if self._model is None or self._embeddings is None:
            return []

        # Embed the query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Compute cosine similarity
        # embeddings are already normalized by sentence-transformers
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            full_name = self._embedding_keys[idx]
            table_meta = self.metadata_cache[full_name]
            relevance = float(similarities[idx])

            # Generate a brief summary
            col_names = [c.name for c in table_meta.columns[:5]]
            if len(table_meta.columns) > 5:
                col_names.append("...")
            summary = f"{table_meta.name}: {', '.join(col_names)} ({table_meta.row_count:,} rows)"

            results.append({
                "table": table_meta.name,
                "database": table_meta.database,
                "full_name": full_name,
                "relevance": round(relevance, 3),
                "summary": summary,
                # Database type - tells LLM what query semantics to use
                "database_type": table_meta.database_type,  # e.g., "postgresql", "mongodb"
                "is_nosql": table_meta.database_type in ("mongodb", "elasticsearch", "dynamodb", "cosmosdb", "firestore", "cassandra"),
            })

        return results

    def list_tables(self) -> list[str]:
        """Return list of all table full names."""
        return list(self.metadata_cache.keys())

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
