"""Database schema introspection, caching, and vector search."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from constat.core.config import Config, DatabaseConfig, DatabaseCredentials


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
        self.connections: dict[str, Engine] = {}
        self.metadata_cache: dict[str, TableMetadata] = {}  # key: "db.table"

        # Vector index components
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_keys: list[str] = []  # Maps index → table full_name
        self._model: Optional[SentenceTransformer] = None

        # Cached overview string
        self._overview: Optional[str] = None

    def initialize(self) -> None:
        """Connect to databases, introspect schemas, build vector index."""
        self._connect_all()
        self._introspect_all()
        self._resolve_reverse_references()
        self._build_vector_index()
        self._generate_overview()

    def _connect_all(self) -> None:
        """Establish connections to all configured databases."""
        for db_name, db_config in self.config.databases.items():
            # Get connection URI with credentials applied
            connection_uri = db_config.get_connection_uri()

            engine = create_engine(connection_uri)
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.connections[db_name] = engine

    def _introspect_all(self) -> None:
        """Introspect all tables in all databases."""
        for db_name, engine in self.connections.items():
            inspector = inspect(engine)

            for table_name in inspector.get_table_names():
                table_meta = self._introspect_table(db_name, engine, inspector, table_name)
                self.metadata_cache[table_meta.full_name] = table_meta

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

        return TableMetadata(
            database=db_name,
            name=table_name,
            comment=table_comment,
            columns=columns,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            row_count=row_count,
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
        self._model = SentenceTransformer(self.EMBEDDING_MODEL)

        # Generate texts for embedding
        texts = []
        self._embedding_keys = []

        for full_name, table_meta in self.metadata_cache.items():
            texts.append(table_meta.to_embedding_text())
            self._embedding_keys.append(full_name)

        # Generate embeddings
        self._embeddings = self._model.encode(texts, convert_to_numpy=True)

    def _generate_overview(self) -> None:
        """Generate token-optimized overview for system prompt."""
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
            table_names = ", ".join(t.name for t in sorted(tables, key=lambda t: t.name))
            total_rows = sum(t.row_count for t in tables)

            # Include database description if available
            if db_name in db_descriptions:
                lines.append(f"  {db_name}: {db_descriptions[db_name]}")
                lines.append(f"    Tables: {table_names} ({len(tables)} tables, ~{total_rows:,} rows)")
            else:
                lines.append(f"  {db_name}: {table_names} ({len(tables)} tables, ~{total_rows:,} rows)")

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
        """Return token-optimized schema overview for system prompt."""
        if self._overview is None:
            self._generate_overview()
        return self._overview or ""

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
            })

        return results

    def list_tables(self) -> list[str]:
        """Return list of all table full names."""
        return list(self.metadata_cache.keys())

    def get_connection(self, database: str) -> Engine:
        """Get SQLAlchemy engine for a database."""
        if database not in self.connections:
            raise KeyError(f"Database not found: {database}")
        return self.connections[database]
