# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pluggable vector store backends for document embedding storage and search.

This module provides an abstracted vector store interface with two implementations:
- DuckDBVectorStore: Persistent storage with HNSW indexing via DuckDB VSS
- NumpyVectorStore: In-memory store with linear search

The DuckDB backend is preferred for persistence and performance with larger
document collections.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import hashlib
import json
import logging

import numpy as np

from constat.discovery.models import DocumentChunk, Entity, ChunkEntity, EnrichedChunk

logger = logging.getLogger(__name__)


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends.

    Vector stores handle embedding storage and similarity search for document chunks.
    Implementations must support:
    - Adding chunks with their embeddings
    - Searching by query embedding
    - Clearing all stored data
    - Counting stored chunks
    """

    # Embedding dimension for BAAI/bge-large-en-v1.5
    EMBEDDING_DIM = 1024

    @abstractmethod
    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add document chunks with their embeddings to the store.

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: numpy array of shape (n_chunks, embedding_dim)
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search for similar chunks by query embedding.

        Args:
            query_embedding: Query embedding vector of shape (embedding_dim,) or (1, embedding_dim)
            limit: Maximum number of results to return

        Returns:
            List of (chunk_id, similarity_score, DocumentChunk) tuples,
            ordered by descending similarity
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored chunks and embeddings."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored chunks."""
        pass

    def get_chunks(self) -> list[DocumentChunk]:
        """Get all stored chunks.

        Returns:
            List of all DocumentChunk objects in the store
        """
        return []


class NumpyVectorStore(VectorStoreBackend):
    """In-memory vector store using numpy arrays.

    Uses brute-force cosine similarity search (O(n) complexity).

    Best for:
    - Small document collections (< 1000 chunks)
    - Development and testing
    - Ephemeral/stateless deployments
    """

    def __init__(self):
        self._chunks: list[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._chunk_ids: list[str] = []

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        session_id: str | None = None,
        project_id: str | None = None,
        config_hash: str | None = None,
    ) -> None:
        """Add chunks with embeddings to in-memory storage."""
        if len(chunks) == 0:
            return

        # Generate IDs and store chunks
        new_ids = [self._generate_chunk_id(c) for c in chunks]
        self._chunks.extend(chunks)
        self._chunk_ids.extend(new_ids)

        # Stack embeddings
        if self._embeddings is None:
            self._embeddings = embeddings.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings])

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search using cosine similarity."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []

        # Ensure query is 1D
        query = query_embedding.flatten()

        # Compute cosine similarity
        # Normalize vectors
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        emb_norms = self._embeddings / (
            np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-10
        )

        similarities = np.dot(emb_norms, query_norm)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[idx]
            similarity = float(similarities[idx])
            chunk = self._chunks[idx]
            results.append((chunk_id, similarity, chunk))

        return results

    def clear(self) -> None:
        """Clear all stored data."""
        self._chunks = []
        self._embeddings = None
        self._chunk_ids = []

    def count(self) -> int:
        """Return number of stored chunks."""
        return len(self._chunks)

    def get_chunks(self) -> list[DocumentChunk]:
        """Get all stored chunks."""
        return self._chunks.copy()


class DuckDBVectorStore(VectorStoreBackend):
    """Persistent vector store using DuckDB with VSS extension.

    Uses DuckDB's vector similarity search extension for efficient
    cosine similarity queries. Data is persisted to a .duckdb file.

    Best for:
    - Large document collections (> 1000 chunks)
    - Production deployments requiring persistence
    - Frequent restarts where index rebuild is costly

    The VSS extension provides:
    - array_cosine_similarity() for similarity computation
    - HNSW indexing for O(log n) approximate search (on supported versions)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize DuckDB vector store.

        Args:
            db_path: Path to DuckDB database file. If None, uses
                     ~/.constat/vectors.duckdb or CONSTAT_VECTOR_STORE_PATH env var
        """
        import os
        from constat.storage.duckdb_pool import ThreadLocalDuckDB

        # Determine database path (env var allows test isolation)
        if db_path:
            self._db_path = Path(db_path).expanduser()
        elif os.environ.get("CONSTAT_VECTOR_STORE_PATH"):
            self._db_path = Path(os.environ["CONSTAT_VECTOR_STORE_PATH"])
        else:
            self._db_path = Path.cwd() / ".constat" / "vectors.duckdb"

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use thread-local connection pool for thread safety
        # Each thread gets its own connection to avoid heap corruption
        self._db = ThreadLocalDuckDB(
            str(self._db_path),
            init_sql=["INSTALL vss", "LOAD vss"],
        )
        self._init_schema()

    @property
    def _conn(self):
        """Get the thread-local connection (backwards compatibility)."""
        return self._db.conn

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        # VSS extension is loaded via init_sql in ThreadLocalDuckDB

        # Create embeddings table with FLOAT array for vectors
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                document_name VARCHAR NOT NULL,
                section VARCHAR,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{self.EMBEDDING_DIM}] NOT NULL,
                session_id VARCHAR,
                project_id VARCHAR,
                config_hash VARCHAR
            )
        """)

        # Create unified entities table - ALL known entities (schema, API, extracted)
        # This is the single source of truth for entity catalog
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS entities (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                type VARCHAR NOT NULL,
                source VARCHAR NOT NULL,
                parent_id VARCHAR,
                embedding FLOAT[{self.EMBEDDING_DIM}],
                metadata JSON,
                config_hash VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR,
                project_id VARCHAR
            )
        """)
        # type: table, column, api_endpoint, api_field, api_schema, extracted
        # source: schema, api, document

        # Create chunk_entities junction table (links document chunks to entities)
        # PK includes session_id so same (chunk_id, entity_id) can have both
        # NULL session_id links (global) and session-scoped links
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                mention_count INTEGER DEFAULT 1,
                confidence FLOAT DEFAULT 1.0,
                mention_text VARCHAR,
                session_id VARCHAR DEFAULT '__none__',
                project_id VARCHAR,
                PRIMARY KEY (chunk_id, entity_id, session_id)
            )
        """)

        # Migration: add new columns to existing tables if missing
        self._migrate_schema()

        # Create indexes for efficient lookups
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_parent ON entities(parent_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id)"
            )
        except Exception as e:
            logger.debug(f"Index creation skipped: {e}")

    def _migrate_schema(self) -> None:
        """Migrate existing tables to new schema if needed."""
        # DuckDB uses information_schema, not PRAGMA (which is SQLite)
        def get_column_names(table_name: str) -> set[str]:
            """Get column names for a table using DuckDB's information_schema."""
            try:
                result = self._conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                    [table_name]
                ).fetchall()
                return {row[0] for row in result}
            except Exception as e:
                logger.warning(f"Failed to get columns for {table_name}: {e}")
                return set()

        # Add session_id column to tables if missing (for multi-session isolation)
        for table in ['embeddings', 'entities', 'chunk_entities']:
            try:
                col_names = get_column_names(table)
                if col_names and 'session_id' not in col_names:
                    logger.debug(f"_migrate_schema: adding session_id column to {table}")
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN session_id VARCHAR"
                    )
            except Exception as e:
                logger.debug(f"_migrate_schema: failed to add session_id to {table}: {e}")

        # Add project_id column to tables if missing (for project-level filtering)
        for table in ['embeddings', 'entities', 'chunk_entities']:
            try:
                col_names = get_column_names(table)
                if col_names and 'project_id' not in col_names:
                    logger.debug(f"_migrate_schema: adding project_id column to {table}")
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN project_id VARCHAR"
                    )
            except Exception as e:
                logger.debug(f"_migrate_schema: failed to add project_id to {table}: {e}")

        # Add config_hash column to embeddings if missing (for cache invalidation)
        try:
            col_names = get_column_names('embeddings')
            if col_names and 'config_hash' not in col_names:
                logger.debug("_migrate_schema: adding config_hash column to embeddings")
                self._conn.execute(
                    "ALTER TABLE embeddings ADD COLUMN config_hash VARCHAR"
                )
        except Exception as e:
            logger.debug(f"_migrate_schema: failed to add config_hash to embeddings: {e}")

        # Add mention_text column to chunk_entities if missing
        try:
            col_names = get_column_names('chunk_entities')
            if col_names and 'mention_text' not in col_names:
                logger.debug("_migrate_schema: adding mention_text column to chunk_entities")
                self._conn.execute(
                    "ALTER TABLE chunk_entities ADD COLUMN mention_text VARCHAR"
                )
        except Exception as e:
            logger.debug(f"_migrate_schema: failed to add mention_text to chunk_entities: {e}")

        # Add new columns to entities table for unified catalog
        try:
            col_names = get_column_names('entities')

            if col_names and 'source' not in col_names:
                logger.debug("_migrate_schema: adding source column to entities")
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN source VARCHAR DEFAULT 'document'"
                )
            if col_names and 'parent_id' not in col_names:
                logger.debug("_migrate_schema: adding parent_id column to entities")
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN parent_id VARCHAR"
                )
            if col_names and 'config_hash' not in col_names:
                logger.debug("_migrate_schema: adding config_hash column to entities")
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN config_hash VARCHAR"
                )
        except Exception as e:
            logger.warning(f"_migrate_schema: failed to add columns to entities: {e}")

        # Drop old tables if they exist (migrating to unified entities)
        try:
            self._conn.execute("DROP TABLE IF EXISTS table_embeddings")
            self._conn.execute("DROP TABLE IF EXISTS api_embeddings")
        except Exception as e:
            logger.debug(f"_migrate_schema: failed to drop old tables: {e}")

        # Migrate chunk_entities to new PK (includes session_id)
        # Check if session_id has a default - if not, we need to recreate the table
        try:
            result = self._conn.execute("""
                SELECT column_default
                FROM information_schema.columns
                WHERE table_name = 'chunk_entities' AND column_name = 'session_id'
            """).fetchone()
            needs_pk_migration = result is None or result[0] != "'__none__'"

            if needs_pk_migration:
                logger.info("_migrate_schema: migrating chunk_entities to new PK schema")
                # Drop and recreate - data will be regenerated during session startup
                self._conn.execute("DROP TABLE IF EXISTS chunk_entities")
                self._conn.execute("""
                    CREATE TABLE chunk_entities (
                        chunk_id VARCHAR NOT NULL,
                        entity_id VARCHAR NOT NULL,
                        mention_count INTEGER DEFAULT 1,
                        confidence FLOAT DEFAULT 1.0,
                        mention_text VARCHAR,
                        session_id VARCHAR DEFAULT '__none__',
                        project_id VARCHAR,
                        PRIMARY KEY (chunk_id, entity_id, session_id)
                    )
                """)
                logger.info("_migrate_schema: chunk_entities table recreated with new PK")
        except Exception as e:
            logger.debug(f"_migrate_schema: chunk_entities PK migration check failed: {e}")

    def clear_session_data(self, session_id: str) -> None:
        """Remove all data for a specific session.

        Args:
            session_id: Session ID to clear
        """
        # Count rows before deletion
        emb_count = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        ent_count = self._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        link_count = self._conn.execute(
            "SELECT COUNT(*) FROM chunk_entities WHERE session_id = ?", [session_id]
        ).fetchone()[0]
        logger.debug(f"clear_session_data({session_id}): found {emb_count} embeddings, {ent_count} entities, {link_count} links")

        # Delete in order (links first)
        self._conn.execute("DELETE FROM chunk_entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM embeddings WHERE session_id = ?", [session_id])

        logger.debug(f"clear_session_data({session_id}): deleted session data")

    def delete_document(self, document_name: str, session_id: str | None = None) -> int:
        """Delete a document and its associated entities.

        Args:
            document_name: Name of the document to delete
            session_id: Optional session ID (if None, deletes from all sessions)

        Returns:
            Number of chunks deleted
        """
        # Get chunk IDs for this document
        if session_id:
            chunk_ids = self._conn.execute(
                "SELECT chunk_id FROM embeddings WHERE document_name = ? AND session_id = ?",
                [document_name, session_id]
            ).fetchall()
        else:
            chunk_ids = self._conn.execute(
                "SELECT chunk_id FROM embeddings WHERE document_name = ?",
                [document_name]
            ).fetchall()

        chunk_ids = [row[0] for row in chunk_ids]

        if not chunk_ids:
            return 0

        # Delete chunk-entity links for these chunks
        placeholders = ",".join(["?" for _ in chunk_ids])
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )

        # Delete orphaned entities (entities with no remaining chunk links)
        # Only for session-specific entities
        if session_id:
            self._conn.execute("""
                DELETE FROM entities
                WHERE session_id = ?
                AND id NOT IN (SELECT DISTINCT entity_id FROM chunk_entities)
            """, [session_id])

        # Delete the document chunks
        if session_id:
            self._conn.execute(
                "DELETE FROM embeddings WHERE document_name = ? AND session_id = ?",
                [document_name, session_id]
            )
        else:
            self._conn.execute(
                "DELETE FROM embeddings WHERE document_name = ?",
                [document_name]
            )

        logger.debug(f"delete_document({document_name}, {session_id}): deleted {len(chunk_ids)} chunks")
        return len(chunk_ids)

    def get_project_config_hash(self, project_id: str) -> str | None:
        """Get the config hash for a project's embeddings.

        Args:
            project_id: Project ID to check

        Returns:
            Config hash string or None if no embeddings exist
        """
        # Debug: check what embeddings exist for this project
        count = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE project_id = ?",
            [project_id],
        ).fetchone()[0]
        logger.debug(f"get_project_config_hash({project_id}): found {count} embeddings")

        result = self._conn.execute(
            "SELECT config_hash FROM embeddings WHERE project_id = ? AND config_hash IS NOT NULL LIMIT 1",
            [project_id],
        ).fetchone()
        hash_val = result[0] if result else None
        logger.debug(f"get_project_config_hash({project_id}): hash={hash_val}")
        return hash_val

    def clear_project_embeddings(self, project_id: str) -> int:
        """Clear all embeddings for a project.

        Args:
            project_id: Project ID to clear

        Returns:
            Number of embeddings deleted
        """
        # Get chunk IDs first for clearing related entities
        chunk_ids = self._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE project_id = ?",
            [project_id]
        ).fetchall()
        chunk_ids = [row[0] for row in chunk_ids]

        if chunk_ids:
            # Clear chunk-entity links
            placeholders = ",".join(["?" for _ in chunk_ids])
            self._conn.execute(
                f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )

        # Clear embeddings
        result = self._conn.execute(
            "DELETE FROM embeddings WHERE project_id = ?",
            [project_id]
        )

        # Clear project entities
        self._conn.execute(
            "DELETE FROM entities WHERE project_id = ?",
            [project_id]
        )

        count = len(chunk_ids)
        logger.debug(f"clear_project_embeddings({project_id}): deleted {count} embeddings")
        return count

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        session_id: str | None = None,
        project_id: str | None = None,
        config_hash: str | None = None,
    ) -> None:
        """Add chunks with embeddings to DuckDB.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
            session_id: Optional session ID for documents added during a session
            project_id: Optional project ID for documents belonging to a project
            config_hash: Optional config hash for cache invalidation
        """
        if len(chunks) == 0:
            return

        doc_names = set(c.document_name for c in chunks)
        print(f"[ADD_CHUNKS] docs={doc_names}, session_id={session_id}, project_id={project_id}")

        # Check which documents already exist - skip re-indexing to preserve project_id
        existing_docs = set()
        for doc_name in doc_names:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
                [doc_name],
            ).fetchone()[0]
            print(f"[ADD_CHUNKS] {doc_name}: existing chunks = {count}")
            if count > 0:
                existing_docs.add(doc_name)
                print(f"[ADD_CHUNKS] SKIPPING {doc_name}")

        # Filter to only new documents
        new_chunks = [c for c in chunks if c.document_name not in existing_docs]
        if not new_chunks:
            logger.debug(f"add_chunks: all documents already indexed, nothing to add")
            return

        # Prepare data for insertion
        records = []
        for i, chunk in enumerate(new_chunks):
            chunk_id = self._generate_chunk_id(chunk)
            # Find the corresponding embedding
            original_idx = chunks.index(chunk)
            embedding = embeddings[original_idx].tolist()
            records.append((
                chunk_id,
                chunk.document_name,
                chunk.section,
                chunk.chunk_index,
                chunk.content,
                embedding,
                session_id,
                project_id,
                config_hash,
            ))

        # Simple INSERT for new documents only
        self._conn.executemany(
            """
            INSERT INTO embeddings
            (chunk_id, document_name, section, chunk_index, content, embedding, session_id, project_id, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search using DuckDB's array_cosine_similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            List of (chunk_id, similarity, DocumentChunk) tuples
        """
        # Ensure query is 1D list
        query = query_embedding.flatten().tolist()

        # Build filter for base + project + session
        # base: project_id IS NULL or '__base__' (always included)
        # project: project_id IN (loaded_projects)
        # session: session_id = current_session_id
        filter_conditions = ["(project_id IS NULL)", "(project_id = '__base__')"]
        params: list = [query]

        if project_ids:
            # Filter out '__base__' since it's already included above
            filtered_project_ids = [p for p in project_ids if p != '__base__']
            if filtered_project_ids:
                placeholders = ",".join(["?" for _ in filtered_project_ids])
                filter_conditions.append(f"project_id IN ({placeholders})")
                params.extend(filtered_project_ids)

        if session_id:
            filter_conditions.append("session_id = ?")
            params.append(session_id)

        where_clause = " OR ".join(filter_conditions)
        params.append(limit)

        # Query with cosine similarity
        result = self._conn.execute(
            f"""
            SELECT
                chunk_id,
                document_name,
                section,
                chunk_index,
                content,
                array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) as similarity
            FROM embeddings
            WHERE ({where_clause})
            ORDER BY similarity DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        # Convert to output format
        results = []
        for row in result:
            chunk_id, doc_name, section, chunk_idx, content, similarity = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
            )
            results.append((chunk_id, float(similarity), chunk))

        return results

    def clear(self) -> None:
        """Clear all stored data."""
        self._conn.execute("DELETE FROM embeddings")

    def clear_chunk_entity_links(self, session_id: str | None = None) -> None:
        """Clear chunk-entity links (but keep entities).

        Args:
            session_id: If provided, only clear links for this session.
                       If None, only clear global links (session_id = '__none__').
                       Session-scoped links are always preserved when clearing global links.
        """
        if session_id:
            # Clear links for a specific session
            self._conn.execute("DELETE FROM chunk_entities WHERE session_id = ?", [session_id])
        else:
            # Only clear global links, preserve session-scoped links
            self._conn.execute("DELETE FROM chunk_entities WHERE session_id = '__none__'")

    def count(self) -> int:
        """Return number of stored chunks."""
        result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    def get_chunks(self) -> list[DocumentChunk]:
        """Get all stored chunks."""
        result = self._conn.execute(
            "SELECT document_name, content, section, chunk_index FROM embeddings"
        ).fetchall()

        return [
            DocumentChunk(
                document_name=row[0],
                content=row[1],
                section=row[2],
                chunk_index=row[3],
            )
            for row in result
        ]

    def delete_by_document(self, document_name: str) -> int:
        """Delete all chunks for a specific document.

        Args:
            document_name: Name of the document to delete chunks for

        Returns:
            Number of chunks deleted
        """
        # Count before deletion
        count_before = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchone()[0]

        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        )

        return count_before

    # =========================================================================
    # Entity Methods
    # =========================================================================

    def add_entity(
        self,
        entity: Entity,
        embedding: Optional[np.ndarray] = None,
        source: str = "document",
    ) -> None:
        """Add a single entity to the store.

        Args:
            entity: Entity object to add
            embedding: Optional embedding for semantic entity search
            source: Source category ('document', 'schema', 'api')
        """
        emb_list = embedding.tolist() if embedding is not None else None
        metadata_json = json.dumps(entity.metadata) if entity.metadata else None

        self._conn.execute(
            """
            INSERT OR REPLACE INTO entities (id, name, type, source, embedding, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                entity.id,
                entity.name,
                entity.type,
                source,
                emb_list,
                metadata_json,
                entity.created_at,
            ],
        )

    def add_entities(
        self,
        entities: list[Entity],
        embeddings: Optional[np.ndarray] = None,
        source: str = "document",
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Add multiple entities to the store.

        Args:
            entities: List of Entity objects to add
            embeddings: Optional embeddings array of shape (n_entities, embedding_dim)
            source: Source category ('document', 'schema', 'api')
            session_id: Optional session ID for entities added during a session
            project_id: Optional project ID for entities belonging to a project
        """
        if not entities:
            return

        records = []
        for i, entity in enumerate(entities):
            emb_list = embeddings[i].tolist() if embeddings is not None else None
            metadata_json = json.dumps(entity.metadata) if entity.metadata else None
            records.append((
                entity.id,
                entity.name,
                entity.type,
                source,
                emb_list,
                metadata_json,
                entity.created_at,
                session_id,
                project_id,
            ))

        self._conn.executemany(
            """
            INSERT INTO entities (id, name, type, source, embedding, metadata, created_at, session_id, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO NOTHING
            """,
            records,
        )

    def link_chunk_entities(
        self,
        links: list[ChunkEntity],
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Create links between chunks and entities.

        Args:
            links: List of ChunkEntity objects defining the relationships
            session_id: Optional session ID for links added during a session
            project_id: Optional project ID for links belonging to a project
        """
        if not links:
            return

        # Use '__none__' instead of NULL for session_id (part of PK)
        effective_session_id = session_id if session_id else '__none__'

        # Deduplicate links by (chunk_id, entity_id)
        seen = set()
        unique_records = []
        for l in links:
            key = (l.chunk_id, l.entity_id)
            if key not in seen:
                seen.add(key)
                unique_records.append(
                    (l.chunk_id, l.entity_id, l.mention_count, l.confidence, l.mention_text, effective_session_id, project_id)
                )

        # Log what we're inserting
        logger.debug(f"link_chunk_entities: inserting {len(unique_records)} links with session_id={effective_session_id}")

        # Use INSERT OR IGNORE to skip duplicates efficiently
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO chunk_entities
                (chunk_id, entity_id, mention_count, confidence, mention_text, session_id, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            unique_records,
        )

    def get_entities_for_chunk(
        self,
        chunk_id: str,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[Entity]:
        """Get all entities associated with a chunk.

        Args:
            chunk_id: The chunk identifier
            project_ids: List of project IDs to include (filters entities)
            session_id: Session ID (required - NER results are scoped to session)

        Returns:
            List of Entity objects linked to this chunk
        """
        # NER results (chunk_entities) are scoped to session_id
        # Entities can come from base + project + session
        entity_filter = ["(e.project_id IS NULL AND e.session_id IS NULL)"]
        params: list = [chunk_id]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            entity_filter.append(f"e.project_id IN ({placeholders})")
            params.extend(project_ids)

        if session_id:
            entity_filter.append("e.session_id = ?")
            params.append(session_id)
            # Filter chunk_entities by session_id (NER results are session-scoped)
            link_filter = "ce.session_id = ?"
            params.append(session_id)
        else:
            link_filter = "1=1"

        entity_where = " OR ".join(entity_filter)

        result = self._conn.execute(
            f"""
            SELECT e.id, e.name, e.type, e.metadata, e.created_at
            FROM entities e
            JOIN chunk_entities ce ON e.id = ce.entity_id
            WHERE ce.chunk_id = ? AND ({entity_where}) AND ({link_filter})
            ORDER BY ce.mention_count DESC, ce.confidence DESC
            """,
            params,
        ).fetchall()

        entities = []
        for row in result:
            entity_id, name, etype, metadata_json, created_at = row
            metadata = json.loads(metadata_json) if metadata_json else {}
            entities.append(Entity(
                id=entity_id,
                name=name,
                type=etype,
                metadata=metadata,
                created_at=created_at,
            ))

        return entities

    def get_chunks_for_entity(
        self,
        entity_id: str,
        limit: int = 10,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[tuple[str, DocumentChunk, int, float]]:
        """Get chunks that mention an entity.

        Args:
            entity_id: The entity identifier
            limit: Maximum number of chunks to return
            project_ids: List of project IDs to include (filters embeddings)
            session_id: Session ID (required - NER results are scoped to session)

        Returns:
            List of (chunk_id, DocumentChunk, mention_count, confidence) tuples
            ordered by mention_count and confidence
        """
        # Embeddings can come from base + project + session
        emb_filter = ["(em.project_id IS NULL AND em.session_id IS NULL)"]
        params: list = [entity_id]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            emb_filter.append(f"em.project_id IN ({placeholders})")
            params.extend(project_ids)

        if session_id:
            emb_filter.append("em.session_id = ?")
            params.append(session_id)
            # Filter chunk_entities by session_id (NER results are session-scoped)
            link_filter = "ce.session_id = ?"
            params.append(session_id)
        else:
            link_filter = "1=1"

        emb_where = " OR ".join(emb_filter)
        params.append(limit)

        result = self._conn.execute(
            f"""
            SELECT
                ce.chunk_id,
                em.document_name,
                em.content,
                em.section,
                em.chunk_index,
                ce.mention_count,
                ce.confidence
            FROM chunk_entities ce
            JOIN embeddings em ON ce.chunk_id = em.chunk_id
            WHERE ce.entity_id = ? AND ({emb_where}) AND ({link_filter})
            ORDER BY ce.mention_count DESC, ce.confidence DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        chunks = []
        for row in result:
            chunk_id, doc_name, content, section, chunk_idx, mention_count, confidence = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
            )
            chunks.append((chunk_id, chunk, mention_count, confidence))

        return chunks

    def find_entity_by_name(
        self,
        name: str,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> Optional[Entity]:
        """Find an entity by its name (case-insensitive).

        Args:
            name: Entity name to search for
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            Entity if found, None otherwise
        """
        # Build filter for base + project + session
        filter_conditions = ["(project_id IS NULL AND session_id IS NULL)"]
        params: list = [name]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            filter_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        if session_id:
            filter_conditions.append("session_id = ?")
            params.append(session_id)

        where_clause = " OR ".join(filter_conditions)

        result = self._conn.execute(
            f"""
            SELECT id, name, type, metadata, created_at
            FROM entities
            WHERE LOWER(name) = LOWER(?) AND ({where_clause})
            LIMIT 1
            """,
            params,
        ).fetchone()

        if not result:
            return None

        entity_id, name, etype, metadata_json, created_at = result
        metadata = json.loads(metadata_json) if metadata_json else {}

        return Entity(
            id=entity_id,
            name=name,
            type=etype,
            metadata=metadata,
            created_at=created_at,
        )

    def search_enriched(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[EnrichedChunk]:
        """Search for chunks and include associated entities.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            List of EnrichedChunk objects with entities
        """
        # First do the regular search with filtering
        results = self.search(query_embedding, limit, project_ids, session_id)

        # Enrich with entities (using same filter)
        enriched = []
        for chunk_id, score, chunk in results:
            entities = self.get_entities_for_chunk(chunk_id, project_ids, session_id)
            enriched.append(EnrichedChunk(
                chunk=chunk,
                score=score,
                entities=entities,
            ))

        return enriched

    def clear_entities(self, source: Optional[str] = "document") -> None:
        """Clear entities and chunk-entity links.

        Args:
            source: If specified, only clear entities with this source.
                   Default is 'document' to preserve schema/api entities.
                   Pass None to clear ALL entities.
        """
        if source:
            # Only delete chunk_entities for document-sourced entities
            self._conn.execute("""
                DELETE FROM chunk_entities
                WHERE entity_id IN (SELECT id FROM entities WHERE source = ?)
            """, [source])
            self._conn.execute("DELETE FROM entities WHERE source = ?", [source])
        else:
            self._conn.execute("DELETE FROM chunk_entities")
            self._conn.execute("DELETE FROM entities")

    def count_entities(self) -> int:
        """Return number of stored entities."""
        result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return result[0] if result else 0

    # ========== Unified Catalog Entity Methods ==========

    def add_catalog_entities(
        self,
        entities: list[dict],
        embeddings: np.ndarray,
        source: str,
        config_hash: str,
    ) -> None:
        """Add catalog entities (schema tables/columns or API endpoints/fields) to the store.

        Args:
            entities: List of entity dicts with:
                - id: unique identifier (e.g., "sales.customers" or "catfacts.GET /breeds")
                - name: display name
                - type: entity type (table, column, api_endpoint, api_field, api_schema)
                - parent_id: optional parent entity id (e.g., table id for columns)
                - metadata: additional entity-specific data (JSON-serializable dict)
            embeddings: numpy array of shape (n_entities, embedding_dim)
            source: source category ('schema' or 'api')
            config_hash: Hash of config for cache invalidation
        """
        if not entities:
            return

        # Deduplicate entities by normalized ID (case-insensitive, keep last occurrence)
        # Use dict to ensure unique IDs - key is normalized_id, value is the record tuple
        unique_records: dict[str, tuple] = {}
        for i, e in enumerate(entities):
            normalized_id = e["id"].lower()
            metadata_json = json.dumps(e.get("metadata", {})) if e.get("metadata") else None
            unique_records[normalized_id] = (
                normalized_id,
                e["name"],
                e["type"],
                source,
                e.get("parent_id").lower() if e.get("parent_id") else None,
                embeddings[i].tolist() if embeddings is not None else None,
                metadata_json,
                config_hash,
            )

        entity_ids = list(unique_records.keys())
        records = list(unique_records.values())

        # Delete existing entities with these IDs first to avoid conflict issues
        # DuckDB's executemany with ON CONFLICT doesn't handle batch duplicates well
        # Use LOWER() for case-insensitive matching since IDs are normalized to lowercase
        if entity_ids:
            placeholders = ",".join(["?" for _ in entity_ids])
            self._conn.execute(
                f"DELETE FROM entities WHERE LOWER(id) IN ({placeholders})",
                entity_ids,
            )

        self._conn.executemany(
            """
            INSERT INTO entities
            (id, name, type, source, parent_id, embedding, metadata, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    def search_catalog_entities(
        self,
        query_embedding: np.ndarray,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        limit: int = 5,
        min_similarity: float = 0.0,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """Search for relevant catalog entities by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            source: Filter by source ('schema', 'api', 'document') or None for all
            entity_type: Filter by type ('table', 'column', 'api_endpoint', etc.) or None for all
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            List of dicts with id, name, type, source, parent_id, metadata, similarity
        """
        query = query_embedding.flatten().tolist()

        # Build filter for base + project + session
        scope_conditions = ["(project_id IS NULL AND session_id IS NULL)"]
        scope_params: list = []

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            scope_conditions.append(f"project_id IN ({placeholders})")
            scope_params.extend(project_ids)

        if session_id:
            scope_conditions.append("session_id = ?")
            scope_params.append(session_id)

        scope_clause = " OR ".join(scope_conditions)

        # Build WHERE clause based on filters
        conditions = ["embedding IS NOT NULL", f"({scope_clause})"]
        params: list = [query]
        params.extend(scope_params)

        if source:
            conditions.append("source = ?")
            params.append(source)
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)

        where_clause = " AND ".join(conditions)
        params.extend([query, min_similarity, limit])

        result = self._conn.execute(
            f"""
            SELECT
                id,
                name,
                type,
                source,
                parent_id,
                metadata,
                array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) as similarity
            FROM entities
            WHERE {where_clause}
              AND array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "source": row[3],
                "parent_id": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
                "similarity": float(row[6]),
            }
            for row in result
        ]

    def get_catalog_config_hash(self, source: str) -> Optional[str]:
        """Get the config hash for cached catalog entities by source.

        Args:
            source: Source category ('schema' or 'api')

        Returns:
            Config hash string or None if not found
        """
        result = self._conn.execute(
            "SELECT config_hash FROM entities WHERE source = ? AND config_hash IS NOT NULL LIMIT 1",
            [source],
        ).fetchone()
        return result[0] if result else None

    def clear_catalog_entities(self, source: str) -> None:
        """Clear all catalog entities for a specific source.

        Args:
            source: Source category ('schema' or 'api')
        """
        self._conn.execute("DELETE FROM entities WHERE source = ?", [source])

    def count_catalog_entities(self, source: Optional[str] = None) -> int:
        """Return number of stored catalog entities.

        Args:
            source: Optional source filter ('schema', 'api', 'document')

        Returns:
            Count of entities
        """
        if source:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM entities WHERE source = ?", [source]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return result[0] if result else 0

    def get_entity_names_by_source(
        self,
        source: Optional[str] = None,
        entity_type: Optional[str] = None,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        """Get all entity names, optionally filtered by source and type.

        Args:
            source: Optional source filter ('schema', 'api', 'document')
            entity_type: Optional type filter ('table', 'column', 'api_endpoint', etc.)
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            List of entity names
        """
        # Build filter for base + project + session
        scope_conditions = ["(project_id IS NULL AND session_id IS NULL)"]
        params: list = []

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            scope_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        if session_id:
            scope_conditions.append("session_id = ?")
            params.append(session_id)

        scope_clause = " OR ".join(scope_conditions)
        conditions = [f"({scope_clause})"]

        if source:
            conditions.append("source = ?")
            params.append(source)
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)

        where_clause = " WHERE " + " AND ".join(conditions)

        result = self._conn.execute(
            f"SELECT DISTINCT name FROM entities{where_clause}",
            params,
        ).fetchall()

        return [row[0] for row in result]

    def get_entities_by_parent(
        self,
        parent_id: str,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """Get all child entities for a given parent.

        Args:
            parent_id: Parent entity ID
            project_ids: List of project IDs to include (None means no project filter)
            session_id: Session ID to include (None means no session filter)

        Returns:
            List of child entity dicts
        """
        # Build filter for base + project + session
        scope_conditions = ["(project_id IS NULL AND session_id IS NULL)"]
        params: list = [parent_id]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            scope_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        if session_id:
            scope_conditions.append("session_id = ?")
            params.append(session_id)

        scope_clause = " OR ".join(scope_conditions)

        result = self._conn.execute(
            f"""
            SELECT id, name, type, source, parent_id, metadata
            FROM entities
            WHERE parent_id = ? AND ({scope_clause})
            ORDER BY name
            """,
            params,
        ).fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "source": row[3],
                "parent_id": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
            }
            for row in result
        ]

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "_conn"):
            self._conn.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def create_vector_store(
    backend: str = "duckdb",
    db_path: Optional[str] = None,
) -> VectorStoreBackend:
    """Factory function to create a vector store backend.

    Args:
        backend: Backend type - "duckdb" or "numpy"
        db_path: Path to DuckDB database file (only for duckdb backend)

    Returns:
        VectorStoreBackend instance

    Raises:
        ImportError: If "duckdb" backend is requested but duckdb is not installed
        ValueError: If unknown backend type is specified
    """
    if backend == "duckdb":
        return DuckDBVectorStore(db_path=db_path)
    elif backend == "numpy":
        return NumpyVectorStore()
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
