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

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from constat.discovery.models import DocumentChunk, Entity, ChunkEntity, EnrichedChunk, GlossaryTerm

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
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
        session_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        """Add document chunks with their embeddings to the store.

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            source: Resource type - 'schema', 'api', or 'document'
            session_id: Optional session ID for session-scoped chunks
            domain_id: Optional domain ID for domain-scoped chunks
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

    @staticmethod
    def _generate_chunk_id(chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
        session_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        """Add chunks with embeddings to in-memory storage."""
        if len(chunks) == 0:
            return

        # Set source on chunks
        for chunk in chunks:
            chunk.source = source

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
        domain_ids: list[str] | None = None,
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

    def get_all_chunk_ids(self, session_id: str | None = None) -> list[str]:
        """Get all chunk IDs, optionally filtered by session."""
        return list(self._chunk_ids)


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
        # source: resource type ('schema', 'api', 'document')
        # chunk_type: granular type ('db_table', 'db_column', 'api_endpoint', 'document', etc.)
        # domain_id: source identifier ('__base__' for config, domain filename for domains)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                document_name VARCHAR NOT NULL,
                source VARCHAR NOT NULL DEFAULT 'document',
                chunk_type VARCHAR NOT NULL DEFAULT 'document',
                section VARCHAR,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{self.EMBEDDING_DIM}] NOT NULL,
                session_id VARCHAR,
                domain_id VARCHAR
            )
        """)

        # Create entities table for NER-extracted entities.
        # Entities are session-scoped (rebuilt when session starts or domains change)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                display_name VARCHAR NOT NULL,
                semantic_type VARCHAR NOT NULL,
                ner_type VARCHAR,
                session_id VARCHAR NOT NULL,
                domain_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # semantic_type: CONCEPT, ATTRIBUTE, ACTION, TERM (linguistic role)
        # ner_type: ORG, PERSON, PRODUCT, GPE, EVENT or NULL (spaCy NER type)

        # Create chunk_entities junction table (links chunks to NER entities)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                PRIMARY KEY (chunk_id, entity_id)
            )
        """)

        # Create source_hashes table for config hashes per source (one row per source)
        # Used for cache invalidation - avoids storing hash on every chunk
        # Three hash types: db (databases), api (APIs), doc (documents)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS source_hashes (
                source_id VARCHAR PRIMARY KEY,
                db_hash VARCHAR,
                api_hash VARCHAR,
                doc_hash VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create resource_hashes table for fine-grained cache invalidation
        # Tracks individual resources (databases, APIs, documents) within a source
        # Enables incremental updates: only rebuild chunks for changed resources
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_hashes (
                resource_id VARCHAR PRIMARY KEY,
                resource_type VARCHAR NOT NULL,
                resource_name VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create glossary_terms table for curated business definitions
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_terms (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                display_name VARCHAR NOT NULL,
                definition TEXT NOT NULL,
                domain VARCHAR,
                parent_id VARCHAR,
                parent_verb VARCHAR DEFAULT 'has',
                aliases TEXT,
                semantic_type VARCHAR,
                cardinality VARCHAR DEFAULT 'many',
                plural VARCHAR,
                list_of VARCHAR,
                tags TEXT,
                owner VARCHAR,
                status VARCHAR DEFAULT 'draft',
                provenance VARCHAR DEFAULT 'llm',
                session_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add user_id column to existing databases (idempotent)
        try:
            self._conn.execute(
                "ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS user_id VARCHAR NOT NULL DEFAULT 'default'"
            )
        except Exception:
            pass  # Column already exists or unsupported syntax

        # Create entity_relationships table for SVO triples (keyed by name)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id VARCHAR PRIMARY KEY,
                subject_name VARCHAR NOT NULL,
                verb VARCHAR NOT NULL,
                object_name VARCHAR NOT NULL,
                sentence TEXT,
                confidence FLOAT DEFAULT 1.0,
                verb_category VARCHAR DEFAULT 'other',
                session_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_name, verb, object_name, session_id)
            )
        """)

        # Create unified_glossary view
        self._conn.execute("""
            CREATE VIEW IF NOT EXISTS unified_glossary AS
            SELECT
                e.id AS entity_id,
                e.name,
                COALESCE(g.display_name, e.display_name) AS display_name,
                e.semantic_type,
                e.ner_type,
                e.session_id,
                g.id AS glossary_id,
                g.domain,
                g.definition,
                g.parent_id,
                g.parent_verb,
                g.aliases,
                g.cardinality,
                g.plural,
                g.list_of,
                g.status,
                g.provenance,
                CASE
                    WHEN g.id IS NOT NULL THEN 'defined'
                    ELSE 'self_describing'
                END AS glossary_status
            FROM entities e
            LEFT JOIN glossary_terms g
                ON e.name = g.name
                AND e.session_id = g.session_id
        """)

        # Create deprecated_glossary view
        self._conn.execute("""
            CREATE VIEW IF NOT EXISTS deprecated_glossary AS
            SELECT g.*
            FROM glossary_terms g
            LEFT JOIN entities e
                ON g.name = e.name
                AND g.session_id = e.session_id
            WHERE e.id IS NULL
        """)

        # Create indexes for efficient lookups
        try:
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_semantic_type ON entities(semantic_type)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_glossary_name ON glossary_terms(name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_glossary_domain ON glossary_terms(domain)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_glossary_parent ON glossary_terms(parent_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_glossary_session ON glossary_terms(session_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_glossary_status ON glossary_terms(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rel_subject ON entity_relationships(subject_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rel_object ON entity_relationships(object_name)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rel_verb ON entity_relationships(verb)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rel_session ON entity_relationships(session_id)"
            )
        except Exception as e:
            logger.debug(f"Index creation skipped: {e}")

    # =========================================================================
    # Visibility filters — single source of truth for scoping queries
    # =========================================================================

    @staticmethod
    def entity_visibility_filter(
        session_id: str,
        active_domains: list[str] | None = None,
        alias: str = "e",
        cross_session: bool = False,
    ) -> tuple[str, list]:
        """Build the 3-part entity visibility WHERE clause.

        Entities are visible if they are:
        - Base-level (domain_id IS NULL AND session_id IS NULL)
        - Domain-scoped (domain_id IN active_domains)
        - Session-scoped (session_id = ?)

        Args:
            session_id: Current session ID
            active_domains: Active domain IDs
            alias: Table alias (e.g. 'e', 'e2', or '' for no alias)
            cross_session: If True, include entities from ANY session
                (for user-scoped glossary checks where session_id changes
                 should not affect grounding)

        Returns:
            (sql_fragment, params) — fragment is parenthesized OR clause
        """
        pfx = f"{alias}." if alias else ""
        parts = [f"({pfx}domain_id IS NULL AND {pfx}session_id IS NULL)"]
        params: list = []

        if active_domains:
            placeholders = ",".join(["?" for _ in active_domains])
            parts.append(f"{pfx}domain_id IN ({placeholders})")
            params.extend(active_domains)

        if cross_session:
            parts.append(f"{pfx}session_id IS NOT NULL")
        else:
            parts.append(f"{pfx}session_id = ?")
            params.append(session_id)

        return f"({' OR '.join(parts)})", params

    @staticmethod
    def chunk_visibility_filter(
        domain_ids: list[str] | None = None,
        alias: str = "",
    ) -> tuple[str, list]:
        """Build the chunk/embedding visibility WHERE clause.

        Chunks are visible if they are:
        - Base-level (domain_id IS NULL or '__base__')
        - Domain-scoped (domain_id IN domain_ids)

        Args:
            domain_ids: Active domain IDs
            alias: Table alias (e.g. 'em', or '' for no alias)

        Returns:
            (sql_fragment, params) — fragment is parenthesized OR clause
        """
        pfx = f"{alias}." if alias else ""
        parts = [f"{pfx}domain_id IS NULL", f"{pfx}domain_id = '__base__'"]
        params: list = []

        if domain_ids:
            placeholders = ",".join(["?" for _ in domain_ids])
            parts.append(f"{pfx}domain_id IN ({placeholders})")
            params.extend(domain_ids)

        return f"({' OR '.join(parts)})", params

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
            "SELECT COUNT(*) FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)", [session_id]
        ).fetchone()[0]
        logger.debug(f"clear_session_data({session_id}): found {emb_count} embeddings, {ent_count} entities, {link_count} links")

        # Delete in order (links first - via entity_id subquery since chunk_entities doesn't have session_id)
        self._conn.execute("DELETE FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)", [session_id])
        self._conn.execute("DELETE FROM entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM embeddings WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM glossary_terms WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM entity_relationships WHERE session_id = ?", [session_id])

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

    def get_source_hash(self, source_id: str, hash_type: str) -> str | None:
        """Get a config hash for a source.

        Args:
            source_id: Source ID (domain filename or "__base__")
            hash_type: One of 'db', 'api', 'doc'

        Returns:
            Config hash string or None if not found
        """
        if hash_type not in ('db', 'api', 'doc'):
            raise ValueError(f"Invalid hash_type: {hash_type}")

        column = f"{hash_type}_hash"
        result = self._conn.execute(
            f"SELECT {column} FROM source_hashes WHERE source_id = ?",
            [source_id],
        ).fetchone()
        hash_val = result[0] if result else None
        logger.debug(f"get_source_hash({source_id}, {hash_type}): {hash_val}")
        return hash_val

    def set_source_hash(self, source_id: str, hash_type: str, config_hash: str) -> None:
        """Set a config hash for a source.

        Args:
            source_id: Source ID (domain filename or "__base__")
            hash_type: One of 'db', 'api', 'doc'
            config_hash: Hash of the source configuration
        """
        if hash_type not in ('db', 'api', 'doc'):
            raise ValueError(f"Invalid hash_type: {hash_type}")

        column = f"{hash_type}_hash"
        # Upsert: insert row if not exists, then update the specific column
        self._conn.execute("""
            INSERT INTO source_hashes (source_id) VALUES (?)
            ON CONFLICT (source_id) DO NOTHING
        """, [source_id])
        self._conn.execute(f"""
            UPDATE source_hashes SET {column} = ?, updated_at = CURRENT_TIMESTAMP
            WHERE source_id = ?
        """, [config_hash, source_id])
        logger.debug(f"set_source_hash({source_id}, {hash_type}): {config_hash}")

    # Backwards compatibility aliases
    def get_domain_config_hash(self, domain_id: str) -> str | None:
        """Get the document config hash for a source (backwards compatibility)."""
        return self.get_source_hash(domain_id, 'doc')

    def set_domain_config_hash(self, domain_id: str, config_hash: str) -> None:
        """Set the document config hash for a source (backwards compatibility)."""
        self.set_source_hash(domain_id, 'doc', config_hash)

    # =========================================================================
    # Resource-Level Hashing (Fine-Grained Cache Invalidation)
    # =========================================================================

    @staticmethod
    def _make_resource_id(source_id: str, resource_type: str, resource_name: str) -> str:
        """Create a unique resource ID.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Type of resource ('database', 'api', 'document')
            resource_name: Name of the resource

        Returns:
            Unique resource ID in format '{source_id}:{type}:{name}'
        """
        return f"{source_id}:{resource_type}:{resource_name}"

    def get_resource_hash(
        self,
        source_id: str,
        resource_type: str,
        resource_name: str,
    ) -> str | None:
        """Get the content hash for a specific resource.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Type of resource ('database', 'api', 'document')
            resource_name: Name of the resource

        Returns:
            Content hash or None if not found
        """
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        result = self._conn.execute(
            "SELECT content_hash FROM resource_hashes WHERE resource_id = ?",
            [resource_id],
        ).fetchone()
        hash_val = result[0] if result else None
        logger.debug(f"get_resource_hash({resource_id}): {hash_val}")
        return hash_val

    def set_resource_hash(
        self,
        source_id: str,
        resource_type: str,
        resource_name: str,
        content_hash: str,
    ) -> None:
        """Set the content hash for a specific resource.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Type of resource ('database', 'api', 'document')
            resource_name: Name of the resource
            content_hash: Hash of the resource's content
        """
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        # Use two statements for DuckDB compatibility (CURRENT_TIMESTAMP not allowed in ON CONFLICT)
        self._conn.execute("""
            INSERT INTO resource_hashes (resource_id, resource_type, resource_name, source_id, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (resource_id) DO UPDATE SET
                content_hash = excluded.content_hash
        """, [resource_id, resource_type, resource_name, source_id, content_hash])
        # Update timestamp separately
        self._conn.execute("""
            UPDATE resource_hashes SET updated_at = CURRENT_TIMESTAMP WHERE resource_id = ?
        """, [resource_id])
        logger.debug(f"set_resource_hash({resource_id}): {content_hash}")

    def delete_resource_hash(
        self,
        source_id: str,
        resource_type: str,
        resource_name: str,
    ) -> bool:
        """Delete the hash for a specific resource.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Type of resource ('database', 'api', 'document')
            resource_name: Name of the resource

        Returns:
            True if a hash was deleted, False if not found
        """
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        result = self._conn.execute(
            "DELETE FROM resource_hashes WHERE resource_id = ? RETURNING resource_id",
            [resource_id],
        ).fetchone()
        deleted = result is not None
        logger.debug(f"delete_resource_hash({resource_id}): deleted={deleted}")
        return deleted

    def get_resource_hashes_for_source(
        self,
        source_id: str,
        resource_type: str | None = None,
    ) -> dict[str, str]:
        """Get all resource hashes for a source.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Optional filter by resource type

        Returns:
            Dict mapping resource_name to content_hash
        """
        if resource_type:
            result = self._conn.execute(
                "SELECT resource_name, content_hash FROM resource_hashes WHERE source_id = ? AND resource_type = ?",
                [source_id, resource_type],
            ).fetchall()
        else:
            result = self._conn.execute(
                "SELECT resource_name, content_hash FROM resource_hashes WHERE source_id = ?",
                [source_id],
            ).fetchall()
        return {row[0]: row[1] for row in result}

    def delete_resource_chunks(
        self,
        source_id: str,
        resource_type: str,
        resource_name: str,
    ) -> int:
        """Delete chunks for a specific resource.

        Used for incremental updates when a single resource changes.

        Args:
            source_id: Source ID ('__base__' or domain filename)
            resource_type: Type of resource ('database', 'api', 'document')
            resource_name: Name of the resource (used as document_name in embeddings)

        Returns:
            Number of chunks deleted
        """
        # Map resource type to source type in embeddings
        source_map = {
            'database': 'schema',
            'api': 'api',
            'document': 'document',
        }
        source_type = source_map.get(resource_type, resource_type)

        # Get chunk IDs for this resource
        if source_id == '__base__':
            # Base config: domain_id is NULL or '__base__'
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (domain_id IS NULL OR domain_id = '__base__')
                """,
                [resource_name, source_type],
            ).fetchall()
        else:
            # Domain: match domain_id
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ? AND domain_id = ?
                """,
                [resource_name, source_type, source_id],
            ).fetchall()

        chunk_ids = [row[0] for row in chunk_ids]

        if not chunk_ids:
            logger.debug(f"delete_resource_chunks({source_id}, {resource_type}, {resource_name}): no chunks found")
            return 0

        # Delete chunk-entity links
        placeholders = ",".join(["?" for _ in chunk_ids])
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )

        # Delete chunks
        if source_id == '__base__':
            self._conn.execute(
                """
                DELETE FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (domain_id IS NULL OR domain_id = '__base__')
                """,
                [resource_name, source_type],
            )
        else:
            self._conn.execute(
                """
                DELETE FROM embeddings
                WHERE document_name = ? AND source = ? AND domain_id = ?
                """,
                [resource_name, source_type, source_id],
            )

        logger.info(f"delete_resource_chunks({source_id}, {resource_type}, {resource_name}): deleted {len(chunk_ids)} chunks")
        return len(chunk_ids)

    def clear_resource_hashes_for_source(self, source_id: str) -> int:
        """Clear all resource hashes for a source.

        Args:
            source_id: Source ID to clear

        Returns:
            Number of hashes deleted
        """
        result = self._conn.execute(
            "DELETE FROM resource_hashes WHERE source_id = ? RETURNING resource_id",
            [source_id],
        ).fetchall()
        count = len(result)
        logger.debug(f"clear_resource_hashes_for_source({source_id}): deleted {count} hashes")
        return count

    def clear_domain_embeddings(self, domain_id: str) -> int:
        """Clear all embeddings for a domain.

        Args:
            domain_id: Domain ID to clear

        Returns:
            Number of embeddings deleted
        """
        # Get chunk IDs first for clearing related entities
        chunk_ids = self._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE domain_id = ?",
            [domain_id]
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
        self._conn.execute(
            "DELETE FROM embeddings WHERE domain_id = ?",
            [domain_id]
        )

        # Clear domain entities
        self._conn.execute(
            "DELETE FROM entities WHERE domain_id = ?",
            [domain_id]
        )

        count = len(chunk_ids)
        logger.debug(f"clear_domain_embeddings({domain_id}): deleted {count} embeddings")
        return count

    @staticmethod
    def _generate_chunk_id(chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
        session_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        """Add chunks with embeddings to DuckDB.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
            source: Resource type - 'schema', 'api', or 'document'
            session_id: Optional session ID for documents added during a session
            domain_id: Optional domain ID for documents belonging to a domain
        """
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        if len(chunks) == 0:
            return

        doc_names = set(c.document_name for c in chunks)
        print(f"[ADD_CHUNKS] docs={doc_names}, session_id={session_id}, domain_id={domain_id}")

        # Check which documents already exist - skip re-indexing to preserve domain_id
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
            # Convert chunk_type enum to string value
            chunk_type_str = chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)
            records.append((
                chunk_id,
                chunk.document_name,
                source,
                chunk_type_str,
                chunk.section,
                chunk.chunk_index,
                chunk.content,
                embedding,
                session_id,
                domain_id,
            ))

        # Simple INSERT for new documents only
        self._conn.executemany(
            """
            INSERT INTO embeddings
            (chunk_id, document_name, source, chunk_type, section, chunk_index, content, embedding, session_id, domain_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
        chunk_types: list[str] | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search using DuckDB's array_cosine_similarity.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            domain_ids: List of domain IDs to include (None means no domain filter)
            session_id: Session ID to include (None means no session filter)
            chunk_types: Optional list of chunk_type values to filter by

        Returns:
            List of (chunk_id, similarity, DocumentChunk) tuples
        """
        # Ensure query is 1D list
        query = query_embedding.flatten().tolist()

        chunk_filter, filter_params = self.chunk_visibility_filter(domain_ids)
        params: list = [query] + filter_params

        # Optional chunk_type filter
        chunk_type_clause = ""
        if chunk_types:
            ct_values = [ct.value if hasattr(ct, 'value') else str(ct) for ct in chunk_types]
            ct_placeholders = ",".join(["?" for _ in ct_values])
            chunk_type_clause = f" AND chunk_type IN ({ct_placeholders})"
            params.extend(ct_values)

        params.append(limit)

        # Query with cosine similarity
        result = self._conn.execute(
            f"""
            SELECT
                chunk_id,
                document_name,
                source,
                chunk_type,
                section,
                chunk_index,
                content,
                array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) as similarity
            FROM embeddings
            WHERE {chunk_filter}{chunk_type_clause}
            ORDER BY similarity DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

        # Convert to output format
        from constat.discovery.models import ChunkType
        results = []
        for row in result:
            chunk_id, doc_name, source, chunk_type_str, section, chunk_idx, content, similarity = row
            # Convert string to ChunkType enum
            try:
                chunk_type = ChunkType(chunk_type_str) if chunk_type_str else ChunkType.DOCUMENT
            except ValueError:
                chunk_type = ChunkType.DOCUMENT
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                source=source or "document",
                chunk_type=chunk_type,
            )
            results.append((chunk_id, float(similarity), chunk))

        return results

    def clear(self) -> None:
        """Clear all stored data."""
        self._conn.execute("DELETE FROM embeddings")

    def clear_chunks(self, source: str) -> None:
        """Clear chunks by source type.

        The vector store holds chunks from 3 resource types:
        - "schema": Database table/column descriptions
        - "api": API endpoint descriptions
        - "document": User documents

        Each resource manager (SchemaManager, APISchemaManager, DocumentDiscoveryTools)
        is responsible for its own chunks and should only clear its own type.

        Args:
            source: Resource type to clear: "schema", "api", or "document"
        """
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        self._conn.execute("DELETE FROM embeddings WHERE source = ?", [source])

    def clear_chunk_entity_links(self, session_id: str | None = None) -> None:
        """Clear chunk-entity links (but keep entities).

        Args:
            session_id: If provided, only clear links for entities in this session.
                       If None, clear all links.
        """
        if session_id:
            # Clear links for entities in a specific session (join via entities table)
            self._conn.execute("""
                DELETE FROM chunk_entities
                WHERE entity_id IN (
                    SELECT id FROM entities WHERE session_id = ?
                )
            """, [session_id])
        else:
            # Clear all links
            self._conn.execute("DELETE FROM chunk_entities")

    def count(self, source: str | None = None) -> int:
        """Return number of stored chunks, optionally filtered by source.

        Args:
            source: If provided, count only chunks with this source ('schema', 'api', 'document').
                    If None, count all chunks.
        """
        if source:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE source = ?", [source]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    def search_by_source(
        self,
        query_embedding: np.ndarray,
        source: str,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[str, float, "DocumentChunk"]]:
        """Search chunks filtered by source type using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            source: Source type to filter by ('schema', 'api', 'document')
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of (chunk_id, similarity, DocumentChunk) tuples
        """
        from constat.discovery.models import ChunkType

        query = query_embedding.flatten().tolist()

        result = self._conn.execute(
            f"""
            SELECT
                chunk_id,
                document_name,
                source,
                chunk_type,
                section,
                chunk_index,
                content,
                array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) as similarity
            FROM embeddings
            WHERE source = ?
              AND array_cosine_similarity(embedding, ?::FLOAT[{self.EMBEDDING_DIM}]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
            """,
            [query, source, query, min_similarity, limit],
        ).fetchall()

        results = []
        for row in result:
            chunk_id, doc_name, src, chunk_type_str, section, chunk_idx, content, similarity = row
            try:
                chunk_type = ChunkType(chunk_type_str) if chunk_type_str else ChunkType.DOCUMENT
            except ValueError:
                chunk_type = ChunkType.DOCUMENT
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                source=src or "document",
                chunk_type=chunk_type,
            )
            results.append((chunk_id, float(similarity), chunk))

        return results

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

    def get_all_chunk_ids(self, session_id: str | None = None) -> list[str]:
        """Get all chunk IDs, optionally filtered by session.

        Args:
            session_id: If provided, only return chunk IDs visible to this session
                       (base + domain chunks). If None, return all chunk IDs.

        Returns:
            List of chunk ID strings
        """
        if session_id:
            result = self._conn.execute(
                "SELECT chunk_id FROM embeddings WHERE session_id IS NULL OR session_id = ?",
                [session_id],
            ).fetchall()
        else:
            result = self._conn.execute("SELECT chunk_id FROM embeddings").fetchall()
        return [row[0] for row in result]

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

    def add_entities(
        self,
        entities: list[Entity],
        session_id: str,
    ) -> None:
        """Add NER-extracted entities to the store.

        Args:
            entities: List of Entity objects to add
            session_id: Session ID (required - entities are session-scoped)
        """
        if not entities:
            return

        # Deduplicate entities by ID
        seen_ids = set()
        records = []
        for entity in entities:
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                records.append((
                    entity.id,
                    entity.name,
                    entity.display_name,
                    entity.semantic_type,
                    entity.ner_type,
                    session_id,
                    entity.domain_id,
                    entity.created_at,
                ))

        # Insert entities one at a time, skipping duplicates
        conn = self._conn
        for record in records:
            try:
                conn.execute(
                    """
                    INSERT INTO entities (id, name, display_name, semantic_type, ner_type, session_id, domain_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record,
                )
            except Exception:
                # Entity already exists, skip
                pass

    def link_chunk_entities(
        self,
        links: list[ChunkEntity],
    ) -> None:
        """Create links between chunks and NER entities.

        Args:
            links: List of ChunkEntity objects defining the relationships
        """
        if not links:
            return

        # Deduplicate links by (chunk_id, entity_id)
        seen = set()
        unique_records = []
        for link in links:
            key = (link.chunk_id, link.entity_id)
            if key not in seen:
                seen.add(key)
                unique_records.append((link.chunk_id, link.entity_id, link.confidence))

        logger.debug(f"link_chunk_entities: inserting {len(unique_records)} links")

        # Insert links one at a time, skipping duplicates
        conn = self._conn
        for record in unique_records:
            try:
                conn.execute(
                    """
                    INSERT INTO chunk_entities (chunk_id, entity_id, confidence)
                    VALUES (?, ?, ?)
                    """,
                    record,
                )
            except Exception:
                # Link already exists, skip
                pass

    def get_entities_for_chunk(
        self,
        chunk_id: str,
        session_id: str,
    ) -> list[Entity]:
        """Get all NER entities associated with a chunk.

        Args:
            chunk_id: The chunk identifier
            session_id: Session ID (required - entities are session-scoped)

        Returns:
            List of Entity objects linked to this chunk
        """
        result = self._conn.execute(
            """
            SELECT e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                   e.session_id, e.domain_id, e.created_at
            FROM entities e
            JOIN chunk_entities ce ON e.id = ce.entity_id
            WHERE ce.chunk_id = ? AND e.session_id = ?
            ORDER BY ce.confidence DESC
            """,
            [chunk_id, session_id],
        ).fetchall()

        entities = []
        for row in result:
            entity_id, name, display_name, semantic_type, ner_type, sess_id, dom_id, created_at = row
            entities.append(Entity(
                id=entity_id,
                name=name,
                display_name=display_name,
                semantic_type=semantic_type,
                ner_type=ner_type,
                session_id=sess_id,
                domain_id=dom_id,
                created_at=created_at,
            ))

        return entities

    def get_chunks_for_entity(
        self,
        entity_id: str,
        limit: int | None = None,
        domain_ids: list[str] | None = None,
    ) -> list[tuple[str, DocumentChunk, float]]:
        """Get chunks that mention an entity.

        Args:
            entity_id: The entity identifier
            limit: Maximum number of chunks to return (None = all)
            domain_ids: List of domain IDs to include (filters embeddings)

        Returns:
            List of (chunk_id, DocumentChunk, confidence) tuples
            ordered by confidence
        """
        emb_where, filter_params = self.chunk_visibility_filter(domain_ids, alias="em")
        params: list = [entity_id] + filter_params

        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)

        result = self._conn.execute(
            f"""
            SELECT
                ce.chunk_id,
                em.document_name,
                em.content,
                em.section,
                em.chunk_index,
                em.source,
                ce.confidence
            FROM chunk_entities ce
            JOIN embeddings em ON ce.chunk_id = em.chunk_id
            WHERE ce.entity_id = ? AND {emb_where}
            ORDER BY ce.confidence DESC
            {limit_clause}
            """,
            params,
        ).fetchall()

        chunks = []
        for row in result:
            chunk_id, doc_name, content, section, chunk_idx, source, confidence = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                source=source,
            )
            chunks.append((chunk_id, chunk, confidence))

        return chunks

    def find_entity_by_name(
        self,
        name: str,
        domain_ids: Optional[list[str]] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Entity]:
        """Find an entity by its name (case-insensitive).

        When session_id is provided, uses the full 3-part visibility filter
        (base + domain + session) so all visible entities are found.

        Args:
            name: Entity name to search for
            domain_ids: Optional domain IDs for visibility
            session_id: Optional session ID for visibility

        Returns:
            Entity if found, None otherwise
        """
        params: list = [name]

        if session_id:
            vis_filter, vis_params = self.entity_visibility_filter(
                session_id, domain_ids, alias="",
            )
            where = f"LOWER(name) = LOWER(?) AND {vis_filter}"
            params.extend(vis_params)
        elif domain_ids:
            chunk_filter, vis_params = self.chunk_visibility_filter(domain_ids)
            where = f"LOWER(name) = LOWER(?) AND {chunk_filter}"
            params.extend(vis_params)
        else:
            where = "LOWER(name) = LOWER(?)"

        result = self._conn.execute(
            f"""
            SELECT id, name, display_name, semantic_type, ner_type,
                   session_id, domain_id, created_at
            FROM entities
            WHERE {where}
            LIMIT 1
            """,
            params,
        ).fetchone()

        if not result:
            return None

        entity_id, entity_name, display_name, semantic_type, ner_type, sess_id, dom_id, created_at = result

        return Entity(
            id=entity_id,
            name=entity_name,
            display_name=display_name,
            semantic_type=semantic_type,
            ner_type=ner_type,
            session_id=sess_id,
            domain_id=dom_id,
            created_at=created_at,
        )

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by its ID."""
        row = self._conn.execute(
            "SELECT id, name, display_name, semantic_type, ner_type, "
            "session_id, domain_id, created_at FROM entities WHERE id = ? LIMIT 1",
            [entity_id],
        ).fetchone()
        if not row:
            return None
        return Entity(
            id=row[0], name=row[1], display_name=row[2],
            semantic_type=row[3], ner_type=row[4], session_id=row[5],
            domain_id=row[6], created_at=row[7],
        )

    def search_enriched(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[EnrichedChunk]:
        """Search for chunks and include associated entities.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            domain_ids: List of domain IDs to include (None means no domain filter)
            session_id: Session ID to include entities (None means no entities)

        Returns:
            List of EnrichedChunk objects with entities
        """
        # First do the regular search with filtering
        results = self.search(query_embedding, limit, domain_ids, session_id)

        # Enrich with entities if session_id provided
        enriched = []
        for chunk_id, score, chunk in results:
            entities = []
            if session_id:
                entities = self.get_entities_for_chunk(chunk_id, session_id)
            enriched.append(EnrichedChunk(
                chunk=chunk,
                score=score,
                entities=entities,
            ))

        return enriched

    def search_similar_entities(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.3,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """Find entities linked to chunks similar to the query embedding.

        Searches chunks by cosine similarity, then returns distinct entities
        linked to those chunks via chunk_entities.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of entity results
            min_similarity: Minimum cosine similarity threshold
            domain_ids: Domain IDs to include (None means no domain filter)
            session_id: Session ID to scope entities

        Returns:
            List of dicts with id, name, type, similarity
        """
        # Search chunks first (fetch more to get enough unique entities)
        results = self.search(query_embedding, limit=limit * 3, domain_ids=domain_ids, session_id=session_id)

        # Collect entities from matched chunks, tracking best similarity
        entity_best: dict[str, dict] = {}
        for chunk_id, similarity, _chunk in results:
            if similarity < min_similarity:
                continue
            # Get entities linked to this chunk
            entity_filter = ["ce.chunk_id = ?"]
            params: list = [chunk_id]
            if session_id:
                entity_filter.append("e.session_id = ?")
                params.append(session_id)
            where = " AND ".join(entity_filter)

            rows = self._conn.execute(
                f"""
                SELECT e.id, e.name, e.semantic_type
                FROM chunk_entities ce
                JOIN entities e ON ce.entity_id = e.id
                WHERE {where}
                """,
                params,
            ).fetchall()

            for eid, entity_name, entity_type in rows:
                if eid not in entity_best or similarity > entity_best[eid]["similarity"]:
                    entity_best[eid] = {
                        "id": eid,
                        "name": entity_name,
                        "type": entity_type,
                        "similarity": similarity,
                    }

        # Sort by similarity and limit
        sorted_entities = sorted(entity_best.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_entities[:limit]

    def clear_entities(self, _source: Optional[str] = None) -> None:  # noqa: ARG002
        """Clear all entities and chunk-entity links.

        DEPRECATED: Use clear_session_entities(session_id) for session-scoped cleanup.
        This method now clears ALL entities regardless of the source parameter.

        Args:
            _source: Ignored (kept for backwards compatibility)
        """
        self._conn.execute("DELETE FROM chunk_entities")
        self._conn.execute("DELETE FROM entities")

    def count_entities(self) -> int:
        """Return number of stored entities."""
        result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return result[0] if result else 0

    def clear_session_entities(self, session_id: str) -> tuple[int, int]:
        """Clear all entities and chunk_entities for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Tuple of (links_deleted, entities_deleted)
        """
        # Count before delete
        link_count = self._conn.execute(
            "SELECT COUNT(*) FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)",
            [session_id]
        ).fetchone()[0]
        entity_count = self._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE session_id = ?",
            [session_id]
        ).fetchone()[0]

        self._conn.execute("DELETE FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)", [session_id])
        self._conn.execute("DELETE FROM entities WHERE session_id = ?", [session_id])

        logger.info(f"clear_session_entities({session_id[:8]}): deleted {link_count} links, {entity_count} entities")
        return link_count, entity_count

    def clear_domain_session_entities(self, session_id: str, domain_id: str) -> int:
        """Clear entities for a specific domain in a session.

        Args:
            session_id: Session ID
            domain_id: Domain ID to clear entities for

        Returns:
            Number of entities deleted
        """
        # Get entity IDs to delete
        result = self._conn.execute(
            "SELECT id FROM entities WHERE session_id = ? AND domain_id = ?",
            [session_id, domain_id]
        ).fetchall()
        entity_ids = [row[0] for row in result]

        if not entity_ids:
            return 0

        # Delete chunk-entity links for these entities
        placeholders = ",".join(["?" for _ in entity_ids])
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE entity_id IN ({placeholders})",
            entity_ids
        )

        # Delete the entities
        self._conn.execute(
            "DELETE FROM entities WHERE session_id = ? AND domain_id = ?",
            [session_id, domain_id]
        )

        logger.debug(f"clear_domain_session_entities({session_id}, {domain_id}): deleted {len(entity_ids)} entities")
        return len(entity_ids)

    def get_entity_names(self, session_id: str) -> list[str]:
        """Get all entity names for a session.

        Args:
            session_id: Session ID

        Returns:
            List of entity names
        """
        result = self._conn.execute(
            "SELECT DISTINCT name FROM entities WHERE session_id = ?",
            [session_id],
        ).fetchall()
        return [row[0] for row in result]

    def extract_entities_for_session(
        self,
        session_id: str,
        domain_ids: list[str] | None = None,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Run NER entity extraction on all chunks for a session.

        Args:
            session_id: Session ID
            domain_ids: Domain IDs to include (chunks from these + base)
            schema_terms: Database table/column names for custom patterns
            api_terms: API endpoint names for custom patterns
            business_terms: Business glossary terms for custom patterns

        Returns:
            Number of entities extracted
        """
        from constat.discovery.entity_extractor import EntityExtractor

        # Clear any existing entities for this session
        self.clear_session_entities(session_id)

        # Get all chunks for base + active domains
        chunks = self.get_all_chunks(domain_ids)
        if not chunks:
            logger.debug(f"No chunks found for session {session_id}")
            return 0

        logger.info(f"Extracting entities from {len(chunks)} chunks for session {session_id}")

        # Create extractor with custom patterns
        extractor = EntityExtractor(
            session_id=session_id,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
        )

        # Extract entities from each chunk
        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            results = extractor.extract(chunk)
            for entity, link in results:
                all_links.append(link)

        # Get all unique entities and add to store
        entities = extractor.get_all_entities()
        if entities:
            self.add_entities(entities, session_id)
            self.link_chunk_entities(all_links)
            logger.info(f"Extracted {len(entities)} entities from {len(chunks)} chunks")

        return len(entities)

    def extract_entities_for_domain(
        self,
        session_id: str,
        domain_id: str,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Extract entities for a specific domain's chunks (incremental add).

        Args:
            session_id: Session ID
            domain_id: Domain ID to extract entities for
            schema_terms: Database table/column names for custom patterns
            api_terms: API endpoint names for custom patterns
            business_terms: Business glossary terms for custom patterns

        Returns:
            Number of entities extracted
        """
        from constat.discovery.entity_extractor import EntityExtractor

        # Get chunks for this specific domain
        chunks = self.get_domain_chunks(domain_id)
        if not chunks:
            logger.debug(f"No chunks found for domain {domain_id}")
            return 0

        logger.info(f"Extracting entities from {len(chunks)} chunks for domain {domain_id} in session {session_id}")

        # Create extractor with custom patterns
        extractor = EntityExtractor(
            session_id=session_id,
            domain_id=domain_id,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
        )

        # Extract entities from each chunk
        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            results = extractor.extract(chunk)
            for entity, link in results:
                all_links.append(link)

        # Get all unique entities and add to store
        entities = extractor.get_all_entities()
        if entities:
            self.add_entities(entities, session_id)
            self.link_chunk_entities(all_links)
            logger.info(f"Extracted {len(entities)} entities from {len(chunks)} chunks for domain {domain_id}")

        return len(entities)

    @staticmethod
    def _rows_to_chunks(rows: list) -> list[DocumentChunk]:
        """Convert raw SQL rows to DocumentChunk objects."""
        from constat.discovery.models import ChunkType
        chunks = []
        for row in rows:
            doc_name, content, section, chunk_idx, source, chunk_type_str = row
            try:
                chunk_type = ChunkType(chunk_type_str) if chunk_type_str else ChunkType.DOCUMENT
            except ValueError:
                chunk_type = ChunkType.DOCUMENT
            chunks.append(DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                source=source or "document",
                chunk_type=chunk_type,
            ))
        return chunks

    def get_domain_chunks(self, domain_id: str) -> list[DocumentChunk]:
        """Get all chunks for a specific domain.

        Args:
            domain_id: Domain ID

        Returns:
            List of DocumentChunk objects
        """
        result = self._conn.execute(
            """
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings
            WHERE domain_id = ?
            ORDER BY document_name, chunk_index
            """,
            [domain_id],
        ).fetchall()

        return self._rows_to_chunks(result)

    def get_all_chunks(self, domain_ids: list[str] | None = None) -> list[DocumentChunk]:
        """Get all chunks for base + specified domains.

        Args:
            domain_ids: Domain IDs to include

        Returns:
            List of DocumentChunk objects
        """
        chunk_filter, params = self.chunk_visibility_filter(domain_ids)

        result = self._conn.execute(
            f"""
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings
            WHERE {chunk_filter}
            ORDER BY document_name, chunk_index
            """,
            params,
        ).fetchall()

        return self._rows_to_chunks(result)

    # =========================================================================
    # Glossary Term Methods
    # =========================================================================

    @staticmethod
    def _term_from_row(row) -> GlossaryTerm:
        """Convert a database row to a GlossaryTerm."""
        import json
        (term_id, name, display_name, definition, domain, parent_id,
         parent_verb, aliases_json, semantic_type, cardinality, plural, list_of,
         tags_json, owner, status, provenance, session_id, user_id,
         created_at, updated_at) = row
        aliases = json.loads(aliases_json) if aliases_json else []
        tags = json.loads(tags_json) if tags_json else {}
        return GlossaryTerm(
            id=term_id,
            name=name,
            display_name=display_name,
            definition=definition,
            domain=domain,
            parent_id=parent_id,
            parent_verb=parent_verb or "has",
            aliases=aliases,
            semantic_type=semantic_type,
            cardinality=cardinality or "many",
            plural=plural,
            list_of=list_of,
            tags=tags,
            owner=owner,
            status=status or "draft",
            provenance=provenance or "llm",
            session_id=session_id,
            user_id=user_id or "default",
            created_at=created_at,
            updated_at=updated_at,
        )

    _GLOSSARY_COLUMNS = (
        "id, name, display_name, definition, domain, parent_id, parent_verb, "
        "aliases, semantic_type, cardinality, plural, list_of, "
        "tags, owner, status, provenance, session_id, user_id, created_at, updated_at"
    )

    def add_glossary_term(self, term: GlossaryTerm) -> None:
        """Insert a glossary term (upsert on conflict)."""
        import json
        self._conn.execute(
            """
            INSERT OR REPLACE INTO glossary_terms
            (id, name, display_name, definition, domain, parent_id, parent_verb,
             aliases, semantic_type, cardinality, plural, list_of,
             tags, owner, status, provenance, session_id, user_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                term.id, term.name, term.display_name, term.definition,
                term.domain, term.parent_id, term.parent_verb,
                json.dumps(term.aliases), term.semantic_type,
                term.cardinality, term.plural, term.list_of,
                json.dumps(term.tags), term.owner,
                term.status, term.provenance, term.session_id,
                term.user_id or "default",
                term.created_at, term.updated_at,
            ],
        )

    def update_glossary_term(self, name: str, session_id: str, updates: dict, *, user_id: str | None = None) -> bool:
        """Update a glossary term by name.

        Filters by user_id when provided (user-scoped glossary), falls back to session_id.

        Args:
            name: Term name
            session_id: Session ID (legacy scope, used as fallback)
            updates: Dict of field -> value to update
            user_id: User ID for user-scoped lookup

        Returns:
            True if a row was updated
        """
        import json
        allowed = {
            "definition", "display_name", "domain", "parent_id", "parent_verb",
            "aliases", "semantic_type", "cardinality", "plural",
            "list_of", "tags", "owner", "status", "provenance",
        }
        sets = []
        params: list = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "aliases":
                value = json.dumps(value) if isinstance(value, list) else value
            elif key == "tags":
                value = json.dumps(value) if isinstance(value, dict) else value
            sets.append(f"{key} = ?")
            params.append(value)

        if not sets:
            return False

        sets.append("updated_at = CURRENT_TIMESTAMP")
        if user_id:
            params.extend([name, user_id])
            where = "name = ? AND user_id = ?"
        else:
            params.extend([name, session_id])
            where = "name = ? AND session_id = ?"
        result = self._conn.execute(
            f"UPDATE glossary_terms SET {', '.join(sets)} WHERE {where} RETURNING id",
            params,
        ).fetchone()
        return result is not None

    def delete_glossary_term(self, name: str, session_id: str, *, user_id: str | None = None) -> bool:
        """Delete a glossary term by name.

        Filters by user_id when provided, falls back to session_id.

        Returns:
            True if a row was deleted
        """
        if user_id:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE name = ? AND user_id = ? RETURNING id",
                [name, user_id],
            ).fetchone()
        else:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE name = ? AND session_id = ? RETURNING id",
                [name, session_id],
            ).fetchone()
        return result is not None

    def get_glossary_term(self, name: str, session_id: str, *, user_id: str | None = None) -> GlossaryTerm | None:
        """Get a single glossary term by name.

        Filters by user_id when provided, falls back to session_id.
        """
        if user_id:
            row = self._conn.execute(
                f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE name = ? AND user_id = ?",
                [name, user_id],
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE name = ? AND session_id = ?",
                [name, session_id],
            ).fetchone()
        return self._term_from_row(row) if row else None

    def get_glossary_term_by_id(self, term_id: str) -> GlossaryTerm | None:
        """Get a glossary term by its ID."""
        row = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE id = ?",
            [term_id],
        ).fetchone()
        return self._term_from_row(row) if row else None

    def list_glossary_terms(
        self,
        session_id: str,
        scope: str = "all",
        domain: str | None = None,
        *,
        user_id: str | None = None,
    ) -> list[GlossaryTerm]:
        """List glossary terms.

        Filters by user_id when provided (user-scoped glossary), falls back to session_id.

        Args:
            session_id: Session ID (legacy scope, used as fallback)
            scope: 'all', 'defined' (has definition), or status filter
            domain: Optional domain filter
            user_id: User ID for user-scoped lookup
        """
        if user_id:
            conditions = ["user_id = ?"]
            params: list = [user_id]
        else:
            conditions = ["session_id = ?"]
            params = [session_id]
        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        where = " AND ".join(conditions)
        rows = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE {where} ORDER BY name",
            params,
        ).fetchall()
        return [self._term_from_row(r) for r in rows]

    def get_unified_glossary(
        self,
        session_id: str,
        scope: str = "all",
        active_domains: list[str] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[dict]:
        """Get the unified glossary for a session.

        Uses the same 3-part visibility filter as the entities endpoint:
        base entities (no domain/session) + domain-scoped + session-scoped.
        Glossary terms are joined by user_id when provided (user-scoped),
        falling back to session_id for backward compat.

        Args:
            session_id: Session ID (for entity visibility)
            scope: 'all' | 'defined' | 'self_describing'
            active_domains: Active domain IDs for visibility filter
            user_id: User ID for glossary term scope

        Returns:
            List of unified glossary dicts
        """
        # Use cross_session=True so glossary (user-scoped) sees entities
        # from any session, not just the current one.
        entity_where, entity_params = self.entity_visibility_filter(
            session_id, active_domains, alias="e", cross_session=True,
        )
        entity_where2, entity_params2 = self.entity_visibility_filter(
            session_id, active_domains, alias="e2", cross_session=True,
        )

        # Glossary scope: user_id if provided, else session_id
        glossary_scope_col = "user_id" if user_id else "session_id"
        glossary_scope_val = user_id if user_id else session_id

        # Part 1 params: glossary_scope, entity_params, glossary_scope (parent subquery)
        # Part 2 params: glossary_scope (glossary_terms), entity_params2 (NOT EXISTS),
        #                glossary_scope (grounding: parent check)
        params: list = (
            [glossary_scope_val] + entity_params + [glossary_scope_val]
            + [glossary_scope_val] + entity_params2 + [glossary_scope_val]
        )

        # Scope filter
        scope_filter_1 = ""
        scope_filter_2 = ""
        if scope == "defined":
            scope_filter_1 = "AND g.id IS NOT NULL"
            scope_filter_2 = ""  # Part 2 terms are always defined
        elif scope == "self_describing":
            scope_filter_1 = "AND g.id IS NULL"
            scope_filter_2 = "AND 1=0"  # Part 2 terms are never self_describing

        rows = self._conn.execute(
            f"""
            -- Part 1: Entities with optional glossary terms
            SELECT
                e.id AS entity_id,
                e.name,
                COALESCE(g.display_name, e.display_name) AS display_name,
                e.semantic_type,
                e.ner_type,
                e.session_id,
                g.id AS glossary_id,
                g.domain,
                g.definition,
                g.parent_id,
                g.parent_verb,
                g.aliases,
                g.cardinality,
                g.plural,
                g.list_of,
                g.status,
                g.provenance,
                CASE
                    WHEN g.id IS NOT NULL THEN 'defined'
                    ELSE 'self_describing'
                END AS glossary_status
            FROM entities e
            LEFT JOIN glossary_terms g
                ON LOWER(e.name) = LOWER(g.name)
                AND g.{glossary_scope_col} = ?
            WHERE {entity_where}
            {scope_filter_1}
            AND (
                g.id IS NOT NULL
                OR EXISTS (
                    SELECT 1 FROM chunk_entities ce
                    JOIN embeddings em ON ce.chunk_id = em.chunk_id
                    WHERE ce.entity_id = e.id
                      AND em.document_name NOT LIKE 'glossary:%'
                      AND em.document_name NOT LIKE 'relationship:%'
                )
                OR e.id IN (
                    SELECT parent_id FROM glossary_terms
                    WHERE {glossary_scope_col} = ? AND parent_id IS NOT NULL
                )
            )

            UNION ALL

            -- Part 2: Glossary terms with no matching entity
            SELECT
                NULL AS entity_id,
                g.name,
                g.display_name,
                g.semantic_type,
                NULL AS ner_type,
                g.session_id,
                g.id AS glossary_id,
                g.domain,
                g.definition,
                g.parent_id,
                g.parent_verb,
                g.aliases,
                g.cardinality,
                g.plural,
                g.list_of,
                g.status,
                g.provenance,
                'defined' AS glossary_status
            FROM glossary_terms g
            WHERE g.{glossary_scope_col} = ?
            {scope_filter_2}
            AND NOT EXISTS (
                SELECT 1 FROM entities e2
                WHERE LOWER(e2.name) = LOWER(g.name) AND {entity_where2}
            )
            AND (
                -- Grounding: only include if it's a parent of another
                -- in-scope glossary term (category/taxonomy term) or
                -- is user-sourced (learning-based draft)
                g.provenance = 'learning'
                OR EXISTS (
                    SELECT 1 FROM glossary_terms g2
                    WHERE g2.parent_id = g.id
                    AND g2.{glossary_scope_col} = ?
                )
            )

            ORDER BY name
            """,
            params,
        ).fetchall()
        import json

        # Deduplicate rows by LOWER(name) — cross_session entity visibility
        # can produce multiple rows for the same entity name across sessions.
        _seen: dict[str, int] = {}
        _unique: list = []
        for row in rows:
            name_lower = row[1].lower()
            if name_lower in _seen:
                idx = _seen[name_lower]
                # Prefer row with glossary term (defined > self_describing)
                if row[6] is not None and _unique[idx][6] is None:
                    _unique[idx] = row
                continue
            _seen[name_lower] = len(_unique)
            _unique.append(row)
        rows = _unique

        results = []
        # Collect all aliases from defined terms to suppress duplicate entities
        alias_set: set[str] = set()
        for row in rows:
            aliases_json = row[11]
            glossary_id = row[6]
            if glossary_id and aliases_json:
                for a in json.loads(aliases_json):
                    if a:
                        alias_set.add(a.strip().lower())

        for row in rows:
            (entity_id, name, display_name, semantic_type, ner_type,
             sess_id, glossary_id, domain, definition, parent_id,
             parent_verb, aliases_json, cardinality, plural, list_of,
             status, provenance, glossary_status) = row
            aliases = json.loads(aliases_json) if aliases_json else []

            # Suppress self-describing entities whose name is an alias of a defined term
            if glossary_status == "self_describing" and name.lower() in alias_set:
                continue

            results.append({
                "entity_id": entity_id,
                "name": name,
                "display_name": display_name,
                "semantic_type": semantic_type,
                "ner_type": ner_type,
                "session_id": sess_id,
                "glossary_id": glossary_id,
                "domain": domain,
                "definition": definition,
                "parent_id": parent_id,
                "parent_verb": parent_verb or "has",
                "aliases": aliases,
                "cardinality": cardinality,
                "plural": plural,
                "list_of": list_of,
                "status": status,
                "provenance": provenance,
                "glossary_status": glossary_status,
            })
        return results

    def get_deprecated_glossary(
        self,
        session_id: str,
        active_domains: list[str] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[GlossaryTerm]:
        """Get glossary terms not grounded to any entity.

        A term is valid if:
        1. Its name matches a visible entity (directly grounded), OR
        2. It is an ancestor of a grounded term (upward walk), OR
        3. It has grounded descendants (downward walk via get_child_terms), OR
        4. Any other glossary term references it as parent_id and that child is valid

        Uses cross_session=True so glossary terms grounded by entities
        from any session remain non-deprecated (glossary is user-scoped,
        so grounding checks should be user-scoped too).
        """
        entity_vis, vis_params = self.entity_visibility_filter(
            session_id, active_domains, alias="e", cross_session=True,
        )

        all_terms = self.list_glossary_terms(session_id, user_id=user_id)
        if not all_terms:
            return []

        by_id = {t.id: t for t in all_terms}

        # Build parent → children map from glossary_terms
        children_of: dict[str, list[str]] = {}
        for t in all_terms:
            if t.parent_id and t.parent_id in by_id:
                children_of.setdefault(t.parent_id, []).append(t.id)

        # Find terms directly grounded to a visible entity
        grounded: set[str] = set()
        for t in all_terms:
            row = self._conn.execute(
                f"SELECT 1 FROM entities e WHERE LOWER(e.name) = LOWER(?) AND {entity_vis} LIMIT 1",
                [t.name] + vis_params,
            ).fetchone()
            if row:
                grounded.add(t.id)

        # Walk parent chains upward from grounded terms — ancestors are valid
        valid = set(grounded)
        for tid in grounded:
            current = by_id.get(tid)
            while current and current.parent_id:
                parent = by_id.get(current.parent_id)
                if not parent or parent.id in valid:
                    break
                valid.add(parent.id)
                current = parent

        # Walk downward — a parent is valid if any child is valid
        changed = True
        while changed:
            changed = False
            for t in all_terms:
                if t.id in valid:
                    continue
                for child_id in children_of.get(t.id, []):
                    if child_id in valid:
                        valid.add(t.id)
                        changed = True
                        break

        return [t for t in all_terms if t.id not in valid]

    def clear_session_glossary(self, session_id: str, *, user_id: str | None = None) -> int:
        """Clear all glossary terms for a session or user.

        When user_id is provided, clears by user_id (user-scoped glossary).

        Returns:
            Number of terms deleted
        """
        if user_id:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE user_id = ? RETURNING id",
                [user_id],
            ).fetchall()
        else:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE session_id = ? RETURNING id",
                [session_id],
            ).fetchall()
        count = len(result)
        logger.debug(f"clear_session_glossary({session_id}, user_id={user_id}): deleted {count} terms")
        return count

    def delete_glossary_by_status(
        self, session_id: str, status: str, *, user_id: str | None = None,
    ) -> int:
        """Delete glossary terms matching a status.

        Returns:
            Number of terms deleted
        """
        if user_id:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE user_id = ? AND status = ? RETURNING id",
                [user_id, status],
            ).fetchall()
        else:
            result = self._conn.execute(
                "DELETE FROM glossary_terms WHERE session_id = ? AND status = ? RETURNING id",
                [session_id, status],
            ).fetchall()
        count = len(result)
        logger.debug(f"delete_glossary_by_status({status}, user_id={user_id}): deleted {count} terms")
        return count

    # ------------------------------------------------------------------
    # Entity Relationship CRUD
    # ------------------------------------------------------------------

    def add_entity_relationship(self, rel) -> None:
        """Insert an EntityRelationship, ignoring duplicates."""
        self._conn.execute(
            """
            INSERT INTO entity_relationships
                (id, subject_name, verb, object_name,
                 sentence, confidence, verb_category, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [
                rel.id, rel.subject_name, rel.verb,
                rel.object_name, rel.sentence,
                rel.confidence, rel.verb_category, rel.session_id,
            ],
        )

    def get_relationships_for_entity(
        self, entity_name: str, session_id: str,
    ) -> list[dict]:
        """Get SVO relationships where entity is subject or object (by name), deduplicated."""
        name_lower = entity_name.lower()
        rows = self._conn.execute(
            """
            SELECT FIRST(r.id), r.subject_name, r.verb, r.object_name,
                   MAX(r.confidence), FIRST(r.verb_category)
            FROM entity_relationships r
            WHERE (LOWER(r.subject_name) = ? OR LOWER(r.object_name) = ?)
              AND r.session_id = ?
            GROUP BY r.subject_name, r.verb, r.object_name
            ORDER BY MAX(r.confidence) DESC
            """,
            [name_lower, name_lower, session_id],
        ).fetchall()
        return [
            {
                "id": r[0],
                "subject_name": r[1],
                "verb": r[2],
                "object_name": r[3],
                "confidence": r[4],
                "verb_category": r[5],
            }
            for r in rows
        ]

    def clear_session_relationships(self, session_id: str) -> int:
        """Delete all entity relationships for a session."""
        result = self._conn.execute(
            "DELETE FROM entity_relationships WHERE session_id = ? RETURNING id",
            [session_id],
        ).fetchall()
        return len(result)

    def delete_entity_relationship(self, rel_id: str) -> bool:
        """Delete a single relationship by ID."""
        result = self._conn.execute(
            "DELETE FROM entity_relationships WHERE id = ? RETURNING id",
            [rel_id],
        ).fetchall()
        return len(result) > 0

    def update_entity_relationship_verb(self, rel_id: str, verb: str) -> bool:
        """Update the verb and verb_category of a relationship."""
        from constat.discovery.relationship_extractor import categorize_verb
        verb_category = categorize_verb(verb.lower())
        result = self._conn.execute(
            "UPDATE entity_relationships SET verb = ?, verb_category = ? WHERE id = ? RETURNING id",
            [verb, verb_category, rel_id],
        ).fetchall()
        return len(result) > 0

    def get_glossary_terms_by_names(self, names: list[str], session_id: str, *, user_id: str | None = None) -> list[GlossaryTerm]:
        """Batch lookup glossary terms by name.

        Args:
            names: List of term names (case-insensitive)
            session_id: Session ID (fallback scope)
            user_id: User ID for user-scoped lookup

        Returns:
            List of matching GlossaryTerm objects
        """
        if not names:
            return []
        lower_names = [n.lower() for n in names]
        placeholders = ",".join(["?" for _ in lower_names])
        if user_id:
            params: list = lower_names + [user_id]
            scope_clause = "user_id = ?"
        else:
            params = lower_names + [session_id]
            scope_clause = "session_id = ?"
        rows = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE LOWER(name) IN ({placeholders}) AND {scope_clause}",
            params,
        ).fetchall()
        return [self._term_from_row(r) for r in rows]

    def get_glossary_term_by_name_or_alias(
        self, name: str, session_id: str, *, user_id: str | None = None,
    ) -> GlossaryTerm | None:
        """Look up a glossary term by exact name or alias.

        Args:
            name: Term name or alias (case-insensitive)
            session_id: Session ID

        Returns:
            GlossaryTerm if found, None otherwise
        """
        import json

        scope_col = "user_id" if user_id else "session_id"
        scope_val = user_id if user_id else session_id

        # 1. Exact name match
        row = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms "
            f"WHERE LOWER(name) = LOWER(?) AND {scope_col} = ?",
            [name, scope_val],
        ).fetchone()
        if row:
            return self._term_from_row(row)

        # 2. Alias search — LIKE on JSON column, then verify exact match
        rows = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms "
            f"WHERE LOWER(aliases) LIKE ? AND {scope_col} = ?",
            [f"%{name.lower()}%", scope_val],
        ).fetchall()
        for row in rows:
            term = self._term_from_row(row)
            if any(a.lower() == name.lower() for a in (term.aliases or [])):
                return term

        return None

    def get_child_terms(self, parent_id: str, *extra_ids: str) -> list[GlossaryTerm]:
        """Get child glossary terms whose parent_id matches any of the given IDs."""
        all_ids = [parent_id] + [i for i in extra_ids if i]
        placeholders = ", ".join("?" for _ in all_ids)
        rows = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE parent_id IN ({placeholders})",
            all_ids,
        ).fetchall()
        return [self._term_from_row(r) for r in rows]

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
