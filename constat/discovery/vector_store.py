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
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
    ) -> None:
        """Add document chunks with their embeddings to the store.

        Args:
            chunks: List of DocumentChunk objects to store
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            source: Resource type - 'schema', 'api', or 'document'
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
        source: str = "document",
        session_id: str | None = None,
        project_id: str | None = None,
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
        # source: resource type ('schema', 'api', 'document')
        # chunk_type: granular type ('db_table', 'db_column', 'api_endpoint', 'document', etc.)
        # project_id: source identifier ('__base__' for config, project filename for projects)
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
                project_id VARCHAR
            )
        """)

        # Create entities table for NER-extracted entities
        # Entities are session-scoped (rebuilt when session starts or projects change)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                display_name VARCHAR NOT NULL,
                semantic_type VARCHAR NOT NULL,
                ner_type VARCHAR,
                session_id VARCHAR NOT NULL,
                project_id VARCHAR,
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
        except Exception as e:
            logger.debug(f"Index creation skipped: {e}")

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
            source_id: Source ID (project filename or "__base__")
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
            source_id: Source ID (project filename or "__base__")
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
    def get_project_config_hash(self, project_id: str) -> str | None:
        """Get the document config hash for a source (backwards compatibility)."""
        return self.get_source_hash(project_id, 'doc')

    def set_project_config_hash(self, project_id: str, config_hash: str) -> None:
        """Set the document config hash for a source (backwards compatibility)."""
        self.set_source_hash(project_id, 'doc', config_hash)

    # =========================================================================
    # Resource-Level Hashing (Fine-Grained Cache Invalidation)
    # =========================================================================

    def _make_resource_id(self, source_id: str, resource_type: str, resource_name: str) -> str:
        """Create a unique resource ID.

        Args:
            source_id: Source ID ('__base__' or project filename)
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
            source_id: Source ID ('__base__' or project filename)
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
            source_id: Source ID ('__base__' or project filename)
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
            source_id: Source ID ('__base__' or project filename)
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
            source_id: Source ID ('__base__' or project filename)
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
            source_id: Source ID ('__base__' or project filename)
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
            # Base config: project_id is NULL or '__base__'
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (project_id IS NULL OR project_id = '__base__')
                """,
                [resource_name, source_type],
            ).fetchall()
        else:
            # Project: match project_id
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ? AND project_id = ?
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
                AND (project_id IS NULL OR project_id = '__base__')
                """,
                [resource_name, source_type],
            )
        else:
            self._conn.execute(
                """
                DELETE FROM embeddings
                WHERE document_name = ? AND source = ? AND project_id = ?
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
        source: str = "document",
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Add chunks with embeddings to DuckDB.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
            source: Resource type - 'schema', 'api', or 'document'
            session_id: Optional session ID for documents added during a session
            project_id: Optional project ID for documents belonging to a project
        """
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
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
                project_id,
            ))

        # Simple INSERT for new documents only
        self._conn.executemany(
            """
            INSERT INTO embeddings
            (chunk_id, document_name, source, chunk_type, section, chunk_index, content, embedding, session_id, project_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

        # Build filter: base (NULL or '__base__') + active projects
        filter_conditions = ["(project_id IS NULL)", "(project_id = '__base__')"]
        params: list = [query]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            filter_conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        where_clause = " OR ".join(filter_conditions)
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
            WHERE ({where_clause})
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
                    entity.project_id,
                    entity.created_at,
                ))

        # Insert entities one at a time, skipping duplicates
        conn = self._conn
        for record in records:
            try:
                conn.execute(
                    """
                    INSERT INTO entities (id, name, display_name, semantic_type, ner_type, session_id, project_id, created_at)
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
                   e.session_id, e.project_id, e.created_at
            FROM entities e
            JOIN chunk_entities ce ON e.id = ce.entity_id
            WHERE ce.chunk_id = ? AND e.session_id = ?
            ORDER BY ce.confidence DESC
            """,
            [chunk_id, session_id],
        ).fetchall()

        entities = []
        for row in result:
            entity_id, name, display_name, semantic_type, ner_type, sess_id, proj_id, created_at = row
            entities.append(Entity(
                id=entity_id,
                name=name,
                display_name=display_name,
                semantic_type=semantic_type,
                ner_type=ner_type,
                session_id=sess_id,
                project_id=proj_id,
                created_at=created_at,
            ))

        return entities

    def get_chunks_for_entity(
        self,
        entity_id: str,
        limit: int = 10,
        project_ids: list[str] | None = None,
    ) -> list[tuple[str, DocumentChunk, float]]:
        """Get chunks that mention an entity.

        Args:
            entity_id: The entity identifier
            limit: Maximum number of chunks to return
            project_ids: List of project IDs to include (filters embeddings)

        Returns:
            List of (chunk_id, DocumentChunk, confidence) tuples
            ordered by confidence
        """
        # Embeddings can come from base + projects
        emb_filter = ["em.project_id IS NULL"]
        params: list = [entity_id]

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            emb_filter.append(f"em.project_id IN ({placeholders})")
            params.extend(project_ids)

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
                em.source,
                ce.confidence
            FROM chunk_entities ce
            JOIN embeddings em ON ce.chunk_id = em.chunk_id
            WHERE ce.entity_id = ? AND ({emb_where})
            ORDER BY ce.confidence DESC
            LIMIT ?
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
        session_id: str,
    ) -> Optional[Entity]:
        """Find an entity by its name (case-insensitive).

        Args:
            name: Entity name to search for
            session_id: Session ID (required - entities are session-scoped)

        Returns:
            Entity if found, None otherwise
        """
        result = self._conn.execute(
            """
            SELECT id, name, display_name, semantic_type, ner_type,
                   session_id, project_id, created_at
            FROM entities
            WHERE LOWER(name) = LOWER(?) AND session_id = ?
            LIMIT 1
            """,
            [name, session_id],
        ).fetchone()

        if not result:
            return None

        entity_id, ename, display_name, semantic_type, ner_type, sess_id, proj_id, created_at = result

        return Entity(
            id=entity_id,
            name=ename,
            display_name=display_name,
            semantic_type=semantic_type,
            ner_type=ner_type,
            session_id=sess_id,
            project_id=proj_id,
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
            session_id: Session ID to include entities (None means no entities)

        Returns:
            List of EnrichedChunk objects with entities
        """
        # First do the regular search with filtering
        results = self.search(query_embedding, limit, project_ids, session_id)

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

    def clear_entities(self, source: Optional[str] = None) -> None:
        """Clear all entities and chunk-entity links.

        DEPRECATED: Use clear_session_entities(session_id) for session-scoped cleanup.
        This method now clears ALL entities regardless of the source parameter.

        Args:
            source: Ignored (kept for backwards compatibility)
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

    def clear_project_session_entities(self, session_id: str, project_id: str) -> int:
        """Clear entities for a specific project in a session.

        Args:
            session_id: Session ID
            project_id: Project ID to clear entities for

        Returns:
            Number of entities deleted
        """
        # Get entity IDs to delete
        result = self._conn.execute(
            "SELECT id FROM entities WHERE session_id = ? AND project_id = ?",
            [session_id, project_id]
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
            "DELETE FROM entities WHERE session_id = ? AND project_id = ?",
            [session_id, project_id]
        )

        logger.debug(f"clear_project_session_entities({session_id}, {project_id}): deleted {len(entity_ids)} entities")
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
        project_ids: list[str] | None = None,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Run NER entity extraction on all chunks for a session.

        Args:
            session_id: Session ID
            project_ids: Project IDs to include (chunks from these + base)
            schema_terms: Database table/column names for custom patterns
            api_terms: API endpoint names for custom patterns
            business_terms: Business glossary terms for custom patterns

        Returns:
            Number of entities extracted
        """
        from constat.discovery.entity_extractor import EntityExtractor

        # Clear any existing entities for this session
        self.clear_session_entities(session_id)

        # Get all chunks for base + active projects
        chunks = self.get_all_chunks(project_ids)
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

    def extract_entities_for_project(
        self,
        session_id: str,
        project_id: str,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Extract entities for a specific project's chunks (incremental add).

        Args:
            session_id: Session ID
            project_id: Project ID to extract entities for
            schema_terms: Database table/column names for custom patterns
            api_terms: API endpoint names for custom patterns
            business_terms: Business glossary terms for custom patterns

        Returns:
            Number of entities extracted
        """
        from constat.discovery.entity_extractor import EntityExtractor

        # Get chunks for this specific project
        chunks = self.get_project_chunks(project_id)
        if not chunks:
            logger.debug(f"No chunks found for project {project_id}")
            return 0

        logger.info(f"Extracting entities from {len(chunks)} chunks for project {project_id} in session {session_id}")

        # Create extractor with custom patterns
        extractor = EntityExtractor(
            session_id=session_id,
            project_id=project_id,
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
            logger.info(f"Extracted {len(entities)} entities from {len(chunks)} chunks for project {project_id}")

        return len(entities)

    def get_project_chunks(self, project_id: str) -> list[DocumentChunk]:
        """Get all chunks for a specific project.

        Args:
            project_id: Project ID

        Returns:
            List of DocumentChunk objects
        """
        result = self._conn.execute(
            """
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings
            WHERE project_id = ?
            ORDER BY document_name, chunk_index
            """,
            [project_id],
        ).fetchall()

        from constat.discovery.models import ChunkType
        chunks = []
        for row in result:
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

    def get_all_chunks(self, project_ids: list[str] | None = None) -> list[DocumentChunk]:
        """Get all chunks for base + specified projects.

        Args:
            project_ids: Project IDs to include

        Returns:
            List of DocumentChunk objects
        """
        # Build filter for base + projects
        conditions = ["project_id IS NULL"]
        params: list = []

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        where_clause = " OR ".join(conditions)

        result = self._conn.execute(
            f"""
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings
            WHERE {where_clause}
            ORDER BY document_name, chunk_index
            """,
            params,
        ).fetchall()

        from constat.discovery.models import ChunkType
        chunks = []
        for row in result:
            doc_name, content, section, chunk_idx, source, chunk_type_str = row
            # Convert string to ChunkType enum
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
