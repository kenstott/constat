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
import threading

import numpy as np

from constat.discovery.models import DocumentChunk, Entity, ChunkEntity, EnrichedChunk


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

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
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
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search using brute-force cosine similarity."""
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
                     ~/.constat/vectors.duckdb
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB is required for DuckDBVectorStore. "
                "Install with: pip install duckdb"
            )

        self._duckdb = duckdb
        # Lock for thread-safe database access - DuckDB connections are NOT thread-safe
        self._lock = threading.Lock()

        # Determine database path
        if db_path:
            self._db_path = Path(db_path).expanduser()
        else:
            self._db_path = Path.cwd() / ".constat" / "vectors.duckdb"

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect and initialize
        self._conn = self._duckdb.connect(str(self._db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        # Load VSS extension for vector operations
        try:
            self._conn.execute("INSTALL vss")
            self._conn.execute("LOAD vss")
        except Exception:
            # VSS might already be loaded or not available in older versions
            pass

        # Create embeddings table with FLOAT array for vectors
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id VARCHAR PRIMARY KEY,
                document_name VARCHAR NOT NULL,
                section VARCHAR,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{self.EMBEDDING_DIM}] NOT NULL,
                ephemeral BOOLEAN DEFAULT FALSE
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
                ephemeral BOOLEAN DEFAULT FALSE
            )
        """)
        # type: table, column, api_endpoint, api_field, api_schema, extracted
        # source: schema, api, document

        # Create chunk_entities junction table (links document chunks to entities)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                mention_count INTEGER DEFAULT 1,
                confidence FLOAT DEFAULT 1.0,
                ephemeral BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (chunk_id, entity_id)
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
        except Exception:
            pass  # Indexes might already exist

    def _migrate_schema(self) -> None:
        """Migrate existing tables to new schema if needed."""
        # Add ephemeral column to tables if missing
        for table in ['embeddings', 'entities', 'chunk_entities']:
            try:
                cols = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
                col_names = [c[1] for c in cols]
                if 'ephemeral' not in col_names:
                    self._conn.execute(
                        f"ALTER TABLE {table} ADD COLUMN ephemeral BOOLEAN DEFAULT FALSE"
                    )
            except Exception:
                pass

        # Add new columns to entities table for unified catalog
        try:
            cols = self._conn.execute("PRAGMA table_info(entities)").fetchall()
            col_names = [c[1] for c in cols]

            if 'source' not in col_names:
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN source VARCHAR DEFAULT 'document'"
                )
            if 'parent_id' not in col_names:
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN parent_id VARCHAR"
                )
            if 'config_hash' not in col_names:
                self._conn.execute(
                    "ALTER TABLE entities ADD COLUMN config_hash VARCHAR"
                )
        except Exception:
            pass

        # Drop old tables if they exist (migrating to unified entities)
        try:
            self._conn.execute("DROP TABLE IF EXISTS table_embeddings")
            self._conn.execute("DROP TABLE IF EXISTS api_embeddings")
        except Exception:
            pass

    def clear_ephemeral(self) -> None:
        """Remove all ephemeral (session-added) data.

        Called at startup to clean up temporary data from previous sessions.
        """
        with self._lock:
            # Delete ephemeral chunk-entity links first (foreign key consideration)
            self._conn.execute("DELETE FROM chunk_entities WHERE ephemeral = TRUE")
            # Delete ephemeral entities
            self._conn.execute("DELETE FROM entities WHERE ephemeral = TRUE")
            # Delete ephemeral chunks
            self._conn.execute("DELETE FROM embeddings WHERE ephemeral = TRUE")

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
        ephemeral: bool = False,
    ) -> None:
        """Add chunks with embeddings to DuckDB.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: numpy array of embeddings
            ephemeral: If True, marks chunks as session-only (cleaned up on restart)
        """
        if len(chunks) == 0:
            return

        # Prepare data for insertion
        records = []
        for i, chunk in enumerate(chunks):
            chunk_id = self._generate_chunk_id(chunk)
            embedding = embeddings[i].tolist()
            records.append((
                chunk_id,
                chunk.document_name,
                chunk.section,
                chunk.chunk_index,
                chunk.content,
                embedding,
                ephemeral,
            ))

        # Use INSERT OR REPLACE to handle updates
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (chunk_id, document_name, section, chunk_index, content, embedding, ephemeral)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def search(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> list[tuple[str, float, DocumentChunk]]:
        """Search using DuckDB's array_cosine_similarity."""
        # Ensure query is 1D list
        query = query_embedding.flatten().tolist()

        # Query with cosine similarity
        with self._lock:
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
                ORDER BY similarity DESC
                LIMIT ?
                """,
                [query, limit],
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
        with self._lock:
            self._conn.execute("DELETE FROM embeddings")

    def count(self) -> int:
        """Return number of stored chunks."""
        with self._lock:
            result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    def get_chunks(self) -> list[DocumentChunk]:
        """Get all stored chunks."""
        with self._lock:
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
        with self._lock:
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

        with self._lock:
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
        ephemeral: bool = False,
        source: str = "document",
    ) -> None:
        """Add multiple entities to the store.

        Args:
            entities: List of Entity objects to add
            embeddings: Optional embeddings array of shape (n_entities, embedding_dim)
            ephemeral: If True, marks entities as session-only (cleaned up on restart)
            source: Source category ('document', 'schema', 'api')
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
                ephemeral,
            ))

        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO entities (id, name, type, source, embedding, metadata, created_at, ephemeral)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                records,
            )

    def link_chunk_entities(
        self,
        links: list[ChunkEntity],
        ephemeral: bool = False,
    ) -> None:
        """Create links between chunks and entities.

        Args:
            links: List of ChunkEntity objects defining the relationships
            ephemeral: If True, marks links as session-only (cleaned up on restart)
        """
        if not links:
            return

        records = [(l.chunk_id, l.entity_id, l.mention_count, l.confidence, ephemeral) for l in links]

        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_entities (chunk_id, entity_id, mention_count, confidence, ephemeral)
                VALUES (?, ?, ?, ?, ?)
                """,
                records,
            )

    def get_entities_for_chunk(self, chunk_id: str) -> list[Entity]:
        """Get all entities associated with a chunk.

        Args:
            chunk_id: The chunk identifier

        Returns:
            List of Entity objects linked to this chunk
        """
        with self._lock:
            result = self._conn.execute(
                """
                SELECT e.id, e.name, e.type, e.metadata, e.created_at
                FROM entities e
                JOIN chunk_entities ce ON e.id = ce.entity_id
                WHERE ce.chunk_id = ?
                ORDER BY ce.mention_count DESC, ce.confidence DESC
                """,
                [chunk_id],
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
    ) -> list[tuple[str, DocumentChunk, int, float]]:
        """Get chunks that mention an entity.

        Args:
            entity_id: The entity identifier
            limit: Maximum number of chunks to return

        Returns:
            List of (chunk_id, DocumentChunk, mention_count, confidence) tuples
            ordered by mention_count and confidence
        """
        with self._lock:
            result = self._conn.execute(
                """
                SELECT
                    e.chunk_id,
                    em.document_name,
                    em.content,
                    em.section,
                    em.chunk_index,
                    e.mention_count,
                    e.confidence
                FROM chunk_entities e
                JOIN embeddings em ON e.chunk_id = em.chunk_id
                WHERE e.entity_id = ?
                ORDER BY e.mention_count DESC, e.confidence DESC
                LIMIT ?
                """,
                [entity_id, limit],
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

    def find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find an entity by its name (case-insensitive).

        Args:
            name: Entity name to search for

        Returns:
            Entity if found, None otherwise
        """
        with self._lock:
            result = self._conn.execute(
                """
                SELECT id, name, type, metadata, created_at
                FROM entities
                WHERE LOWER(name) = LOWER(?)
                LIMIT 1
                """,
                [name],
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
    ) -> list[EnrichedChunk]:
        """Search for chunks and include associated entities.

        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of EnrichedChunk objects with entities
        """
        # First do the regular search
        results = self.search(query_embedding, limit)

        # Enrich with entities
        enriched = []
        for chunk_id, score, chunk in results:
            entities = self.get_entities_for_chunk(chunk_id)
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
        with self._lock:
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
        with self._lock:
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

        records = []
        for i, e in enumerate(entities):
            metadata_json = json.dumps(e.get("metadata", {})) if e.get("metadata") else None
            records.append((
                e["id"],
                e["name"],
                e["type"],
                source,
                e.get("parent_id"),
                embeddings[i].tolist() if embeddings is not None else None,
                metadata_json,
                config_hash,
            ))

        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO entities
                (id, name, type, source, parent_id, embedding, metadata, config_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    type = EXCLUDED.type,
                    source = EXCLUDED.source,
                    parent_id = EXCLUDED.parent_id,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    config_hash = EXCLUDED.config_hash
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
    ) -> list[dict]:
        """Search for relevant catalog entities by embedding similarity.

        Args:
            query_embedding: Query embedding vector
            source: Filter by source ('schema', 'api', 'document') or None for all
            entity_type: Filter by type ('table', 'column', 'api_endpoint', etc.) or None for all
            limit: Maximum number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of dicts with id, name, type, source, parent_id, metadata, similarity
        """
        query = query_embedding.flatten().tolist()

        # Build WHERE clause based on filters
        conditions = ["embedding IS NOT NULL"]
        params = [query]

        if source:
            conditions.append("source = ?")
            params.append(source)
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)

        where_clause = " AND ".join(conditions)
        params.extend([query, min_similarity, limit])

        with self._lock:
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
        with self._lock:
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
        with self._lock:
            self._conn.execute("DELETE FROM entities WHERE source = ?", [source])

    def count_catalog_entities(self, source: Optional[str] = None) -> int:
        """Return number of stored catalog entities.

        Args:
            source: Optional source filter ('schema', 'api', 'document')

        Returns:
            Count of entities
        """
        with self._lock:
            if source:
                result = self._conn.execute(
                    "SELECT COUNT(*) FROM entities WHERE source = ?", [source]
                ).fetchone()
            else:
                result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return result[0] if result else 0

    def get_entity_names_by_source(
        self, source: Optional[str] = None, entity_type: Optional[str] = None
    ) -> list[str]:
        """Get all entity names, optionally filtered by source and type.

        Args:
            source: Optional source filter ('schema', 'api', 'document')
            entity_type: Optional type filter ('table', 'column', 'api_endpoint', etc.)

        Returns:
            List of entity names
        """
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._lock:
            result = self._conn.execute(
                f"SELECT DISTINCT name FROM entities{where_clause}",
                params,
            ).fetchall()

        return [row[0] for row in result]

    def get_entities_by_parent(self, parent_id: str) -> list[dict]:
        """Get all child entities for a given parent.

        Args:
            parent_id: Parent entity ID

        Returns:
            List of child entity dicts
        """
        with self._lock:
            result = self._conn.execute(
                """
                SELECT id, name, type, source, parent_id, metadata
                FROM entities
                WHERE parent_id = ?
                ORDER BY name
                """,
                [parent_id],
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
