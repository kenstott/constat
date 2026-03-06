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
        self, query_embedding: np.ndarray, limit: int = 5, query_text: str | None = None,
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
        query_text: str | None = None,
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

    Thin wrapper that delegates to Store (RelationalStore + DuckDBVectorBackend).
    Schema initialization stays here for Phase 1.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        reranker_model: str | None = None,
        cluster_min_terms: int = 2,
        cluster_divisor: int = 5,
        cluster_max_k: int = 500,
        store_chunk_text: bool = True,
    ):
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
        self._db = ThreadLocalDuckDB(
            str(self._db_path),
            init_sql=["INSTALL vss", "LOAD vss", "INSTALL fts", "LOAD fts"],
        )
        self._reranker_model = reranker_model
        self._store_chunk_text = store_chunk_text
        if reranker_model:
            from constat.reranker_loader import RerankerModelLoader
            RerankerModelLoader.get_instance().start_loading(reranker_model)
        self._init_schema()

        # Build composed store
        from constat.storage.duckdb_backend import DuckDBVectorBackend
        from constat.storage.relational import RelationalStore
        from constat.storage.store import Store

        self._vector = DuckDBVectorBackend(
            self._db,
            reranker_model=reranker_model,
            store_chunk_text=store_chunk_text,
        )
        self._relational = RelationalStore(
            self._db,
            cluster_min_terms=cluster_min_terms,
            cluster_divisor=cluster_divisor,
            cluster_max_k=cluster_max_k,
        )
        self._store = Store(
            relational=self._relational,
            vector=self._vector,
        )

    @property
    def _conn(self):
        """Get the thread-local connection (backwards compatibility)."""
        return self._db.conn

    @property
    def _fts_dirty(self):
        return self._vector._fts_dirty

    @_fts_dirty.setter
    def _fts_dirty(self, value):
        self._vector._fts_dirty = value

    @property
    def _clusters_dirty(self):
        return self._relational._clusters_dirty

    @_clusters_dirty.setter
    def _clusters_dirty(self, value):
        self._relational._clusters_dirty = value

    # ------------------------------------------------------------------
    # Schema init (stays in DuckDBVectorStore for Phase 1)
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Initialize database schema if not exists."""
        try:
            exists = self._conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'embeddings'"
            ).fetchone()[0]
            if exists:
                self._ensure_incremental_schema()
                return
        except Exception:
            pass
        try:
            self._conn.execute("CHECKPOINT")
        except Exception:
            pass

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

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunk_entities (
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                PRIMARY KEY (chunk_id, entity_id)
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS document_urls (
                document_name VARCHAR PRIMARY KEY,
                source_url VARCHAR NOT NULL
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS source_hashes (
                source_id VARCHAR PRIMARY KEY,
                db_hash VARCHAR,
                api_hash VARCHAR,
                doc_hash VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

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

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_terms (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                display_name VARCHAR NOT NULL,
                definition TEXT NOT NULL,
                domain VARCHAR,
                parent_id VARCHAR,
                parent_verb VARCHAR DEFAULT 'HAS_KIND',
                aliases TEXT,
                semantic_type VARCHAR,
                cardinality VARCHAR DEFAULT 'many',
                plural VARCHAR,
                tags TEXT,
                owner VARCHAR,
                status VARCHAR DEFAULT 'draft',
                provenance VARCHAR DEFAULT 'llm',
                session_id VARCHAR NOT NULL,
                user_id VARCHAR NOT NULL DEFAULT 'default',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ignored BOOLEAN DEFAULT FALSE
            )
        """)

        try:
            self._conn.execute(
                "ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS user_id VARCHAR NOT NULL DEFAULT 'default'"
            )
        except Exception:
            pass

        try:
            self._conn.execute(
                "ALTER TABLE glossary_terms ADD COLUMN IF NOT EXISTS ignored BOOLEAN DEFAULT FALSE"
            )
        except Exception:
            pass

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
                user_edited BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(subject_name, verb, object_name, session_id)
            )
        """)

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

        self._conn.execute("""
            CREATE VIEW IF NOT EXISTS deprecated_glossary AS
            SELECT g.*
            FROM glossary_terms g
            LEFT JOIN entities e
                ON g.name = e.name
                AND g.session_id = e.session_id
            WHERE e.id IS NULL
        """)

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

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_clusters (
                term_name VARCHAR NOT NULL,
                cluster_id INTEGER NOT NULL,
                session_id VARCHAR NOT NULL,
                PRIMARY KEY (term_name, session_id)
            )
        """)

        self._create_ner_scope_cache_tables()

    def _create_ner_scope_cache_tables(self) -> None:
        """Create NER scope cache tables for cross-session entity caching."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_scope_cache (
                fingerprint VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                entity_count INTEGER DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_cached_entities (
                fingerprint VARCHAR NOT NULL,
                id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                display_name VARCHAR NOT NULL,
                semantic_type VARCHAR NOT NULL,
                ner_type VARCHAR,
                domain_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_cached_chunk_entities (
                fingerprint VARCHAR NOT NULL,
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 1.0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS ner_cached_clusters (
                fingerprint VARCHAR NOT NULL,
                term_name VARCHAR NOT NULL,
                cluster_id INTEGER NOT NULL
            )
        """)
        try:
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ner_cache_ent_fp ON ner_cached_entities(fingerprint)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ner_cache_ce_fp ON ner_cached_chunk_entities(fingerprint)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ner_cache_cl_fp ON ner_cached_clusters(fingerprint)")
        except Exception:
            pass

    def _ensure_incremental_schema(self) -> None:
        """Create tables added after initial schema, for pre-existing databases."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS glossary_clusters (
                term_name VARCHAR NOT NULL,
                cluster_id INTEGER NOT NULL,
                session_id VARCHAR NOT NULL,
                PRIMARY KEY (term_name, session_id)
            )
        """)
        self._create_ner_scope_cache_tables()
        try:
            self._conn.execute("ALTER TABLE entity_relationships ADD COLUMN user_edited BOOLEAN DEFAULT FALSE")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Vector operations — delegate to DuckDBVectorBackend
    # ------------------------------------------------------------------

    def add_chunks(self, *a, **kw):
        return self._vector.add_chunks(*a, **kw)

    def search(self, *a, **kw):
        return self._vector.search(*a, **kw)

    def search_by_source(self, *a, **kw):
        return self._vector.search_by_source(*a, **kw)

    def delete_by_document(self, *a, **kw):
        return self._vector.delete_by_document(*a, **kw)

    def get_chunks(self, *a, **kw):
        return self._vector.get_chunks(*a, **kw)

    def get_all_chunk_ids(self, *a, **kw):
        return self._vector.get_all_chunk_ids(*a, **kw)

    def get_all_chunks(self, *a, **kw):
        return self._vector.get_all_chunks(*a, **kw)

    def get_domain_chunks(self, *a, **kw):
        return self._vector.get_domain_chunks(*a, **kw)

    def clear(self, *a, **kw):
        return self._vector.clear(*a, **kw)

    def clear_chunks(self, *a, **kw):
        return self._vector.clear_chunks(*a, **kw)

    def clear_domain_embeddings(self, *a, **kw):
        return self._vector.clear_domain_embeddings(*a, **kw)

    def count(self, *a, **kw):
        return self._vector.count(*a, **kw)

    def store_document_url(self, *a, **kw):
        return self._vector.store_document_url(*a, **kw)

    def get_document_url(self, *a, **kw):
        return self._vector.get_document_url(*a, **kw)

    @staticmethod
    def chunk_visibility_filter(domain_ids=None, alias=""):
        from constat.storage.duckdb_backend import DuckDBVectorBackend
        return DuckDBVectorBackend.chunk_visibility_filter(domain_ids, alias)

    @staticmethod
    def _rows_to_chunks(rows):
        from constat.storage.duckdb_backend import DuckDBVectorBackend
        return DuckDBVectorBackend._rows_to_chunks(rows)

    @staticmethod
    def _generate_chunk_id(chunk):
        from constat.storage.duckdb_backend import DuckDBVectorBackend
        return DuckDBVectorBackend._generate_chunk_id(chunk)

    def _rebuild_fts_index(self):
        return self._vector._rebuild_fts_index()

    def _bm25_search(self, *a, **kw):
        return self._vector._bm25_search(*a, **kw)

    @staticmethod
    def _rrf_merge(*a, **kw):
        from constat.storage.duckdb_backend import DuckDBVectorBackend
        return DuckDBVectorBackend._rrf_merge(*a, **kw)

    def _rerank(self, *a, **kw):
        return self._vector._rerank(*a, **kw)

    # ------------------------------------------------------------------
    # Relational operations — delegate to RelationalStore
    # ------------------------------------------------------------------

    def add_entities(self, *a, **kw):
        return self._relational.add_entities(*a, **kw)

    def find_entity_by_name(self, *a, **kw):
        return self._relational.find_entity_by_name(*a, **kw)

    def get_entity_by_id(self, *a, **kw):
        return self._relational.get_entity_by_id(*a, **kw)

    def clear_entities(self, *a, **kw):
        return self._relational.clear_entities(*a, **kw)

    def count_entities(self, *a, **kw):
        return self._relational.count_entities(*a, **kw)

    def clear_session_entities(self, *a, **kw):
        return self._relational.clear_session_entities(*a, **kw)

    def clear_domain_session_entities(self, *a, **kw):
        return self._relational.clear_domain_session_entities(*a, **kw)

    def get_entity_names(self, *a, **kw):
        return self._relational.get_entity_names(*a, **kw)

    def backfill_entity_domains(self, *a, **kw):
        return self._relational.backfill_entity_domains(*a, **kw)

    def link_chunk_entities(self, *a, **kw):
        return self._relational.link_chunk_entities(*a, **kw)

    def get_entities_for_chunk(self, *a, **kw):
        return self._relational.get_entities_for_chunk(*a, **kw)

    def get_chunks_for_entity(self, *a, **kw):
        return self._relational.get_chunks_for_entity(*a, **kw)

    def clear_chunk_entity_links(self, *a, **kw):
        return self._relational.clear_chunk_entity_links(*a, **kw)

    def add_glossary_term(self, *a, **kw):
        return self._relational.add_glossary_term(*a, **kw)

    def update_glossary_term(self, *a, **kw):
        return self._relational.update_glossary_term(*a, **kw)

    def delete_glossary_term(self, *a, **kw):
        return self._relational.delete_glossary_term(*a, **kw)

    def get_glossary_term(self, *a, **kw):
        return self._relational.get_glossary_term(*a, **kw)

    def get_glossary_term_by_id(self, *a, **kw):
        return self._relational.get_glossary_term_by_id(*a, **kw)

    def list_glossary_terms(self, *a, **kw):
        return self._relational.list_glossary_terms(*a, **kw)

    def get_unified_glossary(self, *a, **kw):
        return self._relational.get_unified_glossary(*a, **kw)

    def get_deprecated_glossary(self, *a, **kw):
        return self._relational.get_deprecated_glossary(*a, **kw)

    def delete_glossary_term_cascade(self, *a, **kw):
        return self._relational.delete_glossary_term_cascade(*a, **kw)

    def rename_glossary_term(self, *a, **kw):
        return self._relational.rename_glossary_term(*a, **kw)

    def clear_session_glossary(self, *a, **kw):
        return self._relational.clear_session_glossary(*a, **kw)

    def delete_glossary_by_status(self, *a, **kw):
        return self._relational.delete_glossary_by_status(*a, **kw)

    def get_glossary_terms_by_names(self, *a, **kw):
        return self._relational.get_glossary_terms_by_names(*a, **kw)

    def get_glossary_term_by_name_or_alias(self, *a, **kw):
        return self._relational.get_glossary_term_by_name_or_alias(*a, **kw)

    def get_child_terms(self, *a, **kw):
        return self._relational.get_child_terms(*a, **kw)

    def reconcile_glossary_domains(self, *a, **kw):
        return self._relational.reconcile_glossary_domains(*a, **kw)

    def add_entity_relationship(self, *a, **kw):
        return self._relational.add_entity_relationship(*a, **kw)

    def get_relationships_for_entity(self, *a, **kw):
        return self._relational.get_relationships_for_entity(*a, **kw)

    def clear_session_relationships(self, *a, **kw):
        return self._relational.clear_session_relationships(*a, **kw)

    def clear_non_user_relationships(self, *a, **kw):
        return self._relational.clear_non_user_relationships(*a, **kw)

    def delete_entity_relationship(self, *a, **kw):
        return self._relational.delete_entity_relationship(*a, **kw)

    def update_entity_relationship_verb(self, *a, **kw):
        return self._relational.update_entity_relationship_verb(*a, **kw)

    def get_source_hash(self, *a, **kw):
        return self._relational.get_source_hash(*a, **kw)

    def set_source_hash(self, *a, **kw):
        return self._relational.set_source_hash(*a, **kw)

    def get_domain_config_hash(self, *a, **kw):
        return self._relational.get_domain_config_hash(*a, **kw)

    def set_domain_config_hash(self, *a, **kw):
        return self._relational.set_domain_config_hash(*a, **kw)

    @staticmethod
    def _make_resource_id(source_id, resource_type, resource_name):
        return RelationalStore._make_resource_id(source_id, resource_type, resource_name)

    def get_resource_hash(self, *a, **kw):
        return self._relational.get_resource_hash(*a, **kw)

    def set_resource_hash(self, *a, **kw):
        return self._relational.set_resource_hash(*a, **kw)

    def delete_resource_hash(self, *a, **kw):
        return self._relational.delete_resource_hash(*a, **kw)

    def get_resource_hashes_for_source(self, *a, **kw):
        return self._relational.get_resource_hashes_for_source(*a, **kw)

    def delete_resource_chunks(self, *a, **kw):
        return self._relational.delete_resource_chunks(*a, **kw)

    def clear_resource_hashes_for_source(self, *a, **kw):
        return self._relational.clear_resource_hashes_for_source(*a, **kw)

    def _rebuild_clusters(self, *a, **kw):
        return self._relational._rebuild_clusters(*a, **kw)

    def get_cluster_siblings(self, *a, **kw):
        return self._relational.get_cluster_siblings(*a, **kw)

    def find_matching_clusters(self, *a, **kw):
        return self._relational.find_matching_clusters(*a, **kw)

    def has_ner_scope_cache(self, *a, **kw):
        return self._relational.has_ner_scope_cache(*a, **kw)

    def restore_ner_scope_cache(self, *a, **kw):
        return self._relational.restore_ner_scope_cache(*a, **kw)

    def store_ner_scope_cache(self, *a, **kw):
        return self._relational.store_ner_scope_cache(*a, **kw)

    def evict_ner_scope_cache(self, *a, **kw):
        return self._relational.evict_ner_scope_cache(*a, **kw)

    # Phase 2 relational delegations

    def get_entity_document_names(self, *a, **kw):
        return self._relational.get_entity_document_names(*a, **kw)

    def get_cooccurring_entities(self, *a, **kw):
        return self._relational.get_cooccurring_entities(*a, **kw)

    def get_cooccurrence_pairs(self, *a, **kw):
        return self._relational.get_cooccurrence_pairs(*a, **kw)

    def get_cooccurrence_pairs_by_name(self, *a, **kw):
        return self._relational.get_cooccurrence_pairs_by_name(*a, **kw)

    def get_shared_chunk_ids(self, *a, **kw):
        return self._relational.get_shared_chunk_ids(*a, **kw)

    def get_entities_with_stats(self, *a, **kw):
        return self._relational.get_entities_with_stats(*a, **kw)

    def get_visible_entity_names(self, *a, **kw):
        return self._relational.get_visible_entity_names(*a, **kw)

    def update_entity_name(self, *a, **kw):
        return self._relational.update_entity_name(*a, **kw)

    def mark_relationship_user_edited(self, *a, **kw):
        return self._relational.mark_relationship_user_edited(*a, **kw)

    def list_session_relationships(self, *a, **kw):
        return self._relational.list_session_relationships(*a, **kw)

    def get_promotable_relationships(self, *a, **kw):
        return self._relational.get_promotable_relationships(*a, **kw)

    def get_glossary_parent_child_pairs(self, *a, **kw):
        return self._relational.get_glossary_parent_child_pairs(*a, **kw)

    def list_entities_with_refcount(self, *a, **kw):
        return self._relational.list_entities_with_refcount(*a, **kw)

    def get_entity_references(self, *a, **kw):
        return self._relational.get_entity_references(*a, **kw)

    def count_session_links(self, *a, **kw):
        return self._relational.count_session_links(*a, **kw)

    def entity_exists(self, *a, **kw):
        return self._relational.entity_exists(*a, **kw)

    def get_non_ignored_entities_for_chunk(self, *a, **kw):
        return self._relational.get_non_ignored_entities_for_chunk(*a, **kw)

    # Phase 2 vector delegations

    def get_indexed_document_names(self, *a, **kw):
        return self._vector.get_indexed_document_names(*a, **kw)

    def clear_document_chunks(self, *a, **kw):
        return self._vector.clear_document_chunks(*a, **kw)

    def get_chunks_by_document(self, *a, **kw):
        return self._vector.get_chunks_by_document(*a, **kw)

    def get_chunk_content(self, *a, **kw):
        return self._vector.get_chunk_content(*a, **kw)

    def get_shared_chunk_content(self, *a, **kw):
        return self._vector.get_shared_chunk_content(*a, **kw)

    def get_visible_chunks_with_metadata(self, *a, **kw):
        return self._vector.get_visible_chunks_with_metadata(*a, **kw)

    def delete_chunks_by_pattern(self, *a, **kw):
        return self._vector.delete_chunks_by_pattern(*a, **kw)

    def count_by_domain(self, *a, **kw):
        return self._vector.count_by_domain(*a, **kw)

    @staticmethod
    def entity_visibility_filter(*a, **kw):
        return RelationalStore.entity_visibility_filter(*a, **kw)

    def _term_from_row(self, row):
        return RelationalStore._term_from_row(row)

    @property
    def _GLOSSARY_COLUMNS(self):
        return RelationalStore._GLOSSARY_COLUMNS

    # ------------------------------------------------------------------
    # Cross-layer operations — delegate to Store
    # ------------------------------------------------------------------

    def search_enriched(self, *a, **kw):
        return self._store.search_enriched(*a, **kw)

    def search_similar_entities(self, *a, **kw):
        return self._store.search_similar_entities(*a, **kw)

    def extract_entities_for_session(self, *a, **kw):
        return self._store.extract_entities_for_session(*a, **kw)

    def extract_entities_for_domain(self, *a, **kw):
        return self._store.extract_entities_for_domain(*a, **kw)

    # ------------------------------------------------------------------
    # Session/document cleanup — cross-layer, delegate to relational with fts callback
    # ------------------------------------------------------------------

    def clear_session_data(self, session_id: str) -> None:
        self._relational.clear_session_data(
            session_id,
            fts_dirty_callback=lambda: setattr(self._vector, '_fts_dirty', True),
        )

    def delete_document(self, document_name: str, session_id: str | None = None) -> int:
        return self._relational.delete_document(
            document_name, session_id,
            fts_dirty_callback=lambda: setattr(self._vector, '_fts_dirty', True),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "_db"):
            try:
                self._db.conn.close()
            except Exception:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Import here to avoid circular imports in delegation
from constat.storage.relational import RelationalStore  # noqa: E402


def create_vector_store(
    backend: str = "duckdb",
    db_path: Optional[str] = None,
    reranker_model: str | None = None,
    cluster_min_terms: int = 2,
    cluster_divisor: int = 5,
    cluster_max_k: int = 500,
    store_chunk_text: bool = True,
) -> VectorStoreBackend:
    """Factory function to create a vector store backend.

    Args:
        backend: Backend type - "duckdb" or "numpy"
        db_path: Path to DuckDB database file (only for duckdb backend)
        reranker_model: Cross-encoder model name for reranking (only for duckdb backend)
        cluster_min_terms: Minimum glossary terms to trigger clustering.
        cluster_divisor: k = max(2, n_terms // divisor).
        cluster_max_k: Optional cap on k.
        store_chunk_text: Store original chunk text alongside embeddings.

    Returns:
        VectorStoreBackend instance

    Raises:
        ImportError: If "duckdb" backend is requested but duckdb is not installed
        ValueError: If unknown backend type is specified
    """
    if backend == "duckdb":
        return DuckDBVectorStore(
            db_path=db_path,
            reranker_model=reranker_model,
            cluster_min_terms=cluster_min_terms,
            cluster_divisor=cluster_divisor,
            cluster_max_k=cluster_max_k,
            store_chunk_text=store_chunk_text,
        )
    elif backend == "numpy":
        return NumpyVectorStore()
    else:
        raise ValueError(f"Unknown vector store backend: {backend}")
