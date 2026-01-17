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

import numpy as np

from constat.discovery.models import DocumentChunk


class VectorStoreBackend(ABC):
    """Abstract base class for vector store backends.

    Vector stores handle embedding storage and similarity search for document chunks.
    Implementations must support:
    - Adding chunks with their embeddings
    - Searching by query embedding
    - Clearing all stored data
    - Counting stored chunks
    """

    # Embedding dimension for all-MiniLM-L6-v2
    EMBEDDING_DIM = 384

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
                embedding FLOAT[{self.EMBEDDING_DIM}] NOT NULL
            )
        """)

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        """Generate a unique ID for a chunk."""
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks with embeddings to DuckDB."""
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
            ))

        # Use INSERT OR REPLACE to handle updates
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (chunk_id, document_name, section, chunk_index, content, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
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
        self._conn.execute("DELETE FROM embeddings")

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
