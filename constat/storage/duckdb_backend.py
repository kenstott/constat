# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""DuckDB implementation of the VectorBackend ABC.

Handles the embeddings table, FTS index, BM25/RRF hybrid search,
and optional cross-encoder reranking. Shares a ThreadLocalDuckDB
connection with RelationalStore.
"""

import hashlib
import logging

import numpy as np

from constat.discovery.models import DocumentChunk
from constat.storage.vector_backend import VectorBackend

logger = logging.getLogger(__name__)


class DuckDBVectorBackend(VectorBackend):
    """Vector operations backed by DuckDB VSS + FTS extensions."""

    def __init__(
        self,
        db,
        reranker_model: str | None = None,
        store_chunk_text: bool = True,
    ):
        self._db = db
        self._fts_dirty = True
        self._reranker_model = reranker_model
        self._store_chunk_text = store_chunk_text

    @property
    def _conn(self):
        return self._db.conn

    # ------------------------------------------------------------------
    # Visibility filters
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_visibility_filter(
        domain_ids: list[str] | None = None,
        alias: str = "",
    ) -> tuple[str, list]:
        pfx = f"{alias}." if alias else ""
        parts = [f"{pfx}domain_id IS NULL", f"{pfx}domain_id = '__base__'"]
        params: list = []
        if domain_ids:
            placeholders = ",".join(["?" for _ in domain_ids])
            parts.append(f"{pfx}domain_id IN ({placeholders})")
            params.extend(domain_ids)
        return f"({' OR '.join(parts)})", params

    # ------------------------------------------------------------------
    # Chunk ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_chunk_id(chunk: DocumentChunk) -> str:
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{chunk.section}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    # ------------------------------------------------------------------
    # Add chunks
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
        source: str = "document",
        session_id: str | None = None,
        domain_id: str | None = None,
    ) -> None:
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        if len(chunks) == 0:
            return

        doc_names = set(c.document_name for c in chunks)
        print(f"[ADD_CHUNKS] docs={doc_names}, session_id={session_id}, domain_id={domain_id}")

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

        new_chunks = [c for c in chunks if c.document_name not in existing_docs]
        if not new_chunks:
            logger.debug("add_chunks: all documents already indexed, nothing to add")
            return

        records = []
        for i, chunk in enumerate(new_chunks):
            chunk_id = self._generate_chunk_id(chunk)
            original_idx = chunks.index(chunk)
            embedding = embeddings[original_idx].tolist()
            chunk_type_str = chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type)
            records.append((
                chunk_id,
                chunk.document_name,
                source,
                chunk_type_str,
                chunk.section,
                chunk.chunk_index,
                chunk.content if self._store_chunk_text else "",
                embedding,
                session_id,
                domain_id,
            ))

        self._conn.executemany(
            """
            INSERT INTO embeddings
            (chunk_id, document_name, source, chunk_type, section, chunk_index, content, embedding, session_id, domain_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )
        self._fts_dirty = True

    # ------------------------------------------------------------------
    # FTS / BM25 / RRF / Reranking
    # ------------------------------------------------------------------

    def rebuild_fts_index(self) -> None:
        self._rebuild_fts_index()

    def _rebuild_fts_index(self) -> None:
        if not self._fts_dirty:
            return
        try:
            count = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            if count == 0:
                self._fts_dirty = False
                return
            self._conn.execute(
                "PRAGMA create_fts_index('embeddings', 'chunk_id', 'content', stemmer='porter', overwrite=1)"
            )
            self._fts_dirty = False
        except Exception as e:
            logger.debug(f"FTS index rebuild failed (vector-only mode): {e}")
            self._fts_dirty = False

    def _bm25_search(
        self,
        query_text: str,
        limit: int = 5,
        chunk_filter: str = "1=1",
        filter_params: list | None = None,
        chunk_type_clause: str = "",
        ct_params: list | None = None,
        source_filter: str | None = None,
        source_params: list | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        from constat.discovery.models import ChunkType
        try:
            self._rebuild_fts_index()

            params: list = [query_text]
            where_parts = [
                "bm25_score IS NOT NULL",
                chunk_filter,
            ]
            if filter_params:
                params.extend(filter_params)
            if chunk_type_clause:
                where_parts.append(chunk_type_clause.lstrip(" AND "))
                if ct_params:
                    params.extend(ct_params)
            if source_filter:
                where_parts.append(source_filter)
                if source_params:
                    params.extend(source_params)

            params.append(limit)
            where = " AND ".join(where_parts)

            rows = self._conn.execute(
                f"""
                SELECT chunk_id, document_name, source, chunk_type, section,
                       chunk_index, content,
                       fts_main_embeddings.match_bm25(chunk_id, ?) AS bm25_score
                FROM embeddings
                WHERE {where}
                ORDER BY bm25_score DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

            results = []
            for row in rows:
                chunk_id, doc_name, src, chunk_type_str, section, chunk_idx, content, score = row
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
                results.append((chunk_id, float(score), chunk))
            return results
        except Exception as e:
            logger.debug(f"BM25 search failed (vector-only mode): {e}")
            return []

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple[str, float, DocumentChunk]],
        bm25_results: list[tuple[str, float, DocumentChunk]],
        k: int = 60,
    ) -> list[tuple[str, float, DocumentChunk]]:
        max_rrf = 2.0 / (k + 1)
        scores: dict[str, float] = {}
        chunks: dict[str, tuple[str, DocumentChunk]] = {}

        for rank, (chunk_id, _score, chunk) in enumerate(vector_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk_id] = (chunk_id, chunk)

        for rank, (chunk_id, _score, chunk) in enumerate(bm25_results, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunks:
                chunks[chunk_id] = (chunk_id, chunk)

        merged = []
        for chunk_id, rrf_score in scores.items():
            normalized = rrf_score / max_rrf
            cid, chunk = chunks[chunk_id]
            merged.append((cid, normalized, chunk))

        merged.sort(key=lambda x: x[1], reverse=True)
        return merged

    def _rerank(
        self,
        query_text: str,
        candidates: list[tuple[str, float, DocumentChunk]],
        limit: int,
    ) -> list[tuple[str, float, DocumentChunk]]:
        if not candidates:
            return candidates

        from constat.reranker_loader import RerankerModelLoader
        model = RerankerModelLoader.get_instance().get_model()

        pairs = [(query_text, chunk.content) for _, _, chunk in candidates]
        raw_scores = model.predict(pairs)

        sigmoid = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float64)))

        scored = [
            (cid, float(sigmoid[i]), chunk)
            for i, (cid, _, chunk) in enumerate(candidates)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
        chunk_types: list[str] | None = None,
        query_text: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        query = query_embedding.flatten().tolist()

        chunk_filter, filter_params = self.chunk_visibility_filter(domain_ids)
        params: list = [query] + filter_params

        chunk_type_clause = ""
        ct_params: list = []
        if chunk_types:
            ct_values = [ct.value if hasattr(ct, 'value') else str(ct) for ct in chunk_types]
            ct_placeholders = ",".join(["?" for _ in ct_values])
            chunk_type_clause = f" AND chunk_type IN ({ct_placeholders})"
            ct_params = ct_values
            params.extend(ct_values)

        fetch_limit = limit * 3 if query_text else limit
        params.append(fetch_limit)

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

        from constat.discovery.models import ChunkType
        vector_results = []
        for row in result:
            chunk_id, doc_name, source, chunk_type_str, section, chunk_idx, content, similarity = row
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
            vector_results.append((chunk_id, float(similarity), chunk))

        if not query_text:
            return vector_results

        bm25_results = self._bm25_search(
            query_text,
            limit=fetch_limit,
            chunk_filter=chunk_filter,
            filter_params=list(filter_params),
            chunk_type_clause=chunk_type_clause,
            ct_params=ct_params,
        )
        if not bm25_results:
            results = vector_results[:limit]
        else:
            results = self._rrf_merge(vector_results, bm25_results)[:limit * 3]

        if self._reranker_model and query_text:
            return self._rerank(query_text, results, limit)
        return results[:limit]

    def search_by_source(
        self,
        query_embedding: np.ndarray,
        source: str,
        limit: int = 5,
        min_similarity: float = 0.3,
        query_text: str | None = None,
    ) -> list[tuple[str, float, DocumentChunk]]:
        from constat.discovery.models import ChunkType

        query = query_embedding.flatten().tolist()
        fetch_limit = limit * 3 if query_text else limit

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
            [query, source, query, min_similarity, fetch_limit],
        ).fetchall()

        vector_results = []
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
            vector_results.append((chunk_id, float(similarity), chunk))

        if not query_text:
            return vector_results

        bm25_results = self._bm25_search(
            query_text,
            limit=fetch_limit,
            source_filter="source = ?",
            source_params=[source],
        )
        if not bm25_results:
            results = vector_results[:limit]
        else:
            results = self._rrf_merge(vector_results, bm25_results)[:limit * 3]

        if self._reranker_model and query_text:
            return self._rerank(query_text, results, limit)
        return results[:limit]

    # ------------------------------------------------------------------
    # Chunk retrieval
    # ------------------------------------------------------------------

    def get_chunks(self) -> list[DocumentChunk]:
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
        if session_id:
            result = self._conn.execute(
                "SELECT chunk_id FROM embeddings WHERE session_id IS NULL OR session_id = ?",
                [session_id],
            ).fetchall()
        else:
            result = self._conn.execute("SELECT chunk_id FROM embeddings").fetchall()
        return [row[0] for row in result]

    @staticmethod
    def _rows_to_chunks(rows: list) -> list[DocumentChunk]:
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

    # ------------------------------------------------------------------
    # Delete / clear
    # ------------------------------------------------------------------

    def delete_by_document(self, document_name: str) -> int:
        count_before = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        )
        self._fts_dirty = True
        return count_before

    def delete_by_source(self, source: str, domain_id: str | None = None) -> int:
        if domain_id:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE source = ? AND domain_id = ?",
                [source, domain_id],
            ).fetchone()[0]
            self._conn.execute(
                "DELETE FROM embeddings WHERE source = ? AND domain_id = ?",
                [source, domain_id],
            )
        else:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE source = ?",
                [source],
            ).fetchone()[0]
            self._conn.execute(
                "DELETE FROM embeddings WHERE source = ?",
                [source],
            )
        self._fts_dirty = True
        return count

    def clear(self) -> None:
        self._conn.execute("DELETE FROM embeddings")
        self._fts_dirty = True

    def clear_chunks(self, source: str) -> None:
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        self._conn.execute("DELETE FROM embeddings WHERE source = ?", [source])
        self._fts_dirty = True

    def clear_domain_embeddings(self, domain_id: str) -> int:
        chunk_ids = self._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE domain_id = ?",
            [domain_id]
        ).fetchall()
        chunk_ids = [row[0] for row in chunk_ids]

        if chunk_ids:
            placeholders = ",".join(["?" for _ in chunk_ids])
            self._conn.execute(
                f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
                chunk_ids
            )

        self._conn.execute(
            "DELETE FROM embeddings WHERE domain_id = ?",
            [domain_id]
        )

        self._conn.execute(
            "DELETE FROM entities WHERE domain_id = ?",
            [domain_id]
        )

        self._fts_dirty = True
        count = len(chunk_ids)
        logger.debug(f"clear_domain_embeddings({domain_id}): deleted {count} embeddings")
        return count

    def count(self, source: str | None = None) -> int:
        if source:
            result = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE source = ?", [source]
            ).fetchone()
        else:
            result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    # ------------------------------------------------------------------
    # Document URLs
    # ------------------------------------------------------------------

    def store_document_url(self, document_name: str, source_url: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO document_urls (document_name, source_url) VALUES (?, ?)",
            [document_name, source_url],
        )

    def get_document_url(self, document_name: str) -> str | None:
        row = self._conn.execute(
            "SELECT source_url FROM document_urls WHERE document_name = ?",
            [document_name],
        ).fetchone()
        return row[0] if row else None
