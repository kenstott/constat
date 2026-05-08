# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Vector store adapter for document embedding storage and search.

Wraps chonk's Store for core vector operations while adding
constat-specific resource hashing and entity management on top of the
same DuckDB connection.

Public interface:
    DuckDBVectorStore  — persistent store (chonk Store adapter)
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from constat.discovery.models import DocumentChunk, Entity, ChunkEntity, EnrichedChunk, SemanticType

logger = logging.getLogger(__name__)


def _run_entity_extraction(
    chunks,
    session_id: str,
    project_id: Optional[str],
    schema_terms,
    api_terms,
    business_terms,
    add_entities_fn,
    link_fn,
) -> int:
    """Two-pass entity extraction using chonk's SchemaMatcher + SpacyMatcher."""
    from chonk.ner import SchemaMatcher, SpacyMatcher, merge_matches

    schema_matcher = SchemaMatcher(
        schema_terms=schema_terms or [],
        api_terms=api_terms or [],
        business_terms=business_terms or [],
    )
    spacy_matcher = SpacyMatcher()

    _SEMANTIC = {
        "schema": SemanticType.CONCEPT,
        "api": SemanticType.ACTION,
        "term": SemanticType.TERM,
    }
    _NER = {"schema": "SCHEMA", "api": "API", "term": "TERM"}

    entity_cache: dict[str, Entity] = {}
    all_links: list[ChunkEntity] = []

    for chunk in chunks:
        section_str = (
            " > ".join(chunk.section)
            if isinstance(chunk.section, list)
            else (chunk.section or "")
        )
        chunk_id = (
            f"{chunk.document_name}_{chunk.chunk_index}_"
            + hashlib.sha256(
                f"{chunk.document_name}:{section_str}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
            ).hexdigest()[:16]
        )

        vocab_hits = schema_matcher.match(chunk.content)
        spacy_hits = spacy_matcher.match(chunk.content)
        matches = merge_matches(vocab_hits, spacy_hits, source_text=chunk.content)

        seen: set[str] = set()
        for match in matches:
            key = match.name.lower()
            if key in seen:
                continue
            seen.add(key)

            if key not in entity_cache:
                entity_id = hashlib.sha256(
                    f"{key}:{session_id}".encode()
                ).hexdigest()[:16]
                semantic_type = _SEMANTIC.get(match.entity_type, SemanticType.CONCEPT)
                ner_type_val = _NER.get(match.entity_type, match.entity_type.upper())
                entity_cache[key] = Entity(
                    id=entity_id,
                    name=key,
                    display_name=match.display_name,
                    semantic_type=semantic_type,
                    ner_type=ner_type_val,
                    session_id=session_id,
                    project_id=project_id,
                )

            entity = entity_cache[key]
            confidence = 0.9 if match.entity_type in {"schema", "api", "term"} else 0.75
            all_links.append(ChunkEntity(
                chunk_id=chunk_id,
                entity_id=entity.id,
                confidence=confidence,
            ))

    entities = list(entity_cache.values())
    if entities:
        add_entities_fn(entities)
        link_fn(all_links)
    return len(entities)


class DuckDBVectorStore:
    """Adapter: wraps chonk.storage.Store for core vector operations.

    Adds constat-specific resource hashing and entity management on top of
    the same DuckDB connection.
    """

    EMBEDDING_DIM = 1024

    def __init__(self, db_path: str | None = None):
        import os
        from chonk.storage import Store

        if db_path:
            self._db_path = Path(db_path).expanduser()
        elif os.environ.get("CONSTAT_VECTOR_STORE_PATH"):
            self._db_path = Path(os.environ["CONSTAT_VECTOR_STORE_PATH"])
        else:
            self._db_path = Path.home() / ".constat" / "vectors.duckdb"

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # chonk Store — owns the DuckDB connection and core embeddings table
        self._store = Store(str(self._db_path), embedding_dim=self.EMBEDDING_DIM)

        # Extend the shared connection with constat-specific columns and tables
        self._init_constat_schema()

    @property
    def _conn(self):
        """Direct DuckDB connection access for raw SQL."""
        return self._store.vector._conn

    def _init_constat_schema(self) -> None:
        """Add constat-specific columns and tables to the shared DuckDB."""
        conn = self._conn

        # Add constat-specific columns to chonk's embeddings table (nullable only)
        # Note: chonk's DDL already creates the `namespace` column
        for col in ("source", "session_id"):
            try:
                conn.execute(f"ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS {col} VARCHAR")
            except Exception as e:
                logger.debug(f"Column {col} add skipped: {e}")

        # source_hashes — config hashes per source (one row per source_id)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS source_hashes (
                source_id VARCHAR PRIMARY KEY,
                db_hash VARCHAR,
                api_hash VARCHAR,
                doc_hash VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # resource_hashes — fine-grained cache invalidation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS resource_hashes (
                resource_id VARCHAR PRIMARY KEY,
                resource_type VARCHAR NOT NULL,
                resource_name VARCHAR NOT NULL,
                source_id VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Constat-specific entity tables (extend chonk's entities table concept
        # with session_id / project_id scoping)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_entities (
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_chunk_entities (
                chunk_id VARCHAR NOT NULL,
                entity_id VARCHAR NOT NULL,
                confidence FLOAT DEFAULT 1.0,
                PRIMARY KEY (chunk_id, entity_id)
            )
        """)

        # Indexes
        for ddl in [
            "CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_namespace ON embeddings(namespace)",
            "CREATE INDEX IF NOT EXISTS idx_sess_entities_session ON session_entities(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_sess_chunk_ent ON session_chunk_entities(entity_id)",
        ]:
            try:
                conn.execute(ddl)
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")

    # =========================================================================
    # Core vector operations
    # =========================================================================

    def _generate_chunk_id(self, chunk: DocumentChunk) -> str:
        content_hash = hashlib.sha256(
            f"{chunk.document_name}:{(" > ".join(chunk.section) if chunk.section else "")}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()[:16]
        return f"{chunk.document_name}_{chunk.chunk_index}_{content_hash}"

    def add_chunks(self, chunks, embeddings, source="document",
                   session_id=None, namespace=None, project_id=None) -> None:
        if project_id is not None and namespace is None:
            namespace = project_id
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        if len(chunks) == 0:
            return

        # Skip documents already indexed
        doc_names = set(c.document_name for c in chunks)
        existing_docs: set[str] = set()
        for doc_name in doc_names:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
                [doc_name],
            ).fetchone()[0]
            if count > 0:
                existing_docs.add(doc_name)

        new_chunks = [c for c in chunks if c.document_name not in existing_docs]
        if not new_chunks:
            return

        records = []
        for chunk in new_chunks:
            chunk_id = self._generate_chunk_id(chunk)
            original_idx = chunks.index(chunk)
            embedding = embeddings[original_idx].tolist()
            chunk_type_str = (
                chunk.chunk_type.value
                if hasattr(chunk.chunk_type, 'value')
                else str(chunk.chunk_type)
            )
            records.append((
                chunk_id,
                chunk.document_name,
                " > ".join(chunk.section) if chunk.section else None,
                chunk.chunk_index,
                chunk.content,
                chunk_type_str,
                source,
                session_id,
                namespace,
                embedding,
            ))

        self._conn.executemany(
            """
            INSERT INTO embeddings
                (chunk_id, document_name, section, chunk_index, content,
                 chunk_type, source, session_id, namespace, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            records,
        )

    def search(self, query_embedding, limit=5, namespaces=None, session_id=None,
               query_text=None) -> list:
        from chonk import EnhancedSearch

        # Build effective namespace list: always include __base__; None means all
        if namespaces:
            effective_ns = list({*namespaces, "__base__"})
        else:
            effective_ns = None

        # Proxy that injects namespace filtering into every Store.search() call
        # made internally by EnhancedSearch (structural/entity/cluster expansions)
        store = self._store
        effective_ns_capture = effective_ns

        class _NSProxy(store.__class__):  # type: ignore[misc]
            def __init__(self):
                pass  # skip Store.__init__; all state lives in outer `store`
            def search(self, query_embedding, limit=5, query_text=None, **kw):
                return store.search(query_embedding, limit=limit, query_text=query_text,  # type: ignore[call-arg]
                                    namespaces=effective_ns_capture, **kw)
            def __getattr__(self, name):
                return getattr(store, name)

        enhanced = EnhancedSearch(_NSProxy())
        scored = enhanced.search(query_embedding.flatten(), k=limit, query_text=query_text)
        return [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]

    def search_enriched(self, query_embedding, limit=5, namespaces=None,
                        session_id=None, project_ids=None) -> list:
        effective_namespaces = namespaces or project_ids
        results = self.search(query_embedding, limit=limit, namespaces=effective_namespaces,
                              session_id=session_id)
        return [EnrichedChunk(chunk=chunk, score=score) for _, score, chunk in results]

    def delete_by_document(self, document_name: str) -> int:
        count_before = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings WHERE document_name = ?",
            [document_name],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM embeddings WHERE document_name = ?",
            [document_name],
        )
        return count_before

    def count(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return result[0] if result else 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM embeddings")

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

    def clear_chunks(self, source: str) -> None:
        if source not in ("schema", "api", "document"):
            raise ValueError(f"source must be 'schema', 'api', or 'document', got: {source}")
        self._conn.execute("DELETE FROM embeddings WHERE source = ?", [source])

    def delete_document(self, document_name: str, session_id: str | None = None) -> int:
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
        placeholders = ",".join(["?" for _ in chunk_ids])
        self._conn.execute(
            f"DELETE FROM session_chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )
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
        return len(chunk_ids)

    # =========================================================================
    # Entity methods (session-scoped NER entities)
    # =========================================================================

    def add_entities(self, entities, session_id=None) -> None:
        if not entities:
            return
        seen_ids: set = set()
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
                    session_id or entity.session_id,
                    entity.project_id,
                    entity.created_at,
                ))
        conn = self._conn
        for record in records:
            try:
                conn.execute(
                    """
                    INSERT INTO session_entities
                        (id, name, display_name, semantic_type, ner_type,
                         session_id, project_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record,
                )
            except Exception:
                pass

    def link_chunk_entities(self, links) -> None:
        if not links:
            return
        seen: set = set()
        unique_records = []
        for link in links:
            key = (link.chunk_id, link.entity_id)
            if key not in seen:
                seen.add(key)
                unique_records.append((link.chunk_id, link.entity_id, link.confidence))
        conn = self._conn
        for record in unique_records:
            try:
                conn.execute(
                    "INSERT INTO session_chunk_entities (chunk_id, entity_id, confidence) VALUES (?, ?, ?)",
                    record,
                )
            except Exception:
                pass

    def get_entities_for_chunk(self, chunk_id, session_id=None):
        result = self._conn.execute(
            """
            SELECT e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                   e.session_id, e.project_id, e.created_at
            FROM session_entities e
            JOIN session_chunk_entities ce ON e.id = ce.entity_id
            WHERE ce.chunk_id = ?
            """ + (" AND e.session_id = ?" if session_id else ""),
            [chunk_id] + ([session_id] if session_id else []),
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

    def get_chunks_for_entity(self, entity_id, limit=10, project_ids=None,
                               session_id=None):
        return []

    def find_entity_by_name(self, name, project_ids=None, session_id=None):
        conditions = ["LOWER(e.name) = LOWER(?)"]
        params: list = [name]

        if session_id:
            conditions.append("e.session_id = ?")
            params.append(session_id)

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            conditions.append(f"e.project_id IN ({placeholders})")
            params.extend(project_ids)

        where_clause = " AND ".join(conditions)
        result = self._conn.execute(
            f"""
            SELECT e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                   e.session_id, e.project_id, e.created_at
            FROM session_entities e
            WHERE {where_clause}
            LIMIT 1
            """,
            params,
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

    def clear_entities(self, source=None) -> None:
        self._conn.execute("DELETE FROM session_chunk_entities")
        self._conn.execute("DELETE FROM session_entities")

    def clear_chunk_entity_links(self, session_id=None) -> None:
        if session_id:
            self._conn.execute(
                "DELETE FROM session_chunk_entities WHERE entity_id IN "
                "(SELECT id FROM session_entities WHERE session_id = ?)",
                [session_id],
            )
        else:
            self._conn.execute("DELETE FROM session_chunk_entities")

    def clear_session_data(self, session_id) -> None:
        self._conn.execute(
            "DELETE FROM session_chunk_entities WHERE entity_id IN "
            "(SELECT id FROM session_entities WHERE session_id = ?)",
            [session_id],
        )
        self._conn.execute("DELETE FROM session_entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM embeddings WHERE session_id = ?", [session_id])

    def clear_session_entities(self, session_id) -> tuple:
        link_count = self._conn.execute(
            "SELECT COUNT(*) FROM session_chunk_entities WHERE entity_id IN "
            "(SELECT id FROM session_entities WHERE session_id = ?)",
            [session_id],
        ).fetchone()[0]
        entity_count = self._conn.execute(
            "SELECT COUNT(*) FROM session_entities WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]
        self._conn.execute(
            "DELETE FROM session_chunk_entities WHERE entity_id IN "
            "(SELECT id FROM session_entities WHERE session_id = ?)",
            [session_id],
        )
        self._conn.execute("DELETE FROM session_entities WHERE session_id = ?", [session_id])
        return link_count, entity_count

    # =========================================================================
    # Source and resource hashing
    # =========================================================================

    def get_source_hash(self, source_id: str, hash_type: str) -> str | None:
        if hash_type not in ('db', 'api', 'doc'):
            raise ValueError(f"Invalid hash_type: {hash_type}")
        column = f"{hash_type}_hash"
        result = self._conn.execute(
            f"SELECT {column} FROM source_hashes WHERE source_id = ?",
            [source_id],
        ).fetchone()
        return result[0] if result else None

    def set_source_hash(self, source_id: str, hash_type: str, config_hash: str) -> None:
        if hash_type not in ('db', 'api', 'doc'):
            raise ValueError(f"Invalid hash_type: {hash_type}")
        column = f"{hash_type}_hash"
        self._conn.execute(
            "INSERT INTO source_hashes (source_id) VALUES (?) ON CONFLICT (source_id) DO NOTHING",
            [source_id],
        )
        self._conn.execute(
            f"UPDATE source_hashes SET {column} = ?, updated_at = CURRENT_TIMESTAMP WHERE source_id = ?",
            [config_hash, source_id],
        )

    def get_project_config_hash(self, project_id: str) -> str | None:
        return self.get_source_hash(project_id, 'doc')

    def set_project_config_hash(self, project_id: str, config_hash: str) -> None:
        self.set_source_hash(project_id, 'doc', config_hash)

    def _make_resource_id(self, source_id: str, resource_type: str, resource_name: str) -> str:
        return f"{source_id}:{resource_type}:{resource_name}"

    def get_resource_hash(self, source_id: str, resource_type: str,
                          resource_name: str) -> str | None:
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        result = self._conn.execute(
            "SELECT content_hash FROM resource_hashes WHERE resource_id = ?",
            [resource_id],
        ).fetchone()
        return result[0] if result else None

    def set_resource_hash(self, source_id: str, resource_type: str, resource_name: str,
                          content_hash: str) -> None:
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        self._conn.execute("""
            INSERT INTO resource_hashes (resource_id, resource_type, resource_name, source_id, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (resource_id) DO UPDATE SET content_hash = excluded.content_hash
        """, [resource_id, resource_type, resource_name, source_id, content_hash])
        self._conn.execute(
            "UPDATE resource_hashes SET updated_at = CURRENT_TIMESTAMP WHERE resource_id = ?",
            [resource_id],
        )

    def delete_resource_hash(self, source_id: str, resource_type: str,
                             resource_name: str) -> bool:
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        result = self._conn.execute(
            "DELETE FROM resource_hashes WHERE resource_id = ? RETURNING resource_id",
            [resource_id],
        ).fetchone()
        return result is not None

    def get_resource_hashes_for_source(self, source_id: str,
                                        resource_type: str | None = None) -> dict:
        if resource_type:
            result = self._conn.execute(
                "SELECT resource_name, content_hash FROM resource_hashes "
                "WHERE source_id = ? AND resource_type = ?",
                [source_id, resource_type],
            ).fetchall()
        else:
            result = self._conn.execute(
                "SELECT resource_name, content_hash FROM resource_hashes WHERE source_id = ?",
                [source_id],
            ).fetchall()
        return {row[0]: row[1] for row in result}

    def clear_resource_hashes_for_source(self, source_id: str) -> int:
        result = self._conn.execute(
            "DELETE FROM resource_hashes WHERE source_id = ? RETURNING resource_id",
            [source_id],
        ).fetchall()
        return len(result)

    def delete_resource_chunks(self, source_id: str, resource_type: str,
                               resource_name: str) -> int:
        source_map = {'database': 'schema', 'api': 'api', 'document': 'document'}
        source_type = source_map.get(resource_type, resource_type)

        if source_id == '__base__':
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (namespace IS NULL OR namespace = '__base__')
                """,
                [resource_name, source_type],
            ).fetchall()
        else:
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ? AND namespace = ?
                """,
                [resource_name, source_type, source_id],
            ).fetchall()

        chunk_ids = [row[0] for row in chunk_ids]
        if not chunk_ids:
            return 0

        placeholders = ",".join(["?" for _ in chunk_ids])
        self._conn.execute(
            f"DELETE FROM session_chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )

        if source_id == '__base__':
            self._conn.execute(
                """
                DELETE FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (namespace IS NULL OR namespace = '__base__')
                """,
                [resource_name, source_type],
            )
        else:
            self._conn.execute(
                """
                DELETE FROM embeddings
                WHERE document_name = ? AND source = ? AND namespace = ?
                """,
                [resource_name, source_type, source_id],
            )

        return len(chunk_ids)

    def clear_namespace_embeddings(self, namespace: str) -> int:
        chunk_ids = self._conn.execute(
            "SELECT chunk_id FROM embeddings WHERE namespace = ?",
            [namespace]
        ).fetchall()
        chunk_ids = [row[0] for row in chunk_ids]
        if chunk_ids:
            placeholders = ",".join(["?" for _ in chunk_ids])
            self._conn.execute(
                f"DELETE FROM session_chunk_entities WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
        self._conn.execute("DELETE FROM embeddings WHERE namespace = ?", [namespace])
        return len(chunk_ids)

    def get_all_chunks(self, namespaces: list[str] | None = None) -> list[DocumentChunk]:
        conditions = ["namespace IS NULL"]
        params: list = []
        if namespaces:
            placeholders = ",".join(["?" for _ in namespaces])
            conditions.append(f"namespace IN ({placeholders})")
            params.extend(namespaces)
        where_clause = " OR ".join(conditions)
        result = self._conn.execute(
            f"""
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings WHERE {where_clause}
            ORDER BY document_name, chunk_index
            """,
            params,
        ).fetchall()
        chunks = []
        for row in result:
            doc_name, content, section, chunk_idx, source, chunk_type_str = row
            chunks.append(DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section.split(" > ") if section else [],
                chunk_index=chunk_idx,
                source=source or "document",
                chunk_type=chunk_type_str or "document",
            ))
        return chunks

    def get_project_chunks(self, namespace: str) -> list[DocumentChunk]:
        result = self._conn.execute(
            """
            SELECT document_name, content, section, chunk_index, source, chunk_type
            FROM embeddings WHERE namespace = ?
            ORDER BY document_name, chunk_index
            """,
            [namespace],
        ).fetchall()
        chunks = []
        for row in result:
            doc_name, content, section, chunk_idx, source, chunk_type_str = row
            chunks.append(DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section.split(" > ") if section else [],
                chunk_index=chunk_idx,
                source=source or "document",
                chunk_type=chunk_type_str or "document",
            ))
        return chunks

    def extract_entities_for_session(self, session_id: str, namespaces=None,
                                      schema_terms=None, api_terms=None,
                                      business_terms=None) -> int:
        self.clear_session_entities(session_id)
        chunks = self.get_all_chunks(namespaces)
        if not chunks:
            return 0
        return _run_entity_extraction(
            chunks=chunks,
            session_id=session_id,
            project_id=None,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
            add_entities_fn=lambda ents: self.add_entities(ents, session_id),
            link_fn=self.link_chunk_entities,
        )

    def extract_entities_for_project(self, session_id: str, project_id: str,
                                      schema_terms=None, api_terms=None,
                                      business_terms=None) -> int:
        chunks = self.get_project_chunks(project_id)
        if not chunks:
            return 0
        return _run_entity_extraction(
            chunks=chunks,
            session_id=session_id,
            project_id=project_id,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
            add_entities_fn=lambda ents: self.add_entities(ents, session_id),
            link_fn=self.link_chunk_entities,
        )

    def get_entity_names(self, session_id: str) -> list[str]:
        result = self._conn.execute(
            "SELECT DISTINCT name FROM session_entities WHERE session_id = ?",
            [session_id],
        ).fetchall()
        return [row[0] for row in result]

    def count_entities(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM session_entities").fetchone()
        return result[0] if result else 0

    def clear_project_session_entities(self, session_id: str, project_id: str) -> int:
        result = self._conn.execute(
            "SELECT id FROM session_entities WHERE session_id = ? AND project_id = ?",
            [session_id, project_id]
        ).fetchall()
        entity_ids = [row[0] for row in result]
        if not entity_ids:
            return 0
        placeholders = ",".join(["?" for _ in entity_ids])
        self._conn.execute(
            f"DELETE FROM session_chunk_entities WHERE entity_id IN ({placeholders})",
            entity_ids,
        )
        self._conn.execute(
            "DELETE FROM session_entities WHERE session_id = ? AND project_id = ?",
            [session_id, project_id],
        )
        return len(entity_ids)

    def close(self) -> None:
        pass  # chonk Store manages its own connection lifecycle

    def __del__(self):
        self.close()


