# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Relational CRUD operations for entities, glossary, relationships, hashes,
clusters, and NER scope cache.

Shares a ThreadLocalDuckDB connection with DuckDBVectorBackend.
All queries use raw SQL through the shared connection.
"""

import logging
from typing import Optional

import numpy as np

from constat.discovery.models import (
    ChunkEntity,
    Entity,
    GlossaryTerm,
)

logger = logging.getLogger(__name__)


class RelationalStore:
    """All non-vector relational operations."""

    def __init__(self, db, cluster_min_terms: int = 2, cluster_divisor: int = 5, cluster_max_k: int = 500):
        self._db = db
        self._clusters_dirty = True
        self._cluster_min_terms = cluster_min_terms
        self._cluster_divisor = cluster_divisor
        self._cluster_max_k = cluster_max_k

    @property
    def _conn(self):
        return self._db.conn

    # ------------------------------------------------------------------
    # Visibility filters
    # ------------------------------------------------------------------

    @staticmethod
    def entity_visibility_filter(
        session_id: str,
        active_domains: list[str] | None = None,
        alias: str = "e",
        cross_session: bool = False,
    ) -> tuple[str, list]:
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

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entities(self, entities: list[Entity], session_id: str) -> None:
        if not entities:
            return

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

        conn = self._conn
        try:
            conn.executemany(
                """
                INSERT INTO entities (id, name, display_name, semantic_type, ner_type, session_id, domain_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET session_id = excluded.session_id
                """,
                records,
            )
        except Exception:
            for record in records:
                try:
                    conn.execute(
                        """
                        INSERT INTO entities (id, name, display_name, semantic_type, ner_type, session_id, domain_id, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET session_id = excluded.session_id
                        """,
                        record,
                    )
                except Exception as e:
                    logger.warning(f"Failed to insert entity {record[0]}: {e}")
        logger.debug(f"add_entities: inserted {len(records)} records for session {session_id}")
        self._clusters_dirty = True

    def find_entity_by_name(
        self,
        name: str,
        domain_ids: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        cross_session: bool = False,
    ) -> Optional[Entity]:
        from constat.storage.duckdb_backend import DuckDBVectorBackend

        params: list = [name]

        if session_id:
            vis_filter, vis_params = self.entity_visibility_filter(
                session_id, domain_ids, alias="",
                cross_session=cross_session,
            )
            where = f"LOWER(name) = LOWER(?) AND {vis_filter}"
            params.extend(vis_params)
        elif domain_ids:
            chunk_filter, vis_params = DuckDBVectorBackend.chunk_visibility_filter(domain_ids)
            where = f"LOWER(name) = LOWER(?) AND {chunk_filter}"
            params.extend(vis_params)
        else:
            where = "LOWER(name) = LOWER(?)"

        result = self._conn.execute(
            f"""
            SELECT id, name, display_name, semantic_type, ner_type,
                   session_id, domain_id, created_at
            FROM entities e
            WHERE {where}
            ORDER BY (
                SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id
            ) DESC
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

    def clear_entities(self, _source: Optional[str] = None) -> None:  # noqa: ARG002
        self._conn.execute("DELETE FROM chunk_entities")
        self._conn.execute("DELETE FROM entities")

    def count_entities(self) -> int:
        result = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return result[0] if result else 0

    def clear_session_entities(self, session_id: str) -> tuple[int, int]:
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
        self._clusters_dirty = True

        logger.info(f"clear_session_entities({session_id[:8]}): deleted {link_count} links, {entity_count} entities")
        return link_count, entity_count

    def clear_domain_session_entities(self, session_id: str, domain_id: str) -> int:
        result = self._conn.execute(
            "SELECT id FROM entities WHERE session_id = ? AND domain_id = ?",
            [session_id, domain_id]
        ).fetchall()
        entity_ids = [row[0] for row in result]

        if not entity_ids:
            return 0

        placeholders = ",".join(["?" for _ in entity_ids])
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE entity_id IN ({placeholders})",
            entity_ids
        )
        self._conn.execute(
            "DELETE FROM entities WHERE session_id = ? AND domain_id = ?",
            [session_id, domain_id]
        )

        logger.debug(f"clear_domain_session_entities({session_id}, {domain_id}): deleted {len(entity_ids)} entities")
        return len(entity_ids)

    def get_entity_names(self, session_id: str) -> list[str]:
        result = self._conn.execute(
            "SELECT DISTINCT name FROM entities WHERE session_id = ?",
            [session_id],
        ).fetchall()
        return [row[0] for row in result]

    def search_similar_entities(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.3,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """Find entities linked to chunks similar to query embedding.

        This is a cross-layer method that needs vector search results.
        It's placed here because it primarily returns entity data.
        The vector search is delegated to the caller in Store.
        """
        raise NotImplementedError("Use Store.search_similar_entities instead")

    def backfill_entity_domains(self) -> int:
        result = self._conn.execute("""
            UPDATE entities SET domain_id = sub.domain_id
            FROM (
                SELECT DISTINCT ce.entity_id, c.domain_id
                FROM chunk_entities ce
                JOIN embeddings c ON ce.chunk_id = c.chunk_id
                WHERE c.domain_id IS NOT NULL AND c.domain_id != ''
            ) sub
            WHERE entities.id = sub.entity_id
              AND (entities.domain_id IS NULL OR entities.domain_id = '')
            RETURNING entities.id
        """).fetchall()
        count = len(result)
        logger.debug(f"backfill_entity_domains: updated {count} entities")
        return count

    # ------------------------------------------------------------------
    # Chunk-entity junction
    # ------------------------------------------------------------------

    def link_chunk_entities(self, links: list[ChunkEntity]) -> None:
        if not links:
            return

        seen = set()
        unique_records = []
        for link in links:
            key = (link.chunk_id, link.entity_id)
            if key not in seen:
                seen.add(key)
                unique_records.append((link.chunk_id, link.entity_id, link.confidence))

        logger.debug(f"link_chunk_entities: inserting {len(unique_records)} links")

        conn = self._conn
        try:
            conn.executemany(
                """
                INSERT INTO chunk_entities (chunk_id, entity_id, confidence)
                VALUES (?, ?, ?)
                """,
                unique_records,
            )
        except Exception:
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
                    pass

    def get_entities_for_chunk(self, chunk_id: str, session_id: str) -> list[Entity]:
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
        all_domains: bool = False,
    ) -> list[tuple[str, "DocumentChunk", float]]:
        from constat.discovery.models import DocumentChunk
        from constat.storage.duckdb_backend import DuckDBVectorBackend

        if all_domains:
            emb_where = "1=1"
            filter_params: list = []
        else:
            emb_where, filter_params = DuckDBVectorBackend.chunk_visibility_filter(domain_ids, alias="em")
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

    def clear_chunk_entity_links(self, session_id: str | None = None) -> None:
        if session_id:
            self._conn.execute("""
                DELETE FROM chunk_entities
                WHERE entity_id IN (
                    SELECT id FROM entities WHERE session_id = ?
                )
            """, [session_id])
        else:
            self._conn.execute("DELETE FROM chunk_entities")

    # ------------------------------------------------------------------
    # Glossary CRUD
    # ------------------------------------------------------------------

    # Canonical column order — queried columns are intersected with actual table at runtime
    _ALL_GLOSSARY_COLUMNS = [
        "id", "name", "display_name", "definition", "domain", "parent_id", "parent_verb",
        "aliases", "semantic_type", "cardinality", "plural",
        "tags", "owner", "status", "provenance", "session_id", "user_id",
        "created_at", "updated_at", "ignored", "canonical_source",
    ]
    _glossary_columns_cache: str | None = None
    _glossary_columns_list_cache: list[str] | None = None

    @property
    def _GLOSSARY_COLUMNS(self) -> str:
        if self._glossary_columns_cache is not None:
            return self._glossary_columns_cache
        try:
            rows = self._conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'glossary_terms'"
            ).fetchall()
            actual = {r[0] for r in rows}
            cols = [c for c in self._ALL_GLOSSARY_COLUMNS if c in actual]
            self.__class__._glossary_columns_list_cache = cols
            self.__class__._glossary_columns_cache = ", ".join(cols)
            return self._glossary_columns_cache
        except Exception:
            self.__class__._glossary_columns_list_cache = list(self._ALL_GLOSSARY_COLUMNS)
            self.__class__._glossary_columns_cache = ", ".join(self._ALL_GLOSSARY_COLUMNS)
            return self._glossary_columns_cache

    def _term_from_row(self, row) -> GlossaryTerm:
        import json
        # Build dict from positional tuple using the actual queried columns
        # Ensure cache is populated
        _ = self._GLOSSARY_COLUMNS
        cols = self._glossary_columns_list_cache or self._ALL_GLOSSARY_COLUMNS
        d = {}
        for i, val in enumerate(row):
            if i < len(cols):
                d[cols[i]] = val
        aliases_json = d.get("aliases")
        tags_json = d.get("tags")
        aliases = json.loads(aliases_json) if aliases_json else []
        tags = json.loads(tags_json) if tags_json else {}
        return GlossaryTerm(
            id=d.get("id", ""),
            name=d.get("name", ""),
            display_name=d.get("display_name", ""),
            definition=d.get("definition", ""),
            domain=d.get("domain"),
            parent_id=d.get("parent_id"),
            parent_verb=d.get("parent_verb") or "HAS_KIND",
            aliases=aliases,
            semantic_type=d.get("semantic_type"),
            cardinality=d.get("cardinality") or "many",
            plural=d.get("plural"),
            tags=tags,
            owner=d.get("owner"),
            status=d.get("status") or "draft",
            provenance=d.get("provenance") or "llm",
            session_id=d.get("session_id", ""),
            user_id=d.get("user_id") or "default",
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            ignored=bool(d.get("ignored", False)),
            canonical_source=d.get("canonical_source"),
        )

    def add_glossary_term(self, term: GlossaryTerm) -> None:
        import json
        # Build column/value pairs, only for columns that exist in the table
        _ = self._GLOSSARY_COLUMNS  # ensure cache populated
        actual_cols = set((self._glossary_columns_list_cache or self._ALL_GLOSSARY_COLUMNS))
        all_pairs = [
            ("id", term.id), ("name", term.name), ("display_name", term.display_name),
            ("definition", term.definition), ("domain", term.domain),
            ("parent_id", term.parent_id), ("parent_verb", term.parent_verb),
            ("aliases", json.dumps(term.aliases)), ("semantic_type", term.semantic_type),
            ("cardinality", term.cardinality), ("plural", term.plural),
            ("tags", json.dumps(term.tags)), ("owner", term.owner),
            ("status", term.status), ("provenance", term.provenance),
            ("session_id", term.session_id), ("user_id", term.user_id or "default"),
            ("created_at", term.created_at), ("updated_at", term.updated_at),
            ("ignored", term.ignored), ("canonical_source", term.canonical_source),
        ]
        cols = [c for c, _ in all_pairs if c in actual_cols]
        vals = [v for c, v in all_pairs if c in actual_cols]
        placeholders = ", ".join("?" for _ in cols)
        self._conn.execute(
            f"INSERT OR REPLACE INTO glossary_terms ({', '.join(cols)}) VALUES ({placeholders})",
            vals,
        )
        self._clusters_dirty = True

    def update_glossary_term(self, name: str, session_id: str, updates: dict, *, user_id: str | None = None) -> bool:
        import json
        allowed = {
            "definition", "display_name", "domain", "parent_id", "parent_verb",
            "aliases", "semantic_type", "cardinality", "plural",
            "tags", "owner", "status", "provenance", "ignored", "canonical_source",
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
            where = "LOWER(name) = LOWER(?) AND user_id = ?"
        else:
            params.extend([name, session_id])
            where = "LOWER(name) = LOWER(?) AND session_id = ?"

        import time
        import duckdb
        sql = f"UPDATE glossary_terms SET {', '.join(sets)} WHERE {where} RETURNING id"
        logger.debug(f"update_glossary_term: sql={sql}, params={params}")
        for attempt in range(5):
            try:
                results = self._conn.execute(sql, params).fetchall()
                logger.debug(f"update_glossary_term: results={results}")
                updated = len(results) > 0
                if updated:
                    self._clusters_dirty = True
                return updated
            except duckdb.TransactionException:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def delete_glossary_term(self, name: str, session_id: str, *, user_id: str | None = None) -> bool:
        if user_id:
            results = self._conn.execute(
                "DELETE FROM glossary_terms WHERE LOWER(name) = LOWER(?) AND user_id = ? RETURNING id",
                [name, user_id],
            ).fetchall()
        else:
            results = self._conn.execute(
                "DELETE FROM glossary_terms WHERE LOWER(name) = LOWER(?) AND session_id = ? RETURNING id",
                [name, session_id],
            ).fetchall()
        deleted = len(results) > 0
        if deleted:
            self._clusters_dirty = True
        return deleted

    def get_glossary_term(self, name: str, session_id: str, *, user_id: str | None = None) -> GlossaryTerm | None:
        if user_id:
            row = self._conn.execute(
                f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE LOWER(name) = LOWER(?) AND user_id = ?",
                [name, user_id],
            ).fetchone()
        else:
            row = self._conn.execute(
                f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE LOWER(name) = LOWER(?) AND session_id = ?",
                [name, session_id],
            ).fetchone()
        return self._term_from_row(row) if row else None

    def get_glossary_term_by_id(self, term_id: str) -> GlossaryTerm | None:
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
        entity_where, entity_params = self.entity_visibility_filter(
            session_id, active_domains, alias="e", cross_session=True,
        )
        entity_where2, entity_params2 = self.entity_visibility_filter(
            session_id, active_domains, alias="e2", cross_session=True,
        )

        glossary_scope_col = "user_id" if user_id else "session_id"
        glossary_scope_val = user_id if user_id else session_id

        params: list = (
            [glossary_scope_val] + entity_params + [glossary_scope_val]
            + [glossary_scope_val] + entity_params2 + [glossary_scope_val]
        )

        scope_filter_1 = ""
        scope_filter_2 = ""
        if scope == "defined":
            scope_filter_1 = "AND g.id IS NOT NULL"
            scope_filter_2 = ""
        elif scope == "self_describing":
            scope_filter_1 = "AND g.id IS NULL"
            scope_filter_2 = "AND 1=0"

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
                g.status,
                g.provenance,
                CASE
                    WHEN g.id IS NOT NULL THEN 'defined'
                    ELSE 'self_describing'
                END AS glossary_status,
                e.domain_id AS entity_domain_id,
                COALESCE(g.ignored, FALSE) AS ignored
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
                g.status,
                g.provenance,
                'defined' AS glossary_status,
                NULL AS entity_domain_id,
                COALESCE(g.ignored, FALSE) AS ignored
            FROM glossary_terms g
            WHERE g.{glossary_scope_col} = ?
            {scope_filter_2}
            AND NOT EXISTS (
                SELECT 1 FROM entities e2
                WHERE LOWER(e2.name) = LOWER(g.name) AND {entity_where2}
            )
            AND (
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

        _seen: dict[str, int] = {}
        _unique: list = []
        for row in rows:
            name_lower = row[1].lower()
            if name_lower in _seen:
                idx = _seen[name_lower]
                if row[6] is not None and _unique[idx][6] is None:
                    _unique[idx] = row
                continue
            _seen[name_lower] = len(_unique)
            _unique.append(row)
        rows = _unique

        results = []
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
             parent_verb, aliases_json, cardinality, plural,
             status, provenance, glossary_status, entity_domain_id,
             ignored) = row
            aliases = json.loads(aliases_json) if aliases_json else []

            if glossary_status == "self_describing" and name.lower() in alias_set:
                continue

            results.append({
                "entity_id": entity_id,
                "name": name.lower(),
                "display_name": display_name,
                "semantic_type": semantic_type,
                "ner_type": ner_type,
                "session_id": sess_id,
                "glossary_id": glossary_id,
                "domain": domain,
                "entity_domain_id": entity_domain_id,
                "definition": definition,
                "parent_id": parent_id,
                "parent_verb": parent_verb or "HAS_KIND",
                "aliases": aliases,
                "cardinality": cardinality,
                "plural": plural,
                "status": status,
                "provenance": provenance,
                "glossary_status": glossary_status,
                "ignored": bool(ignored),
            })
        return results

    def get_deprecated_glossary(
        self,
        session_id: str,
        active_domains: list[str] | None = None,
        *,
        user_id: str | None = None,
    ) -> list[GlossaryTerm]:
        entity_vis, vis_params = self.entity_visibility_filter(
            session_id, active_domains, alias="e", cross_session=True,
        )

        all_terms = self.list_glossary_terms(session_id, user_id=user_id)
        if not all_terms:
            return []

        by_id = {t.id: t for t in all_terms}

        # parent_id can reference glossary_id OR entity_id.
        # Build a mapping from any ref ID → glossary term ID so lookups work.
        ref_to_term: dict[str, GlossaryTerm] = dict(by_id)
        for t in all_terms:
            row = self._conn.execute(
                "SELECT id FROM entities WHERE LOWER(name) = LOWER(?) LIMIT 1",
                [t.name],
            ).fetchone()
            if row and row[0] not in ref_to_term:
                ref_to_term[row[0]] = t

        children_of: dict[str, list[str]] = {}
        for t in all_terms:
            if t.parent_id:
                parent = ref_to_term.get(t.parent_id)
                if parent:
                    children_of.setdefault(parent.id, []).append(t.id)

        grounded: set[str] = set()
        for t in all_terms:
            row = self._conn.execute(
                f"SELECT 1 FROM entities e WHERE LOWER(e.name) = LOWER(?) AND {entity_vis} LIMIT 1",
                [t.name] + vis_params,
            ).fetchone()
            if row:
                grounded.add(t.id)

        valid = set(grounded)
        for tid in grounded:
            current = by_id.get(tid)
            while current and current.parent_id:
                parent = ref_to_term.get(current.parent_id)
                if not parent or parent.id in valid:
                    break
                valid.add(parent.id)
                current = parent

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

    def delete_glossary_term_cascade(
        self,
        name: str,
        session_id: str,
        active_domains: list[str] | None = None,
        *,
        user_id: str | None = None,
    ) -> dict:
        term = self.get_glossary_term(name, session_id, user_id=user_id)
        if not term:
            return {}

        former_parent_id = term.parent_id

        children = self.get_child_terms(term.id)
        reparented = []
        for child in children:
            self._conn.execute(
                "UPDATE glossary_terms SET parent_id = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                [child.id],
            )
            reparented.append(child.name)

        self.delete_glossary_term(name, session_id, user_id=user_id)

        deprecated_names: list[str] = []
        if former_parent_id:
            deprecated_terms = self.get_deprecated_glossary(
                session_id, active_domains, user_id=user_id,
            )
            deprecated_ids = {t.id for t in deprecated_terms}
            current_id = former_parent_id
            while current_id:
                ancestor = self.get_glossary_term_by_id(current_id)
                if not ancestor:
                    break
                if ancestor.id in deprecated_ids:
                    deprecated_names.append(ancestor.name)
                current_id = ancestor.parent_id

        return {
            "deleted": name,
            "reparented": reparented,
            "deprecated": deprecated_names,
        }

    def rename_glossary_term(
        self,
        old_name: str,
        new_name: str,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> dict:
        from constat.discovery.models import display_entity_name

        term = self.get_glossary_term(old_name, session_id, user_id=user_id)
        if not term:
            raise ValueError(f"Term '{old_name}' not found")

        entity_row = self._conn.execute(
            "SELECT 1 FROM entities WHERE LOWER(name) = LOWER(?) AND session_id = ? LIMIT 1",
            [old_name, session_id],
        ).fetchone()
        if entity_row:
            raise ValueError(f"Term '{old_name}' is grounded to an entity — name is immutable")

        existing = self.get_glossary_term(new_name, session_id, user_id=user_id)
        if existing:
            raise ValueError(f"Term '{new_name}' already exists")

        new_display = display_entity_name(new_name)

        if user_id:
            self._conn.execute(
                "UPDATE glossary_terms SET name = ?, display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE LOWER(name) = LOWER(?) AND user_id = ?",
                [new_name.lower(), new_display, old_name, user_id],
            )
        else:
            self._conn.execute(
                "UPDATE glossary_terms SET name = ?, display_name = ?, updated_at = CURRENT_TIMESTAMP WHERE LOWER(name) = LOWER(?) AND session_id = ?",
                [new_name.lower(), new_display, old_name, session_id],
            )

        rel_count = 0
        result = self._conn.execute(
            "UPDATE entity_relationships SET subject_name = ? WHERE LOWER(subject_name) = LOWER(?) AND session_id = ? RETURNING id",
            [new_name.lower(), old_name, session_id],
        ).fetchall()
        rel_count += len(result)
        result = self._conn.execute(
            "UPDATE entity_relationships SET object_name = ? WHERE LOWER(object_name) = LOWER(?) AND session_id = ? RETURNING id",
            [new_name.lower(), old_name, session_id],
        ).fetchall()
        rel_count += len(result)

        self._clusters_dirty = True
        return {
            "old_name": old_name,
            "new_name": new_name.lower(),
            "display_name": new_display,
            "relationships_updated": rel_count,
        }

    def clear_session_glossary(self, session_id: str, *, user_id: str | None = None) -> int:
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
        if count:
            self._clusters_dirty = True
        logger.debug(f"clear_session_glossary({session_id}, user_id={user_id}): deleted {count} terms")
        return count

    def delete_glossary_by_status(
        self, session_id: str, status: str, *, user_id: str | None = None,
    ) -> int:
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
        if count:
            self._clusters_dirty = True
        logger.debug(f"delete_glossary_by_status({status}, user_id={user_id}): deleted {count} terms")
        return count

    def get_glossary_terms_by_names(self, names: list[str], session_id: str, *, user_id: str | None = None) -> list[GlossaryTerm]:
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
        import json

        scope_col = "user_id" if user_id else "session_id"
        scope_val = user_id if user_id else session_id

        row = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms "
            f"WHERE LOWER(name) = LOWER(?) AND {scope_col} = ?",
            [name, scope_val],
        ).fetchone()
        if row:
            return self._term_from_row(row)

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
        all_ids = [parent_id] + [i for i in extra_ids if i]
        placeholders = ", ".join("?" for _ in all_ids)
        rows = self._conn.execute(
            f"SELECT {self._GLOSSARY_COLUMNS} FROM glossary_terms WHERE parent_id IN ({placeholders})",
            all_ids,
        ).fetchall()
        return [self._term_from_row(r) for r in rows]

    def reconcile_glossary_domains(self, session_id: str, *, user_id: str | None = None) -> list[dict]:
        scope_col = "user_id" if user_id else "session_id"
        scope_val = user_id or session_id

        rows = self._conn.execute(f"""
            SELECT g.name, g.domain, e.domain_id
            FROM glossary_terms g
            JOIN entities e ON LOWER(g.name) = LOWER(e.name)
            WHERE g.{scope_col} = ?
              AND e.domain_id IS NOT NULL
              AND e.domain_id != ''
              AND g.domain IS NOT NULL
              AND g.domain != ''
              AND g.domain != e.domain_id
        """, [scope_val]).fetchall()

        moved = []
        for name, old_domain, new_domain in rows:
            self._conn.execute(
                f"UPDATE glossary_terms SET domain = ? WHERE LOWER(name) = LOWER(?) AND {scope_col} = ?",
                [new_domain, name, scope_val],
            )
            moved.append({"name": name, "from_domain": old_domain, "to_domain": new_domain})

        if moved:
            self._clusters_dirty = True
            logger.info(f"reconcile_glossary_domains: moved {len(moved)} terms: {moved}")

        return moved

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    def add_entity_relationship(self, rel) -> None:
        self._conn.execute(
            """
            INSERT INTO entity_relationships
                (id, subject_name, verb, object_name,
                 sentence, confidence, verb_category, session_id, user_edited)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            """,
            [
                rel.id, rel.subject_name, rel.verb,
                rel.object_name, rel.sentence,
                rel.confidence, rel.verb_category, rel.session_id,
                getattr(rel, 'user_edited', False),
            ],
        )

    def get_relationships_for_entity(
        self, entity_name: str, session_id: str,
    ) -> list[dict]:
        name_lower = entity_name.lower()
        rows = self._conn.execute(
            """
            SELECT FIRST(r.id), r.subject_name, r.verb, r.object_name,
                   MAX(r.confidence), FIRST(r.verb_category),
                   BOOL_OR(r.user_edited)
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
                "user_edited": r[6],
            }
            for r in rows
        ]

    def clear_session_relationships(self, session_id: str) -> int:
        result = self._conn.execute(
            "DELETE FROM entity_relationships WHERE session_id = ? RETURNING id",
            [session_id],
        ).fetchall()
        return len(result)

    def clear_non_user_relationships(self, session_id: str) -> int:
        result = self._conn.execute(
            "DELETE FROM entity_relationships WHERE session_id = ? AND user_edited = FALSE RETURNING id",
            [session_id],
        ).fetchall()
        return len(result)

    def delete_entity_relationship(self, rel_id: str) -> bool:
        result = self._conn.execute(
            "DELETE FROM entity_relationships WHERE id = ? RETURNING id",
            [rel_id],
        ).fetchall()
        return len(result) > 0

    def update_entity_relationship_verb(self, rel_id: str, verb: str) -> bool:
        from constat.discovery.relationship_extractor import categorize_verb
        verb = verb.upper()
        verb_category = categorize_verb(verb)
        result = self._conn.execute(
            "UPDATE entity_relationships SET verb = ?, verb_category = ?, user_edited = TRUE WHERE id = ? RETURNING id",
            [verb, verb_category, rel_id],
        ).fetchall()
        return len(result) > 0

    # ------------------------------------------------------------------
    # Hashes
    # ------------------------------------------------------------------

    def get_source_hash(self, source_id: str, hash_type: str) -> str | None:
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
        if hash_type not in ('db', 'api', 'doc'):
            raise ValueError(f"Invalid hash_type: {hash_type}")
        column = f"{hash_type}_hash"
        self._conn.execute("""
            INSERT INTO source_hashes (source_id) VALUES (?)
            ON CONFLICT (source_id) DO NOTHING
        """, [source_id])
        self._conn.execute(f"""
            UPDATE source_hashes SET {column} = ?, updated_at = CURRENT_TIMESTAMP
            WHERE source_id = ?
        """, [config_hash, source_id])
        logger.debug(f"set_source_hash({source_id}, {hash_type}): {config_hash}")

    def get_domain_config_hash(self, domain_id: str) -> str | None:
        return self.get_source_hash(domain_id, 'doc')

    def set_domain_config_hash(self, domain_id: str, config_hash: str) -> None:
        self.set_source_hash(domain_id, 'doc', config_hash)

    @staticmethod
    def _make_resource_id(source_id: str, resource_type: str, resource_name: str) -> str:
        return f"{source_id}:{resource_type}:{resource_name}"

    def get_resource_hash(
        self,
        source_id: str,
        resource_type: str,
        resource_name: str,
    ) -> str | None:
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
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        self._conn.execute("""
            INSERT INTO resource_hashes (resource_id, resource_type, resource_name, source_id, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (resource_id) DO UPDATE SET
                content_hash = excluded.content_hash
        """, [resource_id, resource_type, resource_name, source_id, content_hash])
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
        resource_id = self._make_resource_id(source_id, resource_type, resource_name)
        results = self._conn.execute(
            "DELETE FROM resource_hashes WHERE resource_id = ? RETURNING resource_id",
            [resource_id],
        ).fetchall()
        deleted = len(results) > 0
        logger.debug(f"delete_resource_hash({resource_id}): deleted={deleted}")
        return deleted

    def get_resource_hashes_for_source(
        self,
        source_id: str,
        resource_type: str | None = None,
    ) -> dict[str, str]:
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
        source_map = {
            'database': 'schema',
            'api': 'api',
            'document': 'document',
        }
        source_type = source_map.get(resource_type, resource_type)

        if source_id == '__base__':
            chunk_ids = self._conn.execute(
                """
                SELECT chunk_id FROM embeddings
                WHERE document_name = ? AND source = ?
                AND (domain_id IS NULL OR domain_id = '__base__')
                """,
                [resource_name, source_type],
            ).fetchall()
        else:
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

        placeholders = ",".join(["?" for _ in chunk_ids])
        self._conn.execute(
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )

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
        result = self._conn.execute(
            "DELETE FROM resource_hashes WHERE source_id = ? RETURNING resource_id",
            [source_id],
        ).fetchall()
        count = len(result)
        logger.debug(f"clear_resource_hashes_for_source({source_id}): deleted {count} hashes")
        return count

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def _rebuild_clusters(self, session_id: str) -> None:
        if not self._clusters_dirty:
            return

        name_to_vec: dict[str, np.ndarray] = {}

        entity_rows = self._conn.execute(
            """
            SELECT ent.name, e.embedding
            FROM chunk_entities ce
            JOIN entities ent ON ce.entity_id = ent.id
            JOIN embeddings e ON ce.chunk_id = e.chunk_id
            WHERE ent.session_id = ?
            """,
            [session_id],
        ).fetchall()

        accum: dict[str, list[np.ndarray]] = {}
        name_display: dict[str, str] = {}
        for ent_name, embedding in entity_rows:
            key = ent_name.lower()
            accum.setdefault(key, []).append(np.array(embedding, dtype=np.float32))
            name_display.setdefault(key, ent_name)
        for key, vecs in accum.items():
            name_to_vec[name_display[key]] = np.mean(vecs, axis=0)
        logger.debug(f"[_rebuild_clusters] {len(entity_rows)} entity-chunk rows -> {len(name_to_vec)} unique entity vectors")

        glossary_rows = self._conn.execute(
            """
            SELECT gt.name, e.embedding
            FROM embeddings e
            JOIN glossary_terms gt ON e.document_name = 'glossary:' || gt.id
            WHERE (e.session_id = ? OR e.session_id IS NULL)
            """,
            [session_id],
        ).fetchall()
        lower_to_key = {k.lower(): k for k in name_to_vec}
        for gt_name, embedding in glossary_rows:
            key = gt_name.lower()
            existing = lower_to_key.get(key)
            if existing and existing != gt_name:
                del name_to_vec[existing]
            name_to_vec[gt_name] = np.array(embedding, dtype=np.float32)
            lower_to_key[key] = gt_name
        logger.debug(f"[_rebuild_clusters] {len(glossary_rows)} glossary rows, total vectors: {len(name_to_vec)}")

        if len(name_to_vec) < self._cluster_min_terms:
            self._conn.execute(
                "DELETE FROM glossary_clusters WHERE session_id = ?", [session_id]
            )
            self._clusters_dirty = False
            return

        from sklearn.cluster import MiniBatchKMeans

        term_names = list(name_to_vec.keys())
        X = np.vstack([name_to_vec[n] for n in term_names])
        k = max(2, len(term_names) // self._cluster_divisor)
        if self._cluster_max_k is not None:
            k = min(k, self._cluster_max_k)
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024)
        labels = kmeans.fit_predict(X)

        self._conn.execute(
            "DELETE FROM glossary_clusters WHERE session_id = ?", [session_id]
        )
        seen: set[str] = set()
        seen_exact: set[str] = set()
        batch = []
        for name, cluster_id in zip(term_names, labels):
            key = name.lower()
            if key in seen or name in seen_exact:
                continue
            seen.add(key)
            seen_exact.add(name)
            batch.append((name, int(cluster_id), session_id))

        if batch:
            try:
                self._conn.executemany(
                    "INSERT INTO glossary_clusters (term_name, cluster_id, session_id) VALUES (?, ?, ?)",
                    batch,
                )
            except Exception as e:
                if "Duplicate" not in str(e):
                    raise
                for row in batch:
                    try:
                        self._conn.execute(
                            "INSERT INTO glossary_clusters (term_name, cluster_id, session_id) VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
                            list(row),
                        )
                    except Exception:
                        pass

        self._clusters_dirty = False

    def get_cluster_siblings(
        self, term_name: str, session_id: str, limit: int = 10,
    ) -> list[str]:
        self._rebuild_clusters(session_id)

        row = self._conn.execute(
            "SELECT cluster_id FROM glossary_clusters WHERE term_name = ? AND session_id = ?",
            [term_name, session_id],
        ).fetchone()
        if not row:
            return []

        cluster_id = row[0]
        rows = self._conn.execute(
            """
            SELECT term_name FROM glossary_clusters
            WHERE cluster_id = ? AND session_id = ? AND term_name != ?
            LIMIT ?
            """,
            [cluster_id, session_id, term_name, limit],
        ).fetchall()
        return [r[0] for r in rows]

    def find_matching_clusters(
        self, query: str, session_id: str, limit: int = 5,
    ) -> list[dict]:
        self._rebuild_clusters(session_id)

        rows = self._conn.execute(
            "SELECT DISTINCT term_name, cluster_id FROM glossary_clusters WHERE session_id = ?",
            [session_id],
        ).fetchall()

        if not rows:
            return []

        query_lower = query.lower()
        results = []
        seen_clusters = set()

        for term_name, cluster_id in rows:
            if cluster_id in seen_clusters:
                continue
            if len(term_name) > 3 and term_name.lower() in query_lower:
                siblings = self._conn.execute(
                    """
                    SELECT term_name FROM glossary_clusters
                    WHERE cluster_id = ? AND session_id = ? AND term_name != ?
                    LIMIT 10
                    """,
                    [cluster_id, session_id, term_name],
                ).fetchall()
                results.append({
                    "term": term_name,
                    "siblings": [r[0] for r in siblings],
                })
                seen_clusters.add(cluster_id)
                if len(results) >= limit:
                    break

        return results

    # ------------------------------------------------------------------
    # NER scope cache
    # ------------------------------------------------------------------

    def has_ner_scope_cache(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            "SELECT entity_count FROM ner_scope_cache WHERE fingerprint = ?",
            [fingerprint],
        ).fetchone()
        if row is None:
            return False
        if row[0] == 0:
            self._evict_ner_scope_fingerprint(fingerprint)
            logger.info(f"Evicted empty NER scope cache for fingerprint {fingerprint[:12]}...")
            return False
        return True

    def restore_ner_scope_cache(self, fingerprint: str, session_id: str) -> int:
        self._conn.execute(
            """
            DELETE FROM chunk_entities WHERE entity_id IN (
                SELECT id FROM ner_cached_entities WHERE fingerprint = ?
            )
            """,
            [fingerprint],
        )
        self._conn.execute(
            """
            DELETE FROM entities WHERE id IN (
                SELECT id FROM ner_cached_entities WHERE fingerprint = ?
            )
            """,
            [fingerprint],
        )

        self._conn.execute(
            "DELETE FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)",
            [session_id],
        )
        self._conn.execute("DELETE FROM entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM glossary_clusters WHERE session_id = ?", [session_id])

        self._conn.execute(
            """
            INSERT INTO entities (id, name, display_name, semantic_type, ner_type, session_id, domain_id, created_at)
            SELECT DISTINCT id, name, display_name, semantic_type, ner_type, ?, domain_id, created_at
            FROM ner_cached_entities WHERE fingerprint = ?
            ON CONFLICT (id) DO UPDATE SET session_id = excluded.session_id
            """,
            [session_id, fingerprint],
        )
        entity_count = self._conn.execute(
            "SELECT COUNT(DISTINCT id) FROM ner_cached_entities WHERE fingerprint = ?",
            [fingerprint],
        ).fetchone()[0]

        self._conn.execute(
            """
            INSERT INTO chunk_entities (chunk_id, entity_id, confidence)
            SELECT DISTINCT chunk_id, entity_id, confidence
            FROM ner_cached_chunk_entities WHERE fingerprint = ?
            ON CONFLICT DO NOTHING
            """,
            [fingerprint],
        )

        self._conn.execute(
            """
            INSERT INTO glossary_clusters (term_name, cluster_id, session_id)
            SELECT DISTINCT term_name, cluster_id, ?
            FROM ner_cached_clusters WHERE fingerprint = ?
            ON CONFLICT DO NOTHING
            """,
            [session_id, fingerprint],
        )

        self._clusters_dirty = False
        logger.info(f"Restored NER scope cache: {entity_count} entities for fingerprint {fingerprint[:12]}...")
        return entity_count

    def store_ner_scope_cache(self, fingerprint: str, session_id: str) -> None:
        entity_count = self._conn.execute(
            "SELECT COUNT(*) FROM entities WHERE session_id = ?",
            [session_id],
        ).fetchone()[0]

        if entity_count == 0:
            logger.warning(f"Not caching NER scope with 0 entities for fingerprint {fingerprint[:12]}...")
            return

        self._evict_ner_scope_fingerprint(fingerprint)

        self._conn.execute(
            "INSERT INTO ner_scope_cache (fingerprint, entity_count) VALUES (?, ?) ON CONFLICT (fingerprint) DO UPDATE SET entity_count = excluded.entity_count",
            [fingerprint, entity_count],
        )

        self._conn.execute(
            """
            INSERT INTO ner_cached_entities (fingerprint, id, name, display_name, semantic_type, ner_type, domain_id, created_at)
            SELECT ?, id, name, display_name, semantic_type, ner_type, domain_id, created_at
            FROM entities WHERE session_id = ?
            ON CONFLICT DO NOTHING
            """,
            [fingerprint, session_id],
        )

        self._conn.execute(
            """
            INSERT INTO ner_cached_chunk_entities (fingerprint, chunk_id, entity_id, confidence)
            SELECT ?, ce.chunk_id, ce.entity_id, ce.confidence
            FROM chunk_entities ce
            JOIN entities e ON ce.entity_id = e.id
            WHERE e.session_id = ?
            ON CONFLICT DO NOTHING
            """,
            [fingerprint, session_id],
        )

        self._conn.execute(
            """
            INSERT INTO ner_cached_clusters (fingerprint, term_name, cluster_id)
            SELECT ?, term_name, cluster_id
            FROM glossary_clusters WHERE session_id = ?
            ON CONFLICT DO NOTHING
            """,
            [fingerprint, session_id],
        )

        logger.info(f"Stored NER scope cache: {entity_count} entities for fingerprint {fingerprint[:12]}...")

    def evict_ner_scope_cache(self, keep: int = 10) -> None:
        rows = self._conn.execute(
            "SELECT fingerprint FROM ner_scope_cache ORDER BY created_at DESC OFFSET ?",
            [keep],
        ).fetchall()
        for (fp,) in rows:
            self._evict_ner_scope_fingerprint(fp)

    def _evict_ner_scope_fingerprint(self, fingerprint: str) -> None:
        self._conn.execute("DELETE FROM ner_cached_clusters WHERE fingerprint = ?", [fingerprint])
        self._conn.execute("DELETE FROM ner_cached_chunk_entities WHERE fingerprint = ?", [fingerprint])
        self._conn.execute("DELETE FROM ner_cached_entities WHERE fingerprint = ?", [fingerprint])
        self._conn.execute("DELETE FROM ner_scope_cache WHERE fingerprint = ?", [fingerprint])

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def clear_session_data(self, session_id: str, fts_dirty_callback=None) -> None:
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

        self._conn.execute("DELETE FROM chunk_entities WHERE entity_id IN (SELECT id FROM entities WHERE session_id = ?)", [session_id])
        self._conn.execute("DELETE FROM entities WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM embeddings WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM glossary_terms WHERE session_id = ?", [session_id])
        self._conn.execute("DELETE FROM entity_relationships WHERE session_id = ?", [session_id])

        self._conn.execute("DELETE FROM glossary_clusters WHERE session_id = ?", [session_id])
        self._clusters_dirty = True
        if fts_dirty_callback:
            fts_dirty_callback()
        logger.debug(f"clear_session_data({session_id}): deleted session data")

    def delete_document(self, document_name: str, session_id: str | None = None, fts_dirty_callback=None) -> int:
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
            f"DELETE FROM chunk_entities WHERE chunk_id IN ({placeholders})",
            chunk_ids
        )

        if session_id:
            self._conn.execute("""
                DELETE FROM entities
                WHERE session_id = ?
                AND id NOT IN (SELECT DISTINCT entity_id FROM chunk_entities)
            """, [session_id])

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

        if fts_dirty_callback:
            fts_dirty_callback()
        logger.debug(f"delete_document({document_name}, {session_id}): deleted {len(chunk_ids)} chunks")
        return len(chunk_ids)

    # ------------------------------------------------------------------
    # Phase 2: caller-facing query methods
    # ------------------------------------------------------------------

    def get_entity_document_names(self, entity_id: str, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT em.document_name FROM chunk_entities ce "
            "JOIN embeddings em ON ce.chunk_id = em.chunk_id "
            "WHERE ce.entity_id = ? LIMIT ?",
            [entity_id, limit],
        ).fetchall()
        return [r[0] for r in rows]

    def get_cooccurring_entities(
        self, entity_id: str, session_id: str, limit: int = 5,
    ) -> list[dict]:
        rows = self._conn.execute("""
            SELECT e2.name, e2.semantic_type, COUNT(*) as co_occurrences
            FROM chunk_entities ce1
            JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
            JOIN entities e2 ON ce2.entity_id = e2.id
            WHERE ce1.entity_id = ?
              AND ce2.entity_id != ce1.entity_id
              AND (e2.session_id IS NULL OR e2.session_id = ?)
            GROUP BY e2.id, e2.name, e2.semantic_type
            ORDER BY co_occurrences DESC
            LIMIT ?
        """, [entity_id, session_id, limit]).fetchall()
        return [
            {"name": r[0], "type": r[1] or "concept", "co_occurrences": r[2]}
            for r in rows
        ]

    def get_cooccurrence_pairs(
        self,
        session_id: str,
        min_count: int = 2,
        limit: int | None = None,
        include_types: bool = False,
    ) -> list[tuple]:
        if include_types:
            sql = """
                SELECT e1.id, e1.name, e1.semantic_type,
                       e2.id, e2.name, e2.semantic_type,
                       COUNT(*) as co_count
                FROM chunk_entities ce1
                JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id AND ce1.entity_id < ce2.entity_id
                JOIN entities e1 ON ce1.entity_id = e1.id
                JOIN entities e2 ON ce2.entity_id = e2.id
                WHERE e1.session_id = ? AND e2.session_id = ?
                GROUP BY e1.id, e1.name, e1.semantic_type, e2.id, e2.name, e2.semantic_type
                HAVING COUNT(*) >= ?
                ORDER BY co_count DESC
            """
        else:
            sql = """
                SELECT e1.id, e1.name, e2.id, e2.name, COUNT(*) as co_count
                FROM chunk_entities ce1
                JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id AND ce1.entity_id < ce2.entity_id
                JOIN entities e1 ON ce1.entity_id = e1.id
                JOIN entities e2 ON ce2.entity_id = e2.id
                WHERE e1.session_id = ? AND e2.session_id = ?
                GROUP BY e1.id, e1.name, e2.id, e2.name
                HAVING COUNT(*) >= ?
                ORDER BY co_count DESC
            """
        params: list = [session_id, session_id, min_count]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._conn.execute(sql, params).fetchall()

    def get_cooccurrence_pairs_by_name(
        self,
        session_id: str,
        min_count: int = 3,
    ) -> list[tuple]:
        return self._conn.execute("""
            SELECT e1.name, e2.name, COUNT(*) as co_count
            FROM chunk_entities ce1
            JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id AND ce1.entity_id < ce2.entity_id
            JOIN entities e1 ON ce1.entity_id = e1.id
            JOIN entities e2 ON ce2.entity_id = e2.id
            WHERE e1.session_id = ? AND e2.session_id = ?
            GROUP BY e1.name, e2.name
            HAVING COUNT(*) >= ?
            ORDER BY co_count DESC
        """, [session_id, session_id, min_count]).fetchall()

    def get_shared_chunk_ids(
        self, e1_id: str, e2_id: str, limit: int = 10,
    ) -> list[str]:
        rows = self._conn.execute("""
            SELECT DISTINCT ce1.chunk_id
            FROM chunk_entities ce1
            JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
            WHERE ce1.entity_id = ? AND ce2.entity_id = ?
            LIMIT ?
        """, [e1_id, e2_id, limit]).fetchall()
        return [r[0] for r in rows]

    def get_entities_with_stats(
        self, vis_filter: str, vis_params: list,
    ) -> list[tuple]:
        return self._conn.execute(f"""
            WITH entity_stats AS (
                SELECT
                    ce.entity_id,
                    COUNT(*) as ref_count,
                    COUNT(DISTINCT em.source) as source_count,
                    LIST(DISTINCT CASE WHEN em.source = 'document' THEN em.document_name END) as doc_names
                FROM chunk_entities ce
                JOIN embeddings em ON ce.chunk_id = em.chunk_id
                GROUP BY ce.entity_id
            )
            SELECT
                e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                COALESCE(es.ref_count, 0) as ref_count,
                COALESCE(es.source_count, 0) as source_count,
                es.doc_names
            FROM entities e
            LEFT JOIN entity_stats es ON es.entity_id = e.id
            WHERE {vis_filter}
            ORDER BY e.name
        """, vis_params).fetchall()

    def get_visible_entity_names(
        self, vis_filter: str, vis_params: list,
    ) -> list[str]:
        rows = self._conn.execute(
            f"SELECT LOWER(e.name) FROM entities e WHERE {vis_filter}",
            vis_params,
        ).fetchall()
        return [r[0] for r in rows]

    def update_entity_name(
        self, entity_id: str, name: str, display_name: str,
    ) -> None:
        self._conn.execute(
            "UPDATE entities SET name = ?, display_name = ? WHERE id = ?",
            [name, display_name, entity_id],
        )

    def mark_relationship_user_edited(self, rel_id: str) -> bool:
        result = self._conn.execute(
            "UPDATE entity_relationships SET user_edited = TRUE WHERE id = ? RETURNING id",
            [rel_id],
        ).fetchall()
        return len(result) > 0

    def list_session_relationships(self, session_id: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT id, subject_name, verb, object_name, sentence, confidence, user_edited
            FROM entity_relationships
            WHERE session_id = ?
            ORDER BY user_edited DESC, confidence DESC
            """,
            [session_id],
        ).fetchall()
        return [
            {
                "id": r[0], "subject_name": r[1], "verb": r[2],
                "object_name": r[3], "sentence": r[4] or "",
                "confidence": r[5], "user_edited": r[6],
            }
            for r in rows
        ]

    def get_promotable_relationships(self, session_id: str) -> list[tuple]:
        return self._conn.execute(
            """
            SELECT r.id, r.subject_name, r.verb, r.object_name
            FROM entity_relationships r
            WHERE r.session_id = ?
              AND r.user_edited = FALSE
              AND r.verb IN ('HAS_ONE', 'HAS_MANY', 'HAS_KIND')
            """,
            [session_id],
        ).fetchall()

    def get_glossary_parent_child_pairs(self, session_id: str) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            """
            SELECT child.name, parent.name
            FROM glossary_terms child
            JOIN glossary_terms parent ON child.parent_id = parent.id
            WHERE child.session_id = ?
            """,
            [session_id],
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def list_entities_with_refcount(
        self, vis_filter: str, vis_params: list,
    ) -> list[tuple]:
        return self._conn.execute(f"""
            SELECT e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                   (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as ref_count
            FROM entities e
            WHERE {vis_filter}
            ORDER BY e.name
        """, vis_params).fetchall()

    def get_entity_references(
        self, entity_id: str, limit: int = 10,
    ) -> list[tuple]:
        return self._conn.execute("""
            SELECT em.document_name, em.section, ce.confidence
            FROM chunk_entities ce
            JOIN embeddings em ON ce.chunk_id = em.chunk_id
            WHERE ce.entity_id = ?
            ORDER BY ce.confidence DESC
            LIMIT ?
        """, [entity_id, limit]).fetchall()

    def count_session_links(self, session_id: str) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM chunk_entities ce JOIN entities e ON ce.entity_id = e.id WHERE e.session_id = ?",
            [session_id],
        ).fetchone()[0]

    def entity_exists(self, name: str, session_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM entities WHERE LOWER(name) = LOWER(?) AND session_id = ? LIMIT 1",
            [name, session_id],
        ).fetchone()
        return row is not None

    def get_non_ignored_entities_for_chunk(
        self, chunk_id: str, session_id: str | None = None,
    ) -> list[tuple]:
        parts = ["ce.chunk_id = ?"]
        params: list = [chunk_id]
        if session_id:
            parts.append("e.session_id = ?")
            params.append(session_id)
        where = " AND ".join(parts)
        return self._conn.execute(
            f"""
            SELECT e.id, e.name, e.semantic_type
            FROM chunk_entities ce
            JOIN entities e ON ce.entity_id = e.id
            LEFT JOIN glossary_terms g ON g.entity_id = e.id
            WHERE {where}
              AND COALESCE(g.ignored, FALSE) = FALSE
            """,
            params,
        ).fetchall()

    # ------------------------------------------------------------------
    # Proven grounding
    # ------------------------------------------------------------------

    def save_proven_grounding(self, entity_name: str, source_patterns: list[str]) -> None:
        for pattern in source_patterns:
            self._conn.execute(
                "INSERT INTO proven_grounding (entity_name, source_pattern) "
                "VALUES (?, ?) ON CONFLICT DO NOTHING",
                [entity_name, pattern],
            )

    def get_proven_grounding(self, entity_name: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT source_pattern FROM proven_grounding WHERE entity_name = ?",
            [entity_name],
        ).fetchall()
        return [r[0] for r in rows]
