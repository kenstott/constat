# Copyright (c) 2025 Kenneth Stott
# Canary: 8b903a45-d451-43b0-9e4d-24be65b927f8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Sync entities from chonk global store into constat's session-scoped entity table."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _entity_id(name: str, session_id: str) -> str:
    normalized = name.strip().lower().replace("_", " ")
    return hashlib.sha256(f"{normalized}:{session_id}".encode()).hexdigest()[:16]


def sync_entities_from_chonk(
    chonk_store,
    vector_store,
    domain_names: list[str],
    session_id: str,
) -> int:
    """Sync chonk entities for active domains into constat's session entity table.

    Reads entities and their domain associations from chonk_global.duckdb,
    maps to constat's Entity model, and upserts into system.duckdb scoped to
    session_id.  Returns count of entities written.
    """
    from constat.discovery.models import Entity

    if not domain_names:
        return 0

    chonk_domain_ids = [f"global:{name}" for name in domain_names]
    conn = chonk_store.vector._conn

    ph = ", ".join(["?" for _ in chonk_domain_ids])
    rows = conn.execute(
        f"""
        SELECT DISTINCT ent.id, ent.name, ent.display_name, ent.entity_type, src.domain_id
        FROM entities ent
        JOIN chunk_entities ce ON ent.id = ce.entity_id
        JOIN embeddings emb ON ce.chunk_id = emb.chunk_id
        LEFT JOIN sources src ON (
            emb.source_id = src.source_id
            OR src.source_id LIKE emb.chunk_id || '%'
            OR emb.chunk_id LIKE 'schema:' || REPLACE(src.source_id, ':', '.') || '%'
        )
        WHERE src.domain_id IN ({ph}) OR src.domain_id IS NULL
        """,
        chonk_domain_ids,
    ).fetchall()

    if not rows:
        # Fallback: no domain filter — return all entities from chonk
        rows = conn.execute(
            """
            SELECT DISTINCT ent.id, ent.name, ent.display_name, ent.entity_type, NULL AS domain_id
            FROM entities ent
            JOIN chunk_entities ce ON ent.id = ce.entity_id
            JOIN embeddings emb ON ce.chunk_id = emb.chunk_id
            """
        ).fetchall()

    if not rows:
        return 0

    entities = []
    entity_id_map: dict[str, str] = {}  # chonk entity_id → constat entity_id
    for chonk_eid, name, display_name, entity_type, domain_id in rows:
        domain_name = domain_id.replace("global:", "", 1) if domain_id else domain_names[0]
        constat_eid = _entity_id(name, session_id)
        entity_id_map[chonk_eid] = constat_eid
        entities.append(Entity(
            id=constat_eid,
            name=name,
            display_name=display_name,
            semantic_type=entity_type or "concept",
            ner_type="CHONK",
            session_id=session_id,
            domain_id=domain_name,
            created_at=datetime.now(),
            entity_class="metadata",
        ))

    vector_store.add_entities(entities, session_id=session_id)

    # Create chunk_entities links in constat's vector store so glossary can see these entities
    from constat.discovery.models import ChunkEntity
    link_rows = conn.execute(
        "SELECT entity_id, chunk_id, score FROM chunk_entities WHERE entity_id IN ({})".format(
            ", ".join("?" for _ in entity_id_map)
        ),
        list(entity_id_map.keys()),
    ).fetchall()
    links = []
    for chonk_eid, chunk_id, score in link_rows:
        constat_eid = entity_id_map.get(chonk_eid)
        if constat_eid:
            links.append(ChunkEntity(
                chunk_id=chunk_id,
                entity_id=constat_eid,
                confidence=float(score) if score else 0.8,
            ))
    if links:
        vector_store.link_chunk_entities(links)

    logger.info(f"chonk entity sync: {len(entities)} entities → session {session_id}")
    return len(entities)
