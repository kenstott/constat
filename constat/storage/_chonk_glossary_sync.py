# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Sync chonk entity descriptions and SVO hierarchy into constat glossary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.discovery.vector_store import DuckDBVectorBackend

logger = logging.getLogger(__name__)

# SVO verbs that imply parent→child (subject IS child of object)
_HIERARCHY_FORWARD: frozenset[str] = frozenset({
    "type_of", "instance_of", "classified_as", "part_of", "member_of", "extends",
})
# SVO verbs that imply parent→child in reverse (subject IS parent of object)
_HIERARCHY_REVERSE: frozenset[str] = frozenset({"contains", "composed_of"})

_ALL_HIERARCHY_VERBS: list[str] = list(_HIERARCHY_FORWARD | _HIERARCHY_REVERSE)

_PARENT_VERB_MAP: dict[str, str] = {
    "type_of": "HAS_KIND",
    "instance_of": "HAS_KIND",
    "classified_as": "HAS_KIND",
    "part_of": "HAS_ONE",
    "member_of": "HAS_ONE",
    "extends": "HAS_ONE",
    "contains": "HAS_ONE",
    "composed_of": "HAS_ONE",
}


def sync_entity_descriptions_to_glossary(
    chonk_store,
    vector_store: "DuckDBVectorBackend",
    domain_names: list[str],
    session_id: str,
    user_id: str,
) -> int:
    """Upsert chonk entity descriptions as draft GlossaryTerms.

    Reads entity descriptions and SVO hierarchy triples from chonk_store for
    the given domain_names, then upserts them into constat's glossary with
    provenance='chonk_llm' and status='draft'. Skips terms already curated by
    a human (provenance='human').

    Returns count of terms written.
    """
    from constat.discovery.models import GlossaryTerm

    if not domain_names:
        return 0

    chonk_domain_ids = [f"global:{name}" for name in domain_names]
    conn = chonk_store.vector._conn

    ph_d = ", ".join(["?" for _ in chonk_domain_ids])
    entity_domain_rows = conn.execute(
        f"SELECT DISTINCT ce.entity_id, e.domain_id "
        f"FROM chunk_entities ce "
        f"JOIN embeddings e ON ce.chunk_id = e.chunk_id "
        f"WHERE e.domain_id IN ({ph_d})",
        chonk_domain_ids,
    ).fetchall()

    if not entity_domain_rows:
        return 0

    entity_domain_map: dict[str, str] = {}
    for entity_id, domain_id in entity_domain_rows:
        if entity_id not in entity_domain_map:
            entity_domain_map[entity_id] = domain_id.replace("global:", "", 1) if domain_id else domain_names[0]

    entity_ids = list(entity_domain_map.keys())
    descriptions = chonk_store.get_entity_descriptions(entity_ids)
    if not descriptions:
        return 0

    described_ids = [eid for eid in entity_ids if eid in descriptions]
    ph_e = ", ".join(["?" for _ in described_ids])
    entity_meta_rows = conn.execute(
        f"SELECT id, name, display_name, entity_type FROM entities WHERE id IN ({ph_e})",
        described_ids,
    ).fetchall()
    entity_meta: dict[str, tuple[str, str, str]] = {r[0]: (r[1], r[2], r[3]) for r in entity_meta_rows}

    # Hierarchy: build child -> (parent_id, parent_verb)
    parent_map: dict[str, tuple[str, str]] = {}
    if described_ids:
        ph_v = ", ".join(["?" for _ in _ALL_HIERARCHY_VERBS])
        ph_e2 = ", ".join(["?" for _ in described_ids])
        hierarchy_rows = conn.execute(
            f"SELECT subject_id, verb, object_id FROM svo_triples "
            f"WHERE verb IN ({ph_v}) "
            f"AND (subject_id IN ({ph_e2}) OR object_id IN ({ph_e2}))",
            _ALL_HIERARCHY_VERBS + described_ids + described_ids,
        ).fetchall()
        for subj, verb, obj in hierarchy_rows:
            if verb in _HIERARCHY_FORWARD and subj in entity_domain_map:
                parent_map.setdefault(subj, (obj, _PARENT_VERB_MAP[verb]))
            elif verb in _HIERARCHY_REVERSE and obj in entity_domain_map:
                parent_map.setdefault(obj, (subj, _PARENT_VERB_MAP[verb]))

    count = 0
    for entity_id, description in descriptions.items():
        if entity_id not in entity_meta:
            continue
        name, display_name, entity_type = entity_meta[entity_id]
        domain = entity_domain_map.get(entity_id, domain_names[0])
        parent_id, parent_verb = parent_map.get(entity_id, (None, "HAS_KIND"))  # type: ignore[assignment]

        existing = vector_store.get_glossary_term(name, session_id)
        if existing and existing.provenance == "human":
            continue

        term = GlossaryTerm(
            id=f"chonk_{entity_id}_{session_id}",
            name=name,
            display_name=display_name,
            definition=description,
            domain=domain,
            parent_id=parent_id,
            parent_verb=parent_verb,
            semantic_type=entity_type,
            status="draft",
            provenance="chonk_llm",
            session_id=session_id,
            user_id=user_id,
        )
        vector_store.add_glossary_term(term)
        count += 1

    logger.info(f"chonk glossary sync: {count} terms → session {session_id}")
    return count
