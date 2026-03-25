"""In-memory entity cache with JSON Patch support.

Mirrors the web UI's IndexedDB entity cache but uses a plain dict.
Stores compact entity state per session, applies RFC 6902 patches,
and inflates to glossary term dicts matching the HTTP API format.
"""

from __future__ import annotations

from typing import Any

import jsonpatch

# -- Compact key constants (must match constat/server/entity_state.py) ------
K_ENTITIES = "e"
K_GLOSSARY = "g"
K_RELATIONSHIPS = "r"
K_CLUSTERS = "k"

GK_NAME = "a"
GK_DISPLAY = "b"
GK_DEF = "c"
GK_STATUS = "d"
GK_PARENT = "e"
GK_ALIASES = "f"
GK_DOMAIN = "g"
GK_DOMAIN_PATH = "h"
GK_PARENT_VERB = "i"
GK_GLOSSARY_STATUS = "j"
GK_ENTITY_ID = "k"
GK_STYPE = "l"
GK_NER_TYPE = "m"
GK_TAGS = "n"
GK_IGNORED = "o"
GK_CANONICAL_SOURCE = "p"

RK_SUBJECT = "a"
RK_VERB = "b"
RK_OBJECT = "c"
RK_CONFIDENCE = "d"
RK_USER_EDITED = "e"

EMPTY_STATE: dict = {K_ENTITIES: {}, K_GLOSSARY: {}, K_RELATIONSHIPS: {}, K_CLUSTERS: {}}


class CachedEntry:
    __slots__ = ("state", "version")

    def __init__(self, state: dict, version: int) -> None:
        self.state = state
        self.version = version


class EntityCache:
    """Per-session in-memory cache of compact entity state."""

    def __init__(self) -> None:
        self._entries: dict[str, CachedEntry] = {}

    def get(self, session_id: str) -> CachedEntry | None:
        return self._entries.get(session_id)

    def set(self, session_id: str, state: dict, version: int) -> None:
        self._entries[session_id] = CachedEntry(state, version)

    def apply_patch(self, session_id: str, patch_ops: list[dict], version: int) -> dict:
        """Apply RFC 6902 JSON Patch ops, update cache, return new state."""
        entry = self._entries.get(session_id)
        base = entry.state if entry else dict(EMPTY_STATE)
        patch = jsonpatch.JsonPatch(patch_ops)
        new_state = patch.apply(base)
        self._entries[session_id] = CachedEntry(new_state, version)
        return new_state

    def clear(self, session_id: str) -> None:
        self._entries.pop(session_id, None)


def inflate_glossary(compact: dict) -> list[dict]:
    """Inflate compact state to list of glossary term dicts.

    Output format matches the HTTP GET /glossary endpoint.
    """
    glossary = compact.get(K_GLOSSARY, {})
    clusters = compact.get(K_CLUSTERS, {})
    relationships = compact.get(K_RELATIONSHIPS, {})

    # Build relationship list
    rels: list[dict] = []
    for rel_id, r in relationships.items():
        rels.append({
            "id": rel_id,
            "subject": r[RK_SUBJECT],
            "verb": r[RK_VERB],
            "object": r[RK_OBJECT],
            "confidence": r[RK_CONFIDENCE],
            "user_edited": r.get(RK_USER_EDITED, False),
        })

    # Index relationships by lowercase name
    rels_by_name: dict[str, list[dict]] = {}
    for r in rels:
        for key in (r["subject"].lower(), r["object"].lower()):
            rels_by_name.setdefault(key, []).append(r)

    # Index terms by entity_id for parent resolution
    term_by_entity_id: dict[str, dict] = {}
    for g in glossary.values():
        eid = g.get(GK_ENTITY_ID)
        if eid:
            term_by_entity_id[eid] = {"name": g[GK_NAME], "display_name": g[GK_DISPLAY]}

    # Index children by parent_id
    children_by_pid: dict[str, list[dict]] = {}
    for g in glossary.values():
        pid = g.get(GK_PARENT)
        if pid:
            children_by_pid.setdefault(pid, []).append({
                "name": g[GK_NAME],
                "display_name": g[GK_DISPLAY],
                "parent_verb": g.get(GK_PARENT_VERB),
            })

    terms: list[dict] = []
    for g in glossary.values():
        name = g[GK_NAME]
        entity_id = g.get(GK_ENTITY_ID)
        parent_id = g.get(GK_PARENT)
        glossary_status = g.get(GK_GLOSSARY_STATUS) or ("defined" if g.get(GK_DEF) else "self_describing")

        parent_info = term_by_entity_id.get(parent_id) if parent_id else None
        children = children_by_pid.get(entity_id, []) if entity_id else []
        term_rels = rels_by_name.get(name.lower(), [])

        terms.append({
            "name": name,
            "display_name": g[GK_DISPLAY],
            "definition": g.get(GK_DEF),
            "domain": g.get(GK_DOMAIN),
            "domain_path": g.get(GK_DOMAIN_PATH),
            "parent_id": parent_id,
            "parent_verb": g.get(GK_PARENT_VERB, "HAS_KIND"),
            "parent": parent_info,
            "aliases": g.get(GK_ALIASES, []),
            "semantic_type": g.get(GK_STYPE),
            "status": g.get(GK_STATUS),
            "glossary_status": glossary_status,
            "entity_id": entity_id,
            "ner_type": g.get(GK_NER_TYPE),
            "tags": g.get(GK_TAGS, {}),
            "ignored": g.get(GK_IGNORED, False),
            "cluster_siblings": clusters.get(name),
            "children": children,
            "relationships": term_rels,
            "canonical_source": g.get(GK_CANONICAL_SOURCE),
        })

    return terms


# Module-level singleton — shared across all Session instances in the kernel
_cache = EntityCache()
