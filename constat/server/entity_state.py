"""Compact entity state builder and JSON Patch differ for client-side caching.

Builds an ID-keyed JSON object with ultra-short keys (~30-40% smaller than verbose).
Uses RFC 6902 JSON Patch (via jsonpatch) for incremental diffs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jsonpatch

if TYPE_CHECKING:
    from constat.server.session_manager import ManagedSession

# -- Top-level section keys --------------------------------------------------
K_ENTITIES = "e"
K_GLOSSARY = "g"
K_RELATIONSHIPS = "r"
K_CLUSTERS = "k"

# -- Entity field keys -------------------------------------------------------
#  a=name, b=display_name, c=semantic_type, d=ner_type, e=domain_id
EK_NAME = "a"
EK_DISPLAY = "b"
EK_STYPE = "c"
EK_NER = "d"
EK_DOMAIN = "e"

# -- Glossary field keys -----------------------------------------------------
#  a=name, b=display_name, c=definition, d=status, e=parent_id, f=aliases,
#  g=domain, h=domain_path, i=parent_verb, j=glossary_status, k=entity_id,
#  l=semantic_type, m=ner_type, n=tags, o=ignored, p=canonical_source
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

# -- Relationship field keys -------------------------------------------------
#  a=subject, b=verb, c=object, d=confidence, e=user_edited
RK_SUBJECT = "a"
RK_VERB = "b"
RK_OBJECT = "c"
RK_CONFIDENCE = "d"
RK_USER_EDITED = "e"


# -- Domain resolution helpers (shared with glossary route) -------------------

def _build_domain_maps(config, session=None) -> tuple[dict[str, str], dict[str, str]]:
    """Build domain_path_map and source_to_domain from config + session resources.

    Uses SessionResources as the authoritative source for resource->domain mappings,
    since databases may be defined in domain YAMLs but promoted to the root config
    during loading.

    Returns:
        (domain_path_map, source_to_domain)
    """
    domain_path_map: dict[str, str] = {}
    source_to_domain: dict[str, str] = {}

    # Build domain path map from config
    for fname, dcfg in config.domains.items():
        if dcfg.path:
            domain_path_map[fname] = dcfg.path
            domain_path_map[fname.removesuffix(".yaml")] = dcfg.path
        # Also map from domain config's own databases/apis/documents
        for db_name in dcfg.databases:
            source_to_domain[db_name] = fname
        for api_name in dcfg.apis:
            source_to_domain[api_name] = fname
        for doc_name in dcfg.documents:
            source_to_domain[doc_name] = fname

    # Root config resources belong to __base__ (system domain)
    for db_name in config.databases:
        source_to_domain[db_name] = "__base__"
    if config.apis:
        for api_name in config.apis:
            source_to_domain[api_name] = "__base__"
    if config.documents:
        for doc_name in config.documents:
            source_to_domain[doc_name] = "__base__"

    # Authoritative mapping from SessionResources (tracks source for each resource)
    resources = getattr(session, "resources", None) if session else None
    if resources:
        for db_name, db_info in resources.databases.items():
            if db_info.source and db_info.source.startswith("domain:"):
                domain_fname = db_info.source.removeprefix("domain:")
                source_to_domain[db_name] = domain_fname
        for api_name, api_info in resources.apis.items():
            if api_info.source and api_info.source.startswith("domain:"):
                domain_fname = api_info.source.removeprefix("domain:")
                source_to_domain[api_name] = domain_fname
        for doc_name, doc_info in resources.documents.items():
            if doc_info.source and doc_info.source.startswith("domain:"):
                domain_fname = doc_info.source.removeprefix("domain:")
                source_to_domain[doc_name] = domain_fname
            elif doc_info.source and doc_info.source.startswith("user:"):
                user_id = doc_info.source.removeprefix("user:")
                source_to_domain[doc_name] = user_id

    return domain_path_map, source_to_domain


def _resolve_entity_domains(entity_id: str | None, vs, source_to_domain: dict[str, str]) -> list[str]:
    """Resolve all domains for an entity by tracing its chunk document sources.

    Returns a deduplicated list of domain names the entity's chunks map to.
    """
    if not entity_id:
        return []
    try:
        doc_names = vs.get_entity_document_names(entity_id, limit=20)
    except Exception:
        return []
    domains: list[str] = []
    seen: set[str] = set()
    for doc_name in doc_names:
        matched: str | None = None
        if doc_name.startswith("schema:"):
            db_name = doc_name.split(":")[1].split(".")[0]
            matched = source_to_domain.get(db_name)
        elif doc_name.startswith("api:"):
            api_name = doc_name.split(":")[1].split(".")[0]
            matched = source_to_domain.get(api_name)
        if not matched:
            matched = source_to_domain.get(doc_name)
        # Crawled sub-documents: "hr_management:crawled_8" -> try "hr_management"
        if not matched and ":" in doc_name:
            base_name = doc_name.split(":")[0]
            matched = source_to_domain.get(base_name)
        if matched and matched not in seen:
            seen.add(matched)
            domains.append(matched)
    return domains


def _resolve_entity_domain(entity_id: str | None, vs, source_to_domain: dict[str, str]) -> str | None:
    """Resolve effective domain for an entity. Returns 'cross-domain' when it spans multiple."""
    domains = _resolve_entity_domains(entity_id, vs, source_to_domain)
    if len(domains) > 1:
        return "cross-domain"
    if len(domains) == 1:
        return domains[0]
    return None


def build_compact_state(session_id: str, managed: ManagedSession) -> dict:
    """Build compact entity/glossary/relationship/cluster state for a session.

    Returns ID-keyed dict with short keys suitable for JSON Patch diffing.
    Glossary terms are keyed by lowercased name for stable JSON Patch diffing.
    """
    vs = managed.session.doc_tools._vector_store
    store = vs._relational
    conn = store._conn

    # -- Entities --
    rows = conn.execute(
        "SELECT id, name, display_name, semantic_type, ner_type, domain_id "
        "FROM entities WHERE session_id = ? ORDER BY name",
        [session_id],
    ).fetchall()
    entities = {}
    for r in rows:
        rec = {EK_NAME: r[1], EK_DISPLAY: r[2], EK_STYPE: r[3]}
        if r[4]:
            rec[EK_NER] = r[4]
        if r[5]:
            rec[EK_DOMAIN] = r[5]
        entities[r[0]] = rec

    # -- Glossary (unified: NER entities + glossary_terms) --
    glossary_rows = vs.get_unified_glossary(
        session_id,
        scope="all",
        active_domains=managed.active_domains,
        user_id=managed.user_id,
    )

    config = managed.session.config
    domain_path_map, source_to_domain = _build_domain_maps(config, managed.session)

    glossary = {}
    for row in sorted(glossary_rows, key=lambda r: r["name"]):
        # Resolve effective domain
        effective_domain = row.get("domain")
        if not effective_domain:
            effective_domain = _resolve_entity_domain(row.get("entity_id"), vs, source_to_domain)
        if effective_domain == "__base__":
            effective_domain = "system"

        domain_path = domain_path_map.get(effective_domain) if effective_domain else None

        rec: dict = {GK_NAME: row["name"], GK_DISPLAY: row["display_name"]}
        if row.get("definition"):
            rec[GK_DEF] = row["definition"]
        if row.get("status"):
            rec[GK_STATUS] = row["status"]
        if row.get("parent_id"):
            rec[GK_PARENT] = row["parent_id"]
        if row.get("aliases"):
            rec[GK_ALIASES] = row["aliases"]
        if effective_domain:
            rec[GK_DOMAIN] = effective_domain
        if domain_path:
            rec[GK_DOMAIN_PATH] = domain_path
        parent_verb = row.get("parent_verb") or "HAS_KIND"
        if parent_verb != "HAS_KIND":
            rec[GK_PARENT_VERB] = parent_verb
        glossary_status = row.get("glossary_status") or "self_describing"
        if glossary_status != "self_describing":
            rec[GK_GLOSSARY_STATUS] = glossary_status
        if row.get("entity_id"):
            rec[GK_ENTITY_ID] = row["entity_id"]
        if row.get("semantic_type"):
            rec[GK_STYPE] = row["semantic_type"]
        if row.get("ner_type"):
            rec[GK_NER_TYPE] = row["ner_type"]
        tags = {k: v for k, v in (row.get("tags") or {}).items() if not k.startswith("_")}
        if tags:
            rec[GK_TAGS] = tags
        if row.get("ignored"):
            rec[GK_IGNORED] = True
        if row.get("canonical_source"):
            rec[GK_CANONICAL_SOURCE] = row["canonical_source"]

        # Key by lowercased name for stable diffing
        glossary[row["name"]] = rec

    # -- Relationships --
    rels = store.list_session_relationships(session_id)
    relationships = {}
    for r in rels:
        rec_r: dict = {
            RK_SUBJECT: r["subject_name"],
            RK_VERB: r["verb"],
            RK_OBJECT: r["object_name"],
            RK_CONFIDENCE: r["confidence"],
        }
        if r.get("user_edited"):
            rec_r[RK_USER_EDITED] = True
        relationships[r["id"]] = rec_r

    # -- Clusters (term_name -> [siblings]) --
    cluster_rows = conn.execute(
        "SELECT term_name, cluster_id FROM glossary_clusters WHERE session_id = ? ORDER BY cluster_id, term_name",
        [session_id],
    ).fetchall()
    cluster_map: dict[int, list[str]] = {}
    for term_name, cluster_id in cluster_rows:
        cluster_map.setdefault(cluster_id, []).append(term_name.lower())
    clusters: dict[str, list[str]] = {}
    for members in cluster_map.values():
        if len(members) < 2:
            continue
        for term in members:
            clusters[term] = [m for m in members if m != term]

    return {
        K_ENTITIES: entities,
        K_GLOSSARY: glossary,
        K_RELATIONSHIPS: relationships,
        K_CLUSTERS: clusters,
    }


def compute_entity_patch(old_state: dict, new_state: dict) -> list[dict]:
    """Compute RFC 6902 JSON Patch ops from old_state to new_state."""
    patch = jsonpatch.make_patch(old_state, new_state)
    return patch.patch
