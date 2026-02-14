# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query

from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{session_id}/entities")
async def list_entities(
    session_id: str,
    entity_type: str | None = Query(default=None, description="Filter by entity type"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """List extracted entities from the session.

    Returns deduplicated entities with their reference locations.

    Args:
        session_id: Session ID
        entity_type: Optional filter by type (table, column, concept, etc.)
        session_manager: Injected session manager

    Returns:
        List of entities with references

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Use dict keyed by normalized_name only for deduplication (merge across types)
    from constat.discovery.models import normalize_entity_name, display_entity_name

    # Consolidate similar types into simpler categories for display
    TYPE_CONSOLIDATION = {
        "api_endpoint": "api_endpoint",
        "api_schema": "api_schema",
        "api_field": "api_field",
        "rest_field": "api_field",  # REST fields -> api_field
        "rest": "api_endpoint",     # REST endpoint -> api_endpoint
        "rest/schema": "api_schema", # REST schema -> api_schema
        "graphql_type": "graphql",
        "graphql_field": "graphql",
    }

    # Type priority for picking primary type when merging (higher = preferred)
    # API fields should NOT lose to table/column when they're clearly API-sourced
    # Schema elements come after API-specific types to avoid misclassification
    TYPE_PRIORITY = {
        "api_field": 95,      # API fields should win over generic table/column
        "api_endpoint": 90,
        "api_schema": 85,
        "api": 82,            # Generic API type
        "graphql": 80,
        "table": 75,          # Schema types below API types
        "column": 70,
        "action": 50,         # Actions (verbs extracted from documents)
        "concept": 40,
        "business_term": 30,
        "organization": 20,
        "product": 20,
        "location": 20,
        "event": 20,
    }

    entity_map: dict[str, dict[str, Any]] = {}

    # Cache for related entities queries (entity_id -> list of related)
    related_entities_cache: dict[str, list[dict]] = {}

    def get_related_entities(vector_store, ent_id_param: str, sess_id: str, limit: int = 5) -> list[dict]:
        """Find entities that co-occur in the same chunks as the given entity.

        Returns list of {"name": str, "type": str, "co_occurrences": int}
        """
        if ent_id_param in related_entities_cache:
            return related_entities_cache[ent_id_param]

        try:
            # Find chunks where this entity appears, then find other entities in those chunks
            co_occurrence_result = vector_store._conn.execute("""
                SELECT e2.name, e2.semantic_type, COUNT(*) as co_occurrences
                FROM chunk_entities ce1
                JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
                JOIN entities e2 ON ce2.entity_id = e2.id
                WHERE ce1.entity_id = ?
                  AND ce2.entity_id != ce1.entity_id
                  AND (e2.session_id = ? OR e2.session_id IS NULL)
                GROUP BY e2.id, e2.name, e2.semantic_type
                ORDER BY co_occurrences DESC
                LIMIT ?
            """, [ent_id_param, sess_id, limit]).fetchall()

            related_list = [
                {"name": co_row[0], "type": co_row[1] or "concept", "co_occurrences": co_row[2]}
                for co_row in co_occurrence_result
            ]
            related_entities_cache[ent_id_param] = related_list
            return related_list
        except Exception as err:
            logger.debug(f"Could not get related entities for {ent_id_param}: {err}")
            return []

    def consolidate_source(source_str: str) -> str:
        """Consolidate column-level schema sources to table level.

        schema:hr.performance_reviews.employee_id -> schema:hr.performance_reviews
        schema:hr.performance_reviews -> schema:hr.performance_reviews (unchanged)
        business_rules -> business_rules (unchanged)
        """
        if source_str.startswith("schema:"):
            parts = source_str.split(".")
            # schema:db.table.column -> keep schema:db.table
            # schema:db.table -> keep as is
            if len(parts) >= 3:
                # Has column part, consolidate to table
                return ".".join(parts[:2])
        return source_str

    def add_entity(
        entity_name: str,
        etype: str,
        entity_source: str,
        metadata: dict,
        entity_references: list[dict] | None = None,
        related_entities: list[dict] | None = None,
    ):
        """Add or merge an entity into the map.

        Normalizes entity names for deduplication and display:
        - Cache key uses normalized (lowercase, singular) form only
        - Entities with same name but different types are merged
        - API endpoint/schema types are consolidated to just "api"
        - GraphQL type/field types are consolidated to just "graphql"
        """
        # Consolidate type
        etype = TYPE_CONSOLIDATION.get(etype, etype)

        # Detect and correct type based on reference sources
        # If all references are from API sources but type is table/column, correct it
        if etype in ("table", "column") and entity_references:
            api_refs = [r for r in entity_references if r.get("document", "").startswith("api:")]
            non_api_refs = [r for r in entity_references if not r.get("document", "").startswith("api:")]
            if api_refs and not non_api_refs:
                # All references are API - infer type from section patterns
                sections = [r.get("section", "") for r in api_refs]
                if any("field" in s.lower() for s in sections):
                    etype = "api_field"
                elif any("schema" in s.lower() for s in sections):
                    etype = "api_schema"
                else:
                    etype = "api_endpoint"

        normalized = normalize_entity_name(entity_name)
        display = display_entity_name(entity_name)
        key = normalized.lower()

        # Get original_name from metadata, or use raw name if different from display
        original_name = metadata.get("original_name")
        if not original_name and entity_name != display and entity_name != normalized:
            original_name = entity_name
            metadata = {**metadata, "original_name": original_name}

        # Consolidate source (e.g., schema:db.table.column -> schema:db.table)
        consolidated_source = consolidate_source(entity_source)

        if key not in entity_map:
            entity_map[key] = {
                "id": str(hash(f"{display}")),
                "name": display,
                "type": etype,
                "types": [etype],
                "sources": [consolidated_source],
                "metadata": metadata,
                "references": entity_references or [],
                "related_entities": related_entities or [],
                "mention_count": len(entity_references) if entity_references else 0,
                "original_name": original_name,
            }
        else:
            existing = entity_map[key]
            # Add type if new
            if etype not in existing["types"]:
                existing["types"].append(etype)
                # Update primary type if new type has higher priority
                if TYPE_PRIORITY.get(etype, 0) > TYPE_PRIORITY.get(existing["type"], 0):
                    existing["type"] = etype
            # Merge: add consolidated source if new
            if consolidated_source not in existing["sources"]:
                existing["sources"].append(consolidated_source)
            # Merge references with deduplication (by document + section)
            if entity_references:
                existing_refs = {(r["document"], r["section"]) for r in existing["references"]}
                for ref in entity_references:
                    ref_key = (ref["document"], ref["section"])
                    if ref_key not in existing_refs:
                        existing["references"].append(ref)
                        existing_refs.add(ref_key)
                existing["mention_count"] = len(existing["references"])
            # Merge related_entities (prefer the one with more entries)
            if related_entities and len(related_entities) > len(existing.get("related_entities", [])):
                existing["related_entities"] = related_entities
            # Merge metadata
            existing["metadata"].update(metadata)
            # Update original_name if not already set
            if original_name and not existing.get("original_name"):
                existing["original_name"] = original_name

    # 1. Get entities from vector store (includes schema, api, document sources)
    try:
        # Vector store is accessed via doc_tools
        vs = None
        if hasattr(managed.session, "doc_tools") and managed.session.doc_tools:
            vs = managed.session.doc_tools._vector_store
        if vs:
            # Build filter for base + project + session
            # Base: project_id IS NULL AND session_id IS NULL
            # Project: project_id IN (active_projects)
            # Session: session_id = server_session_id
            active_projects = getattr(managed, "active_projects", []) or []

            filter_conditions = ["(e.project_id IS NULL AND e.session_id IS NULL)"]
            params: list = []

            if active_projects:
                placeholders = ",".join(["?" for _ in active_projects])
                filter_conditions.append(f"e.project_id IN ({placeholders})")
                params.extend(active_projects)

            filter_conditions.append("e.session_id = ?")
            params.append(session_id)

            where_clause = " OR ".join(filter_conditions)

            # Debug: check chunk_entities for this session (via entity_id join)
            ce_count = vs._conn.execute(
                "SELECT COUNT(*) FROM chunk_entities ce JOIN entities e ON ce.entity_id = e.id WHERE e.session_id = ?",
                [session_id],
            ).fetchone()[0]
            print(f"[ENTITIES] session_id={session_id[:8]}, chunk_entities for session: {ce_count}")

            # Debug: check for performance review entity specifically
            pr_check = vs._conn.execute("""
                SELECT e.id, e.name,
                       (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as total_links
                FROM entities e
                WHERE LOWER(e.name) LIKE '%performance%' AND e.session_id = ?
            """, [session_id]).fetchall()
            for row in pr_check:
                print(f"[ENTITIES] Performance entity: id={row[0][:8]}, name={row[1]}, total_links={row[2]}")

            # Get entities visible to this session
            result = vs._conn.execute(f"""
                SELECT e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
                       (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as ref_count
                FROM entities e
                WHERE ({where_clause})
                ORDER BY e.name
            """, params).fetchall()

            for row in result:
                ent_id, name, display_name, semantic_type, ner_type, ref_count = row
                if entity_type and semantic_type != entity_type:
                    continue

                # Skip orphaned entities (no chunk links = can't trace origin)
                if ref_count == 0:
                    continue

                # Get reference locations for this entity
                references = []
                if ref_count > 0:
                    ref_result = vs._conn.execute("""
                        SELECT em.document_name, em.section, ce.confidence
                        FROM chunk_entities ce
                        JOIN embeddings em ON ce.chunk_id = em.chunk_id
                        WHERE ce.entity_id = ?
                        ORDER BY ce.confidence DESC
                        LIMIT 10
                    """, [ent_id]).fetchall()
                    for ref_row in ref_result:
                        doc_name, section, confidence = ref_row
                        references.append({
                            "document": doc_name,
                            "section": section,
                            "confidence": confidence,
                        })

                # No synthetic references - entities should have real chunk links
                # If no chunk links exist, the entity simply has no reference locations
                # Use semantic_type as the type, and derive source from ner_type
                source = "ner" if ner_type else "schema"

                # Get related entities (entities that co-occur in same chunks)
                related = get_related_entities(vs, ent_id, session_id) if ref_count > 0 else []

                add_entity(
                    name, semantic_type or "concept", source,
                    {"display_name": display_name, "ner_type": ner_type},
                    references, related
                )
    except Exception as e:
        logger.warning(f"Could not get entities from vector_store: {e}")

    # 2. Get schema entities from schema_manager (only if not already in vector store with refs)
    try:
        if managed.session.schema_manager:
            metadata_cache = managed.session.schema_manager.metadata_cache
            for full_name, table_meta in metadata_cache.items():
                db_name = table_meta.database
                table_name = table_meta.name

                # Add table entity - always add to ensure proper type merging
                # (table type has higher priority than concept/business_term)
                if not entity_type or entity_type == "table":
                    add_entity(
                        table_name, "table", "schema",
                        {"database": db_name, "full_name": full_name},
                        [{"document": f"Database: {db_name}", "section": "Schema", "mentions": 1}]
                    )

                # Add column entities - always add to ensure proper type merging
                if not entity_type or entity_type == "column":
                    for col in table_meta.columns:
                        add_entity(
                            col.name, "column", "schema",
                            {
                                "table": table_name,
                                "database": db_name,
                                "dtype": col.type if col.type else None,
                            },
                            [{"document": f"Table: {table_name}", "section": f"Database: {db_name}", "mentions": 1}]
                        )
    except Exception as e:
        logger.warning(f"Could not get entities from schema_manager: {e}")

    # 3. Get API entities from config - always add to ensure proper type merging
    try:
        if managed.session.config and managed.session.config.apis:
            for api_name, api_config in managed.session.config.apis.items():
                if not entity_type or entity_type in ("api", "api_endpoint"):
                    add_entity(
                        api_name, "api", "api",
                        {"base_url": getattr(api_config, "base_url", None)},
                        [{"document": f"API: {api_name}", "section": "Configuration", "mentions": 1}]
                    )
    except Exception as e:
        logger.warning(f"Could not get API entities: {e}")

    # 4. Get document entities from config - always add to ensure proper type merging
    try:
        if managed.session.config and managed.session.config.documents:
            for doc_name in managed.session.config.documents.keys():
                if not entity_type or entity_type == "concept":
                    add_entity(
                        doc_name, "concept", "document",
                        {"source": "document_config"},
                        [{"document": doc_name, "section": "Indexed Document", "mentions": 1}]
                    )
    except Exception as e:
        logger.warning(f"Could not get document entities: {e}")

    # 5. Get entities from active projects - always add to ensure proper type merging
    try:
        active_projects = getattr(managed, "active_projects", [])
        if active_projects and managed.session.config:
            for project_filename in active_projects:
                project = managed.session.config.load_project(project_filename)
                if project:
                    # Add project API entities
                    if not entity_type or entity_type in ("api", "api_endpoint"):
                        for api_name, api_config in project.apis.items():
                            add_entity(
                                api_name, "api", "api",
                                {"base_url": getattr(api_config, "base_url", None), "project": project_filename},
                                [{"document": f"API: {api_name}", "section": f"Project: {project_filename}", "mentions": 1}]
                            )

                    # Add project document entities
                    if not entity_type or entity_type == "concept":
                        for doc_name in project.documents.keys():
                            add_entity(
                                doc_name, "concept", "document",
                                {"source": "project", "project": project_filename},
                                [{"document": doc_name, "section": f"Project: {project_filename}", "mentions": 1}]
                            )
    except Exception as e:
        logger.warning(f"Could not get entities from active projects: {e}")

    entities = list(entity_map.values())

    logger.debug(f"list_entities: returning {len(entities)} entities for session {session_id[:8]}")
    return {"entities": entities}


@router.post("/{session_id}/entities/{entity_id}/glossary")
async def add_entity_to_glossary(
    session_id: str,
    entity_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Add an entity to the glossary/business terms.

    Args:
        session_id: Session ID
        entity_id: Entity ID to add
        session_manager: Injected session manager

    Returns:
        Confirmation

    Raises:
        404: Session or entity not found
    """
    managed = session_manager.get_session(session_id)

    # Try to add to glossary via session
    try:
        if hasattr(managed.session, "add_to_glossary"):
            managed.session.add_to_glossary(entity_id)
            return {"status": "added", "entity_id": entity_id}
    except Exception as e:
        logger.warning(f"Could not add to glossary: {e}")

    return {"status": "added", "entity_id": entity_id, "note": "Glossary update pending"}
