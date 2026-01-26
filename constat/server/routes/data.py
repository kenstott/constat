# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data access REST endpoints (tables, artifacts, facts)."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from constat.server.models import (
    ArtifactContentResponse,
    ArtifactInfo,
    ArtifactListResponse,
    FactInfo,
    FactListResponse,
    TableDataResponse,
    TableInfo,
    TableListResponse,
)
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


@router.get("/{session_id}/tables", response_model=TableListResponse)
async def list_tables(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableListResponse:
    """List all tables in the session's datastore.

    Args:
        session_id: Session ID

    Returns:
        List of tables with metadata

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        return TableListResponse(tables=[])

    try:
        tables = managed.session.datastore.list_tables()
        return TableListResponse(
            tables=[
                TableInfo(
                    name=t["name"],
                    row_count=t.get("row_count", 0),
                    step_number=t.get("step_number", 0),
                    columns=t.get("columns", []),
                )
                for t in tables
            ]
        )
    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/tables/{table_name}", response_model=TableDataResponse)
async def get_table_data(
    session_id: str,
    table_name: str,
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=100, ge=1, le=1000, description="Rows per page"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TableDataResponse:
    """Get table data with pagination.

    Args:
        session_id: Session ID
        table_name: Table name to retrieve
        page: Page number (1-indexed)
        page_size: Number of rows per page

    Returns:
        Table data with pagination info

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        # Load the full DataFrame
        df = managed.session.datastore.load_dataframe(table_name)

        total_rows = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        # Slice for pagination
        page_df = df.iloc[start_idx:end_idx]

        # Convert to records
        data = page_df.to_dict(orient="records")

        return TableDataResponse(
            name=table_name,
            columns=list(df.columns),
            data=data,
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_rows,
        )

    except Exception as e:
        logger.error(f"Error getting table data: {e}")
        raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")


@router.get("/{session_id}/artifacts", response_model=ArtifactListResponse)
async def list_artifacts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactListResponse:
    """List all artifacts in the session.

    Args:
        session_id: Session ID

    Returns:
        List of artifacts with metadata

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        return ArtifactListResponse(artifacts=[])

    try:
        artifacts = managed.session.datastore.list_artifacts()
        tables = managed.session.datastore.list_tables()

        # Determine which artifacts are key results
        # Key results: visualizations - NOT code
        visualization_types = {'chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega', 'markdown', 'md'}
        # Code types are explicitly excluded from key results
        code_types = {'code', 'python', 'sql', 'script', 'text', 'output', 'error'}

        def is_key_result(a: dict) -> bool:
            artifact_type = a.get("type", "").lower()
            # Code is NEVER a key result
            if artifact_type in code_types:
                return False
            # Visualizations are always key results
            if artifact_type in visualization_types:
                return True
            return False

        # Build artifact list
        artifact_list = [
            ArtifactInfo(
                id=a["id"],
                name=a["name"],
                artifact_type=a["type"],
                step_number=a.get("step_number", 0),
                title=a.get("title"),
                description=a.get("description"),
                mime_type=a.get("content_type") or "application/octet-stream",
                created_at=a.get("created_at"),
                is_key_result=is_key_result(a),
            )
            for a in artifacts
        ]

        # Add consequential tables as virtual artifacts
        # A table is consequential if it's from the final step or has significant data
        if tables:
            max_step = max((t.get("step_number", 0) for t in tables), default=0)
            for t in tables:
                table_name = t["name"]
                # Skip internal tables
                if table_name.startswith("_"):
                    continue
                # Final step tables are consequential
                is_final_step = t.get("step_number", 0) == max_step and max_step > 0
                # Tables with substantial data are consequential
                has_data = t.get("row_count", 0) > 0
                if is_final_step and has_data:
                    # Create a virtual artifact entry for this table
                    # Use negative IDs to distinguish from real artifacts
                    virtual_id = -hash(table_name) % 1000000
                    artifact_list.append(
                        ArtifactInfo(
                            id=virtual_id,
                            name=table_name,
                            artifact_type="table",
                            step_number=t.get("step_number", 0),
                            title=f"Table: {table_name}",
                            description=f"{t.get('row_count', 0)} rows",
                            mime_type="application/x-dataframe",
                            created_at=None,
                            is_key_result=True,
                        )
                    )

        logger.debug(f"[list_artifacts] Returning {len(artifact_list)} artifacts (including {len([a for a in artifact_list if a.artifact_type == 'table'])} tables)")
        return ArtifactListResponse(artifacts=artifact_list)
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/artifacts/{artifact_id}", response_model=ArtifactContentResponse)
async def get_artifact(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ArtifactContentResponse:
    """Get artifact content by ID.

    Args:
        session_id: Session ID
        artifact_id: Artifact ID

    Returns:
        Artifact with content

    Raises:
        404: Session or artifact not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)

        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        return ArtifactContentResponse(
            id=artifact.id,
            name=artifact.name,
            artifact_type=artifact.artifact_type.value,
            content=artifact.content,
            mime_type=artifact.mime_type,
            is_binary=artifact.is_binary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/facts", response_model=FactListResponse)
async def list_facts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FactListResponse:
    """List all resolved facts in the session.

    Facts are values extracted from user queries or resolved during execution.

    Args:
        session_id: Session ID

    Returns:
        List of resolved facts

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()

        # Get persisted fact names from FactStore
        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        persisted_fact_names = set(fact_store.list_facts().keys())

        return FactListResponse(
            facts=[
                FactInfo(
                    name=name,
                    value=fact.value,
                    source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                    reasoning=fact.reasoning,
                    confidence=getattr(fact, "confidence", None),
                    is_persisted=name in persisted_fact_names,
                )
                for name, fact in all_facts.items()
            ]
        )
    except Exception as e:
        logger.error(f"Error listing facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/proof-tree")
async def get_proof_tree(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get the proof tree for auditable mode execution.

    The proof tree shows how facts were resolved and combined
    to produce the final answer.

    Args:
        session_id: Session ID

    Returns:
        Proof tree structure

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        # Get all facts with their provenance
        all_facts = managed.session.fact_resolver.get_all_facts()

        # Build proof tree structure
        nodes = []
        for name, fact in all_facts.items():
            node = {
                "name": name,
                "value": fact.value,
                "source": fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                "reasoning": fact.reasoning,
                "dependencies": getattr(fact, "dependencies", []),
            }
            nodes.append(node)

        return {
            "facts": nodes,
            "execution_trace": [],  # Could be populated from datastore logs
        }

    except Exception as e:
        logger.error(f"Error getting proof tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/output")
async def get_output(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get the final output/answer from the session.

    Args:
        session_id: Session ID

    Returns:
        Final output with any suggestions

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Get output from scratchpad or last execution
    output = ""
    suggestions = []

    if managed.session.scratchpad:
        # Try to get synthesized output from scratchpad
        recent = managed.session.scratchpad.get_recent_context(max_steps=1)
        if recent:
            output = recent

    return {
        "output": output,
        "suggestions": suggestions,
        "current_query": managed.current_query,
    }


# ============================================================================
# Entity Endpoints
# ============================================================================


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

    Returns:
        List of entities with references

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Use dict keyed by normalized_name only for deduplication (merge across types)
    from constat.discovery.models import normalize_entity_name, display_entity_name

    # Consolidate similar types into simpler categories
    TYPE_CONSOLIDATION = {
        "api_endpoint": "api",
        "api_schema": "api",
        "graphql_type": "graphql",
        "graphql_field": "graphql",
    }

    # Type priority for picking primary type when merging (higher = preferred)
    TYPE_PRIORITY = {
        "table": 100,
        "api": 90,
        "graphql": 80,
        "column": 60,
        "concept": 40,
        "business_term": 30,
        "organization": 20,
        "product": 20,
        "location": 20,
        "event": 20,
    }

    entity_map: dict[str, dict[str, Any]] = {}

    def add_entity(name: str, etype: str, source: str, metadata: dict, references: list[dict] | None = None):
        """Add or merge an entity into the map.

        Normalizes entity names for deduplication and display:
        - Cache key uses normalized (lowercase, singular) form only
        - Entities with same name but different types are merged
        - API endpoint/schema types are consolidated to just "api"
        - GraphQL type/field types are consolidated to just "graphql"
        """
        # Consolidate type
        etype = TYPE_CONSOLIDATION.get(etype, etype)
        normalized = normalize_entity_name(name)
        display = display_entity_name(name)
        key = normalized.lower()

        # Get original_name from metadata, or use raw name if different from display
        original_name = metadata.get("original_name")
        if not original_name and name != display and name != normalized:
            original_name = name
            metadata = {**metadata, "original_name": original_name}

        if key not in entity_map:
            entity_map[key] = {
                "id": str(hash(f"{display}")),
                "name": display,
                "type": etype,
                "types": [etype],
                "sources": [source],
                "metadata": metadata,
                "references": references or [],
                "mention_count": len(references) if references else 0,
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
            # Merge: add source if new
            if source not in existing["sources"]:
                existing["sources"].append(source)
            # Merge references with deduplication (by document + section)
            if references:
                existing_refs = {(r["document"], r["section"]) for r in existing["references"]}
                for ref in references:
                    ref_key = (ref["document"], ref["section"])
                    if ref_key not in existing_refs:
                        existing["references"].append(ref)
                        existing_refs.add(ref_key)
                existing["mention_count"] = len(existing["references"])
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
            # Get all entities from the store
            result = vs._conn.execute("""
                SELECT e.id, e.name, e.type, e.source, e.metadata,
                       (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as ref_count
                FROM entities e
                ORDER BY e.name
            """).fetchall()

            for row in result:
                ent_id, name, etype, source, metadata_json, ref_count = row
                if entity_type and etype != entity_type:
                    continue

                metadata = {}
                if metadata_json:
                    import json
                    metadata = json.loads(metadata_json)

                # Get reference locations for this entity (including mention_text)
                references = []
                if ref_count > 0:
                    ref_result = vs._conn.execute("""
                        SELECT em.document_name, em.section, ce.mention_count, ce.mention_text
                        FROM chunk_entities ce
                        JOIN embeddings em ON ce.chunk_id = em.chunk_id
                        WHERE ce.entity_id = ?
                        ORDER BY ce.mention_count DESC
                        LIMIT 10
                    """, [ent_id]).fetchall()
                    for ref_row in ref_result:
                        doc_name, section, mentions, mention_text = ref_row
                        references.append({
                            "document": doc_name,
                            "section": section,
                            "mentions": mentions,
                            "mention_text": mention_text,
                        })

                # For API entities without chunk refs, create synthetic reference from metadata
                if not references and source == "api":
                    api_name = metadata.get("api_name", "API")
                    http_method = metadata.get("http_method", "")
                    http_path = metadata.get("http_path", "")
                    if http_method and http_path:
                        references.append({
                            "document": f"API: {api_name}",
                            "section": f"{http_method} {http_path}",
                            "mentions": 1,
                            "mention_text": metadata.get("original_name", name),
                        })
                    else:
                        references.append({
                            "document": f"API: {api_name}",
                            "section": "Schema Definition",
                            "mentions": 1,
                            "mention_text": name,
                        })

                add_entity(name, etype, source or "unknown", metadata, references)
    except Exception as e:
        logger.warning(f"Could not get entities from vector_store: {e}")

    # 2. Get schema entities from schema_manager (in case vector store doesn't have them)
    try:
        if managed.session.schema_manager:
            metadata_cache = managed.session.schema_manager.metadata_cache
            for full_name, table_meta in metadata_cache.items():
                db_name = table_meta.database
                table_name = table_meta.name

                # Add table entity
                if not entity_type or entity_type == "table":
                    add_entity(
                        table_name, "table", "schema",
                        {"database": db_name, "full_name": full_name},
                        [{"document": f"Database: {db_name}", "section": "Schema", "mentions": 1}]
                    )

                # Add column entities
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

    # 3. Get API entities from config
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

    # 4. Get document entities from config
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

    entities = list(entity_map.values())
    logger.debug(f"[list_entities] Returning {len(entities)} deduplicated entities")
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


# ============================================================================
# Fact Action Endpoints
# ============================================================================


@router.post("/{session_id}/facts")
async def add_fact(
    session_id: str,
    body: dict[str, Any],
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Add a new fact to the session.

    Args:
        session_id: Session ID
        body: Request body with name, value, and optional persist flag

    Returns:
        Created fact

    Raises:
        400: Missing name or value
    """
    managed = session_manager.get_session(session_id)

    if "name" not in body:
        raise HTTPException(status_code=400, detail="Missing 'name' in request body")
    if "value" not in body:
        raise HTTPException(status_code=400, detail="Missing 'value' in request body")

    try:
        fact_name = body["name"]
        fact_value = body["value"]
        persist = body.get("persist", False)

        # Add the fact via fact_resolver
        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=fact_value,
            source=FactSource.USER_PROVIDED,
            reasoning="Added via UI",
        )

        # Optionally persist to FactStore
        is_persisted = False
        if persist:
            from constat.storage.facts import FactStore
            fact_store = FactStore(user_id=managed.user_id)
            fact_store.save_fact(
                name=fact_name,
                value=fact_value,
                description="Added via UI",
            )
            is_persisted = True

        return {
            "status": "created",
            "fact": {
                "name": fact_name,
                "value": fact_value,
                "source": FactSource.USER_PROVIDED.value,
                "is_persisted": is_persisted,
            },
        }

    except Exception as e:
        logger.error(f"Error adding fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/facts/{fact_name}/persist")
async def persist_fact(
    session_id: str,
    fact_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Cache a fact for future use.

    Args:
        session_id: Session ID
        fact_name: Name of the fact to persist

    Returns:
        Confirmation

    Raises:
        404: Session or fact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        # Persist the fact
        if hasattr(managed.session.fact_resolver, "persist_fact"):
            managed.session.fact_resolver.persist_fact(fact_name)

        return {"status": "persisted", "fact_name": fact_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error persisting fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/facts/{fact_name}/forget")
async def forget_fact(
    session_id: str,
    fact_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Forget a fact (removes from both session and persistent storage).

    Args:
        session_id: Session ID
        fact_name: Name of the fact to forget

    Returns:
        Confirmation with what was deleted

    Raises:
        404: Session or fact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        deleted_persistent = False
        deleted_session = False

        # Delete from persistent storage (facts.yaml) if exists
        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        if fact_store.delete_fact(fact_name):
            deleted_persistent = True

        # Remove from session cache
        if hasattr(managed.session.fact_resolver, "_cache"):
            if fact_name in managed.session.fact_resolver._cache:
                managed.session.fact_resolver._cache.pop(fact_name, None)
                deleted_session = True

        return {
            "status": "forgotten",
            "fact_name": fact_name,
            "deleted_persistent": deleted_persistent,
            "deleted_session": deleted_session,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forgetting fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{session_id}/facts/{fact_name}")
async def edit_fact(
    session_id: str,
    fact_name: str,
    body: dict[str, Any],
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Edit a fact value.

    Args:
        session_id: Session ID
        fact_name: Name of the fact to edit
        body: Request body with new value

    Returns:
        Updated fact

    Raises:
        404: Session or fact not found
        400: Missing value in request
    """
    managed = session_manager.get_session(session_id)

    if "value" not in body:
        raise HTTPException(status_code=400, detail="Missing 'value' in request body")

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        # Edit the fact by updating the cache directly
        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=body["value"],
            source=FactSource.USER_PROVIDED,
            reasoning="Edited via UI",
        )

        return {
            "status": "updated",
            "fact_name": fact_name,
            "new_value": body["value"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Step Code Endpoints
# ============================================================================


@router.get("/{session_id}/steps")
async def list_step_codes(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """List all step codes for a session.

    Returns the code executed for each step in the plan, stored on disk.

    Args:
        session_id: Session ID

    Returns:
        List of step codes with step_number, goal, and code

    Raises:
        404: Session not found
    """
    # Try to get the session from memory first
    managed = session_manager.get_session_or_none(session_id)
    history = None
    history_session_id = None

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
        logger.debug(f"[list_step_codes] Found managed session. Server: {session_id}, History: {history_session_id}")
    else:
        # Session not in memory - try reverse lookup from disk
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id="default")  # TODO: get user_id properly
        history_session_id = history.find_session_by_server_id(session_id)
        logger.debug(f"[list_step_codes] Session not in memory. Reverse lookup found: {history_session_id}")

    try:
        steps = history.list_step_codes(history_session_id) if history_session_id else []
        logger.debug(f"[list_step_codes] Found {len(steps)} steps")

        return {
            "steps": steps,
            "total": len(steps),
            # Include session ID info for debugging
            "session_info": {
                "server_session_id": session_id,
                "history_session_id": history_session_id,
            },
        }

    except Exception as e:
        logger.error(f"Error listing step codes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/download-code")
async def download_code(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Download all step codes as a standalone Python script.

    Generates a self-contained Python script that can be run independently
    to reproduce the analysis. Includes all step functions, imports, and
    helper utilities. Facts are loaded from _facts.parquet and passed as
    explicit arguments to run_analysis().

    Args:
        session_id: Session ID

    Returns:
        Python script file download

    Raises:
        404: Session not found or no code available
    """
    from fastapi.responses import Response
    import pandas as pd

    # Try to get the session from memory first
    managed = session_manager.get_session_or_none(session_id)
    history = None
    history_session_id = None

    if managed:
        history = managed.session.history
        history_session_id = managed.session.session_id
        logger.debug(f"[download-code] Found managed session. Server: {session_id}, History: {history_session_id}")
    else:
        # Session not in memory - try reverse lookup from disk
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id="default")  # TODO: get user_id properly
        history_session_id = history.find_session_by_server_id(session_id)
        logger.debug(f"[download-code] Session not in memory. Reverse lookup found: {history_session_id}")

    try:
        if not history_session_id:
            logger.warning(f"[download-code] No history session ID for server session {session_id}")
            raise HTTPException(
                status_code=404,
                detail="No code available for this session. Run a query first to generate step code."
            )

        # Check if the steps directory exists
        steps_dir = history._steps_dir(history_session_id)
        logger.debug(f"[download-code] Steps directory: {steps_dir}, exists: {steps_dir.exists() if steps_dir else 'N/A'}")

        steps = history.list_step_codes(history_session_id)
        logger.debug(f"[download-code] Found {len(steps)} steps for history session {history_session_id}")

        if not steps:
            # Provide more context for debugging
            detail = "No code available for this session."
            if history_session_id:
                detail += f" History session {history_session_id} has no steps."
                if steps_dir.exists():
                    # Check what's in the directory
                    try:
                        contents = list(steps_dir.iterdir())
                        detail += f" Steps dir exists with {len(contents)} files."
                    except Exception:
                        pass
            detail += " Run a query first to generate step code."
            raise HTTPException(status_code=404, detail=detail)

        # Get facts from the _facts table (if session is in memory and has datastore)
        facts_list = []
        if managed and managed.session.datastore:
            try:
                facts_df = managed.session.datastore.load_dataframe("_facts")
                for _, row in facts_df.iterrows():
                    facts_list.append({
                        "name": row.get("name", ""),
                        "value": row.get("value", ""),
                        "description": row.get("description", ""),
                    })
            except Exception:
                # No facts table - that's okay
                pass

        # Get data sources from config (if session is in memory)
        databases = []
        apis = []
        files = []

        if managed and managed.session.config:
            config = managed.session.config
            if config.databases:
                for name, db_config in config.databases.items():
                    if db_config.is_file_source():
                        files.append({
                            "name": name,
                            "path": db_config.path,
                            "description": db_config.description or "",
                        })
                    else:
                        databases.append({
                            "name": name,
                            "type": db_config.type or "sql",
                            "uri": db_config.uri or "",
                            "description": db_config.description or "",
                        })

            if config.apis:
                for name, api_config in config.apis.items():
                    apis.append({
                        "name": name,
                        "type": api_config.type,
                        "url": api_config.url or "",
                        "description": api_config.description or "",
                    })

        # Build standalone Python script
        script_lines = [
            '#!/usr/bin/env python3',
            '"""',
            f'Constat Analysis Script - Session {session_id[:8]}',
            '',
            'This script was automatically generated from a Constat analysis session.',
            'It contains all the step code that was executed during the analysis.',
            '',
            'Usage:',
            '  1. Edit _facts.parquet with your context values, then run:',
            '     python script.py',
            '',
            '  2. Or call run_analysis() directly with your values:',
            '     from script import run_analysis',
        ]

        # Add example call with explicit args
        if facts_list:
            args_example = ", ".join(f'{f["name"]}="..."' for f in facts_list)
            script_lines.append(f'     run_analysis({args_example})')
        script_lines.extend([
            '"""',
            '',
            'import os',
            'import pandas as pd',
            'import numpy as np',
            'import duckdb',
            'from pathlib import Path',
        ])

        # Add SQLAlchemy import if there are SQL databases
        if any(db['type'] in ('sql', 'postgresql', 'mysql', 'sqlite') for db in databases):
            script_lines.append('from sqlalchemy import create_engine')

        # Add data sources section if there are any
        if databases or apis or files:
            script_lines.extend([
                '',
                '# ============================================================================',
                '# Data Sources (from Constat config)',
                '# ============================================================================',
                '# Configure these for your environment. Values containing secrets should',
                '# use environment variables: os.environ["VAR_NAME"]',
                '',
            ])

            # Helper to format multi-line descriptions as comments
            def format_description_comment(desc: str, prefix: str = "#   ") -> list[str]:
                """Format a description as properly wrapped comment lines."""
                if not desc:
                    return []
                # Split on newlines and wrap each line
                lines = []
                for line in desc.replace('\n', ' ').split('. '):
                    line = line.strip()
                    if line:
                        if not line.endswith('.'):
                            line += '.'
                        lines.append(f"{prefix}{line}")
                return lines

            if databases:
                script_lines.append('# Databases')
                for db in databases:
                    uri = db['uri']
                    # Mask passwords in URIs for safety, suggest env var
                    if '@' in uri and ':' in uri.split('@')[0]:
                        # Has embedded credentials - suggest env var
                        script_lines.append(f"# db_{db['name']}: {db['type']} - credentials detected, use env var")
                        script_lines.append(f"# db_{db['name']} = create_engine(os.environ['DB_{db['name'].upper()}_URI'])")
                        # Also show masked version
                        masked = uri.split('://')[0] + '://***:***@' + uri.split('@')[-1] if '://' in uri else uri
                        script_lines.append(f"# Original (masked): {masked}")
                    else:
                        script_lines.append(f"# db_{db['name']}: {db['type']}")
                        # Add description as wrapped comment lines
                        if db['description']:
                            script_lines.extend(format_description_comment(db['description']))
                        if 'duckdb' in uri.lower():
                            # DuckDB uses its own connect() method
                            script_lines.append(f"db_{db['name']} = duckdb.connect('{uri.replace('duckdb:///', '')}')")
                        else:
                            # SQLite, PostgreSQL, MySQL, etc. use SQLAlchemy
                            script_lines.append(f"db_{db['name']} = create_engine('{uri}')")
                    script_lines.append('')

            if apis:
                script_lines.append('# APIs')
                for api in apis:
                    script_lines.append(f"# api_{api['name']}: {api['type']} - {api['url']}")
                    # Add description as wrapped comment lines
                    if api['description']:
                        script_lines.extend(format_description_comment(api['description']))
                    script_lines.append(f"# api_{api['name']} = ...  # Configure your API client")
                    script_lines.append('')

            if files:
                script_lines.append('# Files')
                for f in files:
                    script_lines.append(f"# file_{f['name']}")
                    # Add description as wrapped comment lines
                    if f['description']:
                        script_lines.extend(format_description_comment(f['description']))
                    script_lines.append(f"file_{f['name']} = Path('{f['path']}')")
                script_lines.append('')

        script_lines.extend([
            '',
            '# ============================================================================',
            '# Helper Functions',
            '# ============================================================================',
            '',
            '_dataframes: dict[str, pd.DataFrame] = {}',
            '',
            'def save_dataframe(name: str, df: pd.DataFrame) -> None:',
            '    """Save a DataFrame to the local cache."""',
            '    _dataframes[name] = df',
            '    print(f"Saved table: {name} ({len(df)} rows)")',
            '',
            'def load_dataframe(name: str) -> pd.DataFrame:',
            '    """Load a DataFrame from the local cache."""',
            '    if name not in _dataframes:',
            '        raise ValueError(f"Table not found: {name}")',
            '    return _dataframes[name]',
            '',
            'def query(sql: str) -> pd.DataFrame:',
            '    """Execute SQL query against local DataFrames."""',
            '    conn = duckdb.connect()',
            '    for table_name, df in _dataframes.items():',
            '        conn.register(table_name, df)',
            '    result = conn.execute(sql).fetchdf()',
            '    conn.close()',
            '    return result',
            '',
            '# ============================================================================',
            '# Step Functions',
            '# ============================================================================',
            '',
        ])

        # Add step functions
        for step in steps:
            step_num = step.get("step_number", 0)
            goal = step.get("goal", "Unknown goal")
            code = step.get("code", "pass")

            script_lines.append(f'def step_{step_num}(facts: dict):')
            script_lines.append(f'    """Step {step_num}: {goal}"""')

            # Indent the code properly
            for line in code.split('\n'):
                if line.strip():
                    script_lines.append(f'    {line}')
                else:
                    script_lines.append('')

            script_lines.append('')

        # Build run_analysis function with explicit fact arguments
        script_lines.append('# ============================================================================')
        script_lines.append('# Main Analysis Function')
        script_lines.append('# ============================================================================')
        script_lines.append('')

        # Build function signature with explicit args
        if facts_list:
            args_with_types = ", ".join(f'{f["name"]}: str' for f in facts_list)
            script_lines.append(f'def run_analysis({args_with_types}):')
            script_lines.append('    """')
            script_lines.append('    Run the complete analysis with the given facts.')
            script_lines.append('')
            script_lines.append('    Args:')
            for fact in facts_list:
                desc = fact["description"] or "No description"
                script_lines.append(f'        {fact["name"]}: {desc}')
            script_lines.append('    """')
            # Build facts dict from explicit args
            script_lines.append('    facts = {')
            for fact in facts_list:
                script_lines.append(f'        "{fact["name"]}": {fact["name"]},')
            script_lines.append('    }')
        else:
            script_lines.append('def run_analysis():')
            script_lines.append('    """Run the complete analysis."""')
            script_lines.append('    facts = {}')

        script_lines.append('')
        for step in steps:
            step_num = step.get("step_number", 0)
            goal = step.get("goal", "Unknown goal")
            script_lines.append(f'    print("\\n=== Step {step_num}: {goal} ===")')
            script_lines.append(f'    step_{step_num}(facts)')
        script_lines.append('')

        # Add main function that loads facts and calls run_analysis
        script_lines.append('')
        script_lines.append('# ============================================================================')
        script_lines.append('# Main Entry Point')
        script_lines.append('# ============================================================================')
        script_lines.append('')
        script_lines.append('def main():')
        script_lines.append('    """Load facts from _facts.parquet and run analysis."""')

        if facts_list:
            # Generate the facts table schema comment
            script_lines.append('    # Expected _facts.parquet schema:')
            script_lines.append('    #   name (str)         | value (str)')
            script_lines.append('    #   -------------------+' + '-' * 40)
            for fact in facts_list:
                desc = fact["description"][:35] + "..." if len(fact.get("description", "")) > 38 else fact.get("description", "")
                script_lines.append(f'    #   {fact["name"]:<18} | {desc}')
            script_lines.append('    #')
            script_lines.append('    facts_df = pd.read_parquet("_facts.parquet")')
            script_lines.append('    facts = dict(zip(facts_df["name"], facts_df["value"]))')
            script_lines.append('')
            # Call run_analysis with explicit args from facts dict
            args_from_dict = ", ".join(f'{f["name"]}=facts["{f["name"]}"]' for f in facts_list)
            script_lines.append(f'    run_analysis({args_from_dict})')
        else:
            script_lines.append('    run_analysis()')

        script_lines.extend([
            '',
            '',
            'if __name__ == "__main__":',
            '    main()',
            '',
        ])

        script_content = '\n'.join(script_lines)

        return Response(
            content=script_content,
            media_type="text/x-python",
            headers={
                "Content-Disposition": f'attachment; filename="session_{session_id[:8]}_code.py"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading code: {e}")
        raise HTTPException(status_code=500, detail=str(e))
