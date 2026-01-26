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
from fastapi.responses import Response

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
        starred_tables = set(managed.session.datastore.get_starred_tables())
        return TableListResponse(
            tables=[
                TableInfo(
                    name=t["name"],
                    row_count=t.get("row_count", 0),
                    step_number=t.get("step_number", 0),
                    columns=t.get("columns", []),
                    is_starred=t["name"] in starred_tables,
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


@router.get("/{session_id}/tables/{table_name}/download")
async def download_table(
    session_id: str,
    table_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> Response:
    """Download table data as CSV.

    Args:
        session_id: Session ID
        table_name: Table name to download

    Returns:
        CSV file response

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    if not managed.session.datastore:
        raise HTTPException(status_code=404, detail="No datastore for this session")

    try:
        df = managed.session.datastore.load_dataframe(table_name)
        csv_content = df.to_csv(index=False)

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{table_name}.csv"'
            },
        )

    except Exception as e:
        logger.error(f"Error downloading table: {e}")
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
        # Key results: visualizations (unless explicitly unstarred) OR user-starred
        visualization_types = {'chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega', 'markdown', 'md'}
        # Code types are explicitly excluded from key results (unless user-starred)
        code_types = {'code', 'python', 'sql', 'script', 'text', 'output', 'error'}

        def get_starred_and_key_result(a: dict) -> tuple[bool, bool]:
            """Get is_starred and is_key_result for an artifact.

            Returns:
                (is_starred, is_key_result) tuple
            """
            artifact_obj = managed.session.datastore.get_artifact_by_id(a["id"])
            metadata = artifact_obj.metadata if artifact_obj else {}
            artifact_type = a.get("type", "").lower()

            # is_starred: explicit user action
            is_starred = metadata.get("is_starred", False)

            # is_key_result: determines if shown in Artifacts section
            if "is_starred" in metadata:
                # User has explicitly set starred status - use that
                is_key_result = metadata["is_starred"]
            elif artifact_type in code_types:
                # Code is NEVER a key result by default
                is_key_result = False
            elif artifact_type in visualization_types:
                # Visualizations are key results by default
                is_key_result = True
            else:
                is_key_result = False

            return is_starred, is_key_result

        # Build artifact list
        artifact_list = []
        for a in artifacts:
            is_starred, is_key_result = get_starred_and_key_result(a)
            artifact_list.append(
                ArtifactInfo(
                    id=a["id"],
                    name=a["name"],
                    artifact_type=a["type"],
                    step_number=a.get("step_number", 0),
                    title=a.get("title"),
                    description=a.get("description"),
                    mime_type=a.get("content_type") or "application/octet-stream",
                    created_at=a.get("created_at"),
                    is_key_result=is_key_result,
                    is_starred=is_starred,
                )
            )

        # Add consequential tables as virtual artifacts
        # A table is consequential if it's published, from the final step, or starred
        if tables:
            starred_tables = set(managed.session.datastore.get_starred_tables())
            for t in tables:
                table_name = t["name"]
                # Skip internal tables
                if table_name.startswith("_"):
                    continue
                # Check if table is published or from final step (from registry metadata)
                is_published = t.get("is_published", False)
                is_final_step = t.get("is_final_step", False)
                # Tables with substantial data are consequential
                has_data = t.get("row_count", 0) > 0
                # Check starred status
                is_starred = table_name in starred_tables

                # Determine if table should appear in artifacts list
                # Tables appear if: published, final step with data, OR explicitly starred
                should_include = is_published or (is_final_step and has_data) or is_starred

                if should_include:
                    # Create a virtual artifact entry for this table
                    # Use negative IDs to distinguish from real artifacts
                    virtual_id = -hash(table_name) % 1000000
                    # Table is a key result if starred, or auto-published (but user can unstar)
                    # is_starred tracks explicit user action
                    # is_key_result determines if shown in Artifacts section
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
                            is_key_result=is_starred or is_published or is_final_step,
                            is_starred=is_starred,
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
# Star/Promote Endpoints
# ============================================================================


@router.post("/{session_id}/artifacts/{artifact_id}/star")
async def toggle_artifact_star(
    session_id: str,
    artifact_id: int,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Toggle an artifact's starred/key-result status.

    Args:
        session_id: Session ID
        artifact_id: Artifact ID

    Returns:
        New starred status

    Raises:
        404: Session or artifact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        if not managed.session.datastore:
            raise HTTPException(status_code=404, detail="No datastore")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if not artifact:
            raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")

        # Toggle is_starred in metadata
        current = artifact.metadata.get("is_starred", False)
        new_status = not current
        managed.session.datastore.update_artifact_metadata(artifact_id, {"is_starred": new_status})

        return {"artifact_id": artifact_id, "is_starred": new_status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling artifact star: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/tables/{table_name}/star")
async def toggle_table_star(
    session_id: str,
    table_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Toggle a table's starred status.

    Args:
        session_id: Session ID
        table_name: Table name

    Returns:
        New starred status

    Raises:
        404: Session or table not found
    """
    managed = session_manager.get_session(session_id)

    try:
        if not managed.session.datastore:
            raise HTTPException(status_code=404, detail="No datastore")

        # Verify table exists
        tables = managed.session.datastore.list_tables()
        if not any(t["name"] == table_name for t in tables):
            raise HTTPException(status_code=404, detail=f"Table not found: {table_name}")

        # Toggle starred status
        is_starred = managed.session.datastore.toggle_table_star(table_name)

        return {"table_name": table_name, "is_starred": is_starred}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling table star: {e}")
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
        llm_config = None
        email_config = None

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

            # Extract LLM config
            if config.llm:
                llm_config = {
                    "provider": config.llm.provider,
                    "model": config.llm.model,
                    "api_key": config.llm.api_key,
                    "base_url": config.llm.base_url,
                }

            # Extract email config
            if config.email:
                email_config = {
                    "smtp_host": config.email.smtp_host,
                    "smtp_port": config.email.smtp_port,
                    "smtp_user": config.email.smtp_user,
                    "smtp_password": config.email.smtp_password,
                    "from_address": config.email.from_address,
                    "from_name": config.email.from_name,
                    "tls": config.email.tls,
                }

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
                    # Add config variable for the API base URL
                    script_lines.append(f"API_{api['name'].upper()}_URL = '{api['url']}'")
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

        # Add LLM configuration section
        script_lines.extend([
            '',
            '# ============================================================================',
            '# LLM Configuration',
            '# ============================================================================',
            '',
        ])
        if llm_config and llm_config['api_key']:
            script_lines.append(f'LLM_PROVIDER = "{llm_config["provider"]}"')
            script_lines.append(f'LLM_MODEL = "{llm_config["model"]}"')
            script_lines.append(f'LLM_API_KEY = "{llm_config["api_key"]}"')
            script_lines.append(f'LLM_BASE_URL = "{llm_config["base_url"]}"' if llm_config['base_url'] else 'LLM_BASE_URL = None')
        else:
            script_lines.append('LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")')
            script_lines.append('LLM_MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")')
            script_lines.append('LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY")')
            script_lines.append('LLM_BASE_URL = os.environ.get("LLM_BASE_URL")')
        script_lines.append('')

        # Add Email configuration section
        script_lines.extend([
            '',
            '# ============================================================================',
            '# Email Configuration',
            '# ============================================================================',
            '',
        ])
        if email_config and email_config['smtp_host']:
            script_lines.append(f'SMTP_HOST = "{email_config["smtp_host"]}"')
            script_lines.append(f'SMTP_PORT = {email_config["smtp_port"]}')
            script_lines.append(f'SMTP_USER = "{email_config["smtp_user"]}"' if email_config['smtp_user'] else 'SMTP_USER = os.environ.get("SMTP_USER")')
            script_lines.append(f'SMTP_PASSWORD = "{email_config["smtp_password"]}"' if email_config['smtp_password'] else 'SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")')
            script_lines.append(f'SMTP_FROM = "{email_config["from_address"]}"')
            script_lines.append(f'SMTP_FROM_NAME = "{email_config["from_name"]}"')
            script_lines.append(f'SMTP_USE_TLS = {email_config["tls"]}')
        else:
            script_lines.append('SMTP_HOST = os.environ.get("SMTP_HOST")')
            script_lines.append('SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))')
            script_lines.append('SMTP_USER = os.environ.get("SMTP_USER")')
            script_lines.append('SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")')
            script_lines.append('SMTP_FROM = os.environ.get("SMTP_FROM")')
            script_lines.append('SMTP_FROM_NAME = "Constat"')
            script_lines.append('SMTP_USE_TLS = True')
        script_lines.append('')

        script_lines.extend([
            '',
            '# ============================================================================',
            '# Store Class (Data persistence between steps)',
            '# ============================================================================',
            '',
            'class _DataStore:',
            '    """Simple datastore for sharing data between steps."""',
            '',
            '    def __init__(self):',
            '        self._tables: dict[str, pd.DataFrame] = {}',
            '        self._state: dict[str, any] = {}',
            '        self._conn = duckdb.connect()',
            '',
            '    def save_dataframe(self, name: str, df: pd.DataFrame, step_number: int = 0, description: str = "") -> None:',
            '        """Save a DataFrame to the store."""',
            '        self._tables[name] = df',
            '        self._conn.register(name, df)',
            '        print(f"Saved table: {name} ({len(df)} rows)")',
            '',
            '    def load_dataframe(self, name: str) -> pd.DataFrame:',
            '        """Load a DataFrame from the store."""',
            '        if name not in self._tables:',
            '            raise ValueError(f"Table not found: {name}. Available: {list(self._tables.keys())}")',
            '        return self._tables[name]',
            '',
            '    def query(self, sql: str) -> pd.DataFrame:',
            '        """Execute SQL query against stored DataFrames."""',
            '        return self._conn.execute(sql).fetchdf()',
            '',
            '    def set_state(self, key: str, value: any, step_number: int = 0) -> None:',
            '        """Save a state variable."""',
            '        self._state[key] = value',
            '',
            '    def get_state(self, key: str) -> any:',
            '        """Get a state variable (returns None if not found)."""',
            '        return self._state.get(key)',
            '',
            '    def list_tables(self) -> list[str]:',
            '        """List all stored tables."""',
            '        return list(self._tables.keys())',
            '',
            '    def table_exists(self, name: str) -> bool:',
            '        """Check if a table exists."""',
            '        return name in self._tables',
            '',
            '',
            '# ============================================================================',
            '# Visualization Helper (Save charts, files, and outputs)',
            '# ============================================================================',
            '',
            'class _VizHelper:',
            '    """Helper for saving visualizations and files."""',
            '',
            '    def __init__(self, output_dir: Path = None):',
            '        self.output_dir = output_dir or Path("./outputs")',
            '        self.output_dir.mkdir(parents=True, exist_ok=True)',
            '',
            '    def _save_and_print(self, filepath: Path, description: str) -> Path:',
            '        """Print file URI and return path."""',
            '        print(f"{description}: {filepath.resolve().as_uri()}")',
            '        return filepath',
            '',
            '    def save_file(self, name: str, content: str, ext: str = "txt", title: str = None, description: str = None) -> Path:',
            '        """Save a text file."""',
            '        filepath = self.output_dir / f"{name}.{ext}"',
            '        filepath.write_text(content)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_html(self, name: str, html_content: str, title: str = None, description: str = None) -> Path:',
            '        """Save HTML content."""',
            '        return self.save_file(name, html_content, ext="html", title=title)',
            '',
            '    def save_chart(self, name: str, figure: any, title: str = None, description: str = None, chart_type: str = "plotly") -> Path:',
            '        """Save a Plotly or matplotlib chart."""',
            '        filepath = self.output_dir / f"{name}.html"',
            '        if hasattr(figure, "write_html"):  # Plotly',
            '            figure.write_html(str(filepath))',
            '        elif hasattr(figure, "savefig"):  # Matplotlib',
            '            filepath = self.output_dir / f"{name}.png"',
            '            figure.savefig(str(filepath))',
            '        else:',
            '            raise ValueError(f"Unknown figure type: {type(figure)}")',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_image(self, name: str, figure: any, format: str = "png", title: str = None, description: str = None) -> Path:',
            '        """Save a matplotlib figure as image."""',
            '        filepath = self.output_dir / f"{name}.{format}"',
            '        figure.savefig(str(filepath), format=format)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_map(self, name: str, folium_map: any, title: str = None, description: str = None) -> Path:',
            '        """Save a folium map."""',
            '        filepath = self.output_dir / f"{name}.html"',
            '        folium_map.save(str(filepath))',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_binary(self, name: str, content: bytes, ext: str = "xlsx", title: str = None, description: str = None) -> Path:',
            '        """Save binary content (Excel, images, etc.)."""',
            '        filepath = self.output_dir / f"{name}.{ext}"',
            '        filepath.write_bytes(content)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_excel(self, name: str, df: pd.DataFrame, title: str = None, sheet_name: str = "Sheet1") -> Path:',
            '        """Save DataFrame to Excel file (.xlsx). Requires openpyxl."""',
            '        try:',
            '            filepath = self.output_dir / f"{name}.xlsx"',
            '            df.to_excel(str(filepath), sheet_name=sheet_name, index=False)',
            '            return self._save_and_print(filepath, title or name)',
            '        except ModuleNotFoundError:',
            '            raise ImportError("openpyxl required: pip install openpyxl")',
            '',
            '    def save_csv(self, name: str, df: pd.DataFrame, title: str = None) -> Path:',
            '        """Save DataFrame to CSV file."""',
            '        filepath = self.output_dir / f"{name}.csv"',
            '        df.to_csv(str(filepath), index=False)',
            '        return self._save_and_print(filepath, title or name)',
            '',
            '    def save_word(self, name: str, content: str, title: str = None) -> Path:',
            '        """Save content to Word document (.docx). Requires python-docx."""',
            '        try:',
            '            from docx import Document',
            '            doc = Document()',
            '            for para in content.split("\\n\\n"):',
            '                doc.add_paragraph(para)',
            '            filepath = self.output_dir / f"{name}.docx"',
            '            doc.save(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("python-docx required: pip install python-docx")',
            '',
            '    def save_pdf(self, name: str, content: str, title: str = None) -> Path:',
            '        """Save content to PDF. Requires fpdf2."""',
            '        try:',
            '            from fpdf import FPDF',
            '            pdf = FPDF()',
            '            pdf.add_page()',
            '            pdf.set_font("Helvetica", size=11)',
            '            pdf.multi_cell(0, 5, content)',
            '            filepath = self.output_dir / f"{name}.pdf"',
            '            pdf.output(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("fpdf2 required: pip install fpdf2")',
            '',
            '    def save_powerpoint(self, name: str, slides: list[dict], title: str = None) -> Path:',
            '        """Save to PowerPoint (.pptx). Requires python-pptx.',
            '        ',
            '        Args:',
            '            slides: List of dicts with "title" and "content" keys',
            '        """',
            '        try:',
            '            from pptx import Presentation',
            '            from pptx.util import Inches, Pt',
            '            prs = Presentation()',
            '            for slide_data in slides:',
            '                slide = prs.slides.add_slide(prs.slide_layouts[1])',
            '                slide.shapes.title.text = slide_data.get("title", "")',
            '                slide.placeholders[1].text = slide_data.get("content", "")',
            '            filepath = self.output_dir / f"{name}.pptx"',
            '            prs.save(str(filepath))',
            '            return self._save_and_print(filepath, title or name)',
            '        except ImportError:',
            '            raise ImportError("python-pptx required: pip install python-pptx")',
            '',
            '',
            '# Create global instances',
            'store = _DataStore()',
            'viz = _VizHelper()',
            '',
            '# Legacy helper functions (for backwards compatibility)',
            'save_dataframe = store.save_dataframe',
            'load_dataframe = store.load_dataframe',
            'query = store.query',
            '',
            '',
            '# ============================================================================',
            '# LLM Helper Function',
            '# ============================================================================',
            '',
            'def llm_ask(question: str) -> str:',
            '    """Query an LLM for general knowledge.',
            '    ',
            '    Args:',
            '        question: The question to ask',
            '    ',
            '    Returns:',
            '        The LLM response text',
            '    """',
            '    if not LLM_API_KEY:',
            '        raise ValueError("llm_ask() requires LLM_API_KEY")',
            '    ',
            '    if LLM_PROVIDER == "anthropic":',
            '        try:',
            '            import anthropic',
            '            client = anthropic.Anthropic(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL else anthropic.Anthropic(api_key=LLM_API_KEY)',
            '            response = client.messages.create(',
            '                model=LLM_MODEL,',
            '                max_tokens=2048,',
            '                messages=[{"role": "user", "content": question}]',
            '            )',
            '            return response.content[0].text',
            '        except ImportError:',
            '            raise ImportError("anthropic package required: pip install anthropic")',
            '    elif LLM_PROVIDER == "openai":',
            '        try:',
            '            import openai',
            '            client = openai.OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if LLM_BASE_URL else openai.OpenAI(api_key=LLM_API_KEY)',
            '            response = client.chat.completions.create(',
            '                model=LLM_MODEL,',
            '                messages=[{"role": "user", "content": question}]',
            '            )',
            '            return response.choices[0].message.content',
            '        except ImportError:',
            '            raise ImportError("openai package required: pip install openai")',
            '    else:',
            '        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")',
            '',
            '',
            '# ============================================================================',
            '# Email Helper Function',
            '# ============================================================================',
            '',
            '# Basic email-safe CSS for rendered Markdown',
            '_EMAIL_CSS = """',
            '<style>',
            'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.5; color: #333; }',
            'h1, h2, h3 { color: #2c3e50; margin-top: 1em; margin-bottom: 0.5em; }',
            'p { margin: 0.5em 0; }',
            'ul, ol { margin: 0.5em 0; padding-left: 1.5em; }',
            'table { border-collapse: collapse; margin: 1em 0; }',
            'th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }',
            'th { background-color: #f5f5f5; }',
            'code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: monospace; }',
            '</style>',
            '"""',
            '',
            'def _markdown_to_html(text: str) -> str:',
            '    """Convert Markdown text to HTML with email-safe styling."""',
            '    try:',
            '        import markdown',
            '        html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "nl2br"])',
            '        return f"<!DOCTYPE html><html><head>{_EMAIL_CSS}</head><body>{html_body}</body></html>"',
            '    except ImportError:',
            '        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")',
            '        html_body = f"<pre>{escaped}</pre>"',
            '        return f"<!DOCTYPE html><html><head>{_EMAIL_CSS}</head><body>{html_body}</body></html>"',
            '',
            '',
            'def send_email(',
            '    to: str,',
            '    subject: str,',
            '    body: str,',
            '    format: str = "plain",',
            '    df: pd.DataFrame = None,',
            '    attachment_name: str = "data.csv",',
            ') -> bool:',
            '    """Send an email with optional DataFrame attachment.',
            '    ',
            '    Args:',
            '        to: Recipient email address (or comma-separated list)',
            '        subject: Email subject',
            '        body: Email body (plain text, Markdown, or HTML)',
            '        format: Body format - "plain" (default), "markdown", or "html"',
            '        df: Optional DataFrame to attach',
            '        attachment_name: Filename for the DataFrame attachment',
            '    ',
            '    Returns:',
            '        True if sent successfully',
            '    """',
            '    import smtplib',
            '    from email.mime.text import MIMEText',
            '    from email.mime.multipart import MIMEMultipart',
            '    from email.mime.base import MIMEBase',
            '    from email import encoders',
            '    ',
            '    if not all([SMTP_HOST, SMTP_USER, SMTP_PASSWORD]):',
            '        raise ValueError("send_email() requires SMTP_HOST, SMTP_USER, SMTP_PASSWORD")',
            '    ',
            '    # Handle format conversion',
            '    send_as_html = False',
            '    final_body = body',
            '    if format == "markdown":',
            '        final_body = _markdown_to_html(body)',
            '        send_as_html = True',
            '    elif format == "html":',
            '        send_as_html = True',
            '    ',
            '    # Create message',
            '    msg = MIMEMultipart()',
            '    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM}>"',
            '    msg["To"] = to',
            '    msg["Subject"] = subject',
            '    msg.attach(MIMEText(final_body, "html" if send_as_html else "plain"))',
            '    ',
            '    # Attach DataFrame if provided',
            '    if df is not None:',
            '        attachment_content = df.to_csv(index=False)',
            '        part = MIMEBase("text", "csv")',
            '        part.set_payload(attachment_content.encode("utf-8"))',
            '        encoders.encode_base64(part)',
            '        part.add_header("Content-Disposition", f"attachment; filename={attachment_name}")',
            '        msg.attach(part)',
            '    ',
            '    # Send email',
            '    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:',
            '        if SMTP_USE_TLS:',
            '            server.starttls()',
            '        server.login(SMTP_USER, SMTP_PASSWORD)',
            '        recipients = [addr.strip() for addr in to.split(",")]',
            '        server.sendmail(SMTP_FROM, recipients, msg.as_string())',
            '    ',
            '    print(f"Email sent to {to}")',
            '    return True',
            '',
        ])

        # Add API helper functions if there are APIs configured
        if apis:
            script_lines.extend([
                '',
                '# ============================================================================',
                '# API Helper Functions',
                '# ============================================================================',
                '',
                'import requests',
                '',
            ])

            for api in apis:
                api_name = api['name']
                api_type = api['type']
                api_url_var = f"API_{api_name.upper()}_URL"

                if api_type == 'graphql':
                    # GraphQL API helper
                    script_lines.extend([
                        f'def api_{api_name}(query: str, variables: dict = None) -> dict:',
                        f'    """Execute GraphQL query against {api_name} API.',
                        '    ',
                        '    Args:',
                        '        query: GraphQL query string',
                        '        variables: Optional dict of query variables',
                        '    ',
                        '    Returns:',
                        '        Response JSON data',
                        '    """',
                        f'    response = requests.post(',
                        f'        {api_url_var},',
                        '        json={"query": query, "variables": variables or {}},',
                        '        headers={"Content-Type": "application/json"}',
                        '    )',
                        '    response.raise_for_status()',
                        '    result = response.json()',
                        '    if "errors" in result:',
                        '        raise ValueError(f"GraphQL error: {result[\'errors\']}")',
                        '    return result.get("data", result)',
                        '',
                        '',
                    ])
                else:
                    # REST/OpenAPI helper
                    script_lines.extend([
                        f'def api_{api_name}(method_path: str, params: dict = None, body: dict = None) -> dict:',
                        f'    """Make REST API call to {api_name}.',
                        '    ',
                        '    Args:',
                        '        method_path: HTTP method and path, e.g., "GET /breeds" or "POST /users"',
                        '        params: Optional query parameters dict',
                        '        body: Optional request body dict (for POST/PUT/PATCH)',
                        '    ',
                        '    Returns:',
                        '        Response JSON data',
                        '    """',
                        '    parts = method_path.split(" ", 1)',
                        '    method = parts[0].upper() if len(parts) > 0 else "GET"',
                        '    path = parts[1] if len(parts) > 1 else "/"',
                        '    ',
                        f'    base_url = {api_url_var}.rstrip("/")',
                        '    if not path.startswith("/"):',
                        '        path = "/" + path',
                        '    url = base_url + path',
                        '    ',
                        '    response = requests.request(',
                        '        method=method,',
                        '        url=url,',
                        '        params=params,',
                        '        json=body if body else None,',
                        '        headers={"Accept": "application/json"}',
                        '    )',
                        '    response.raise_for_status()',
                        '    return response.json()',
                        '',
                        '',
                    ])

        script_lines.extend([
            '',
            '# ============================================================================',
            '# Publish Function (no-op in standalone mode)',
            '# ============================================================================',
            '',
            'def publish(tables: list[str] = None, artifacts: list[str] = None) -> None:',
            '    """Publish tables/artifacts (no-op in standalone mode)."""',
            '    if tables:',
            '        print(f"Would publish tables: {tables}")',
            '    if artifacts:',
            '        print(f"Would publish artifacts: {artifacts}")',
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
