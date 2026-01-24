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
        return ArtifactListResponse(
            artifacts=[
                ArtifactInfo(
                    id=a["id"],
                    name=a["name"],
                    artifact_type=a["type"],
                    step_number=a.get("step_number", 0),
                    title=a.get("title"),
                    description=a.get("description"),
                    mime_type=a.get("content_type", "application/octet-stream"),
                    created_at=a.get("created_at"),
                )
                for a in artifacts
            ]
        )
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
        print(f"[list_facts] Retrieved {len(all_facts)} facts from fact_resolver")
        for name, fact in all_facts.items():
            print(f"[list_facts] Fact: {name} = {fact.value} (source: {fact.source})")

        return FactListResponse(
            facts=[
                FactInfo(
                    name=name,
                    value=fact.value,
                    source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                    reasoning=fact.reasoning,
                    confidence=getattr(fact, "confidence", None),
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

    Args:
        session_id: Session ID
        entity_type: Optional filter by type (table, column, concept, etc.)

    Returns:
        List of entities

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    entities = []
    seen_ids = set()

    # Try to get entities from session's entity extractor if available
    try:
        if hasattr(managed.session, "entity_extractor"):
            all_entities = managed.session.entity_extractor.get_entities()
            for ent in all_entities:
                if entity_type and ent.type.value != entity_type:
                    continue
                ent_id = str(hash(f"{ent.name}_{ent.type.value}"))
                if ent_id not in seen_ids:
                    seen_ids.add(ent_id)
                    entities.append({
                        "id": ent_id,
                        "name": ent.name,
                        "type": ent.type.value,
                        "metadata": ent.metadata or {},
                        "mention_count": getattr(ent, "mention_count", 0),
                    })
    except Exception as e:
        logger.warning(f"Could not get entities from entity_extractor: {e}")

    # Also get schema entities (tables and columns) from schema_manager
    try:
        print(f"[list_entities] Checking schema_manager: {managed.session.schema_manager}")
        if managed.session.schema_manager:
            metadata_cache = managed.session.schema_manager.metadata_cache
            print(f"[list_entities] metadata_cache has {len(metadata_cache)} tables")
            for full_name, table_meta in metadata_cache.items():
                print(f"[list_entities] Table: {full_name}, columns: {len(table_meta.columns)}")
                db_name = table_meta.database
                table_name = table_meta.name

                # Add table entity
                if not entity_type or entity_type == "table":
                    ent_id = str(hash(f"{full_name}_table"))
                    if ent_id not in seen_ids:
                        seen_ids.add(ent_id)
                        entities.append({
                            "id": ent_id,
                            "name": table_name,
                            "type": "table",
                            "metadata": {"database": db_name, "full_name": full_name},
                            "mention_count": 0,
                        })

                # Add column entities
                if not entity_type or entity_type == "column":
                    for col in table_meta.columns:
                        col_name = col.name
                        ent_id = str(hash(f"{full_name}.{col_name}_column"))
                        if ent_id not in seen_ids:
                            seen_ids.add(ent_id)
                            entities.append({
                                "id": ent_id,
                                "name": col_name,
                                "type": "column",
                                "metadata": {
                                    "table": table_name,
                                    "database": db_name,
                                    "dtype": col.type if col.type else None,
                                },
                                "mention_count": 0,
                            })
    except Exception as e:
        import traceback
        print(f"[list_entities] ERROR from schema_manager: {e}")
        print(f"[list_entities] Traceback: {traceback.format_exc()}")

    print(f"[list_entities] Returning {len(entities)} entities total")
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
    """Invalidate a cached fact.

    Args:
        session_id: Session ID
        fact_name: Name of the fact to forget

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

        # Forget the fact
        if hasattr(managed.session.fact_resolver, "forget_fact"):
            managed.session.fact_resolver.forget_fact(fact_name)

        return {"status": "forgotten", "fact_name": fact_name}

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

        # Edit the fact
        if hasattr(managed.session.fact_resolver, "set_fact"):
            managed.session.fact_resolver.set_fact(fact_name, body["value"], source="user")

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
