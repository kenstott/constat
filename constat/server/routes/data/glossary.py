# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Glossary CRUD and AI endpoints."""

import hashlib
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from constat.discovery.models import GlossaryTerm, display_entity_name
from constat.server.models import (
    GlossaryCreateRequest,
    GlossaryListResponse,
    GlossaryTermResponse,
    GlossaryUpdateRequest,
)
from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_vector_store(managed):
    """Get vector store from managed session."""
    session = managed.session
    if hasattr(session, "doc_tools") and session.doc_tools:
        vs = session.doc_tools._vector_store
        if vs:
            return vs
    raise HTTPException(status_code=503, detail="Vector store not available")


def _make_term_id(name: str, session_id: str, domain: str | None = None) -> str:
    key = f"{name}:{session_id}:{domain or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@router.get("/{session_id}/glossary", response_model=GlossaryListResponse)
async def list_glossary(
    session_id: str,
    scope: str = Query(default="all", description="all | defined | self_describing"),
    domain: str | None = Query(default=None, description="Filter by domain"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> GlossaryListResponse:
    """Get the unified glossary for a session."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    rows = vs.get_unified_glossary(session_id, scope=scope)

    # Optionally filter by domain
    if domain:
        rows = [r for r in rows if r.get("domain") == domain]

    terms = []
    for row in rows:
        terms.append(GlossaryTermResponse(
            name=row["name"],
            display_name=row["display_name"],
            definition=row.get("definition"),
            domain=row.get("domain"),
            parent_id=row.get("parent_id"),
            aliases=row.get("aliases", []),
            semantic_type=row.get("semantic_type"),
            cardinality=row.get("cardinality", "many"),
            status=row.get("status"),
            provenance=row.get("provenance"),
            glossary_status=row.get("glossary_status", "self_describing"),
            entity_id=row.get("entity_id"),
            ner_type=row.get("ner_type"),
        ))

    defined = sum(1 for t in terms if t.glossary_status == "defined")
    self_describing = sum(1 for t in terms if t.glossary_status == "self_describing")

    return GlossaryListResponse(
        terms=terms,
        total_defined=defined,
        total_self_describing=self_describing,
    )


@router.get("/{session_id}/glossary/deprecated")
async def list_deprecated(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get deprecated glossary terms (no matching entity)."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    terms = vs.get_deprecated_glossary(session_id)
    return {
        "terms": [
            {
                "name": t.name,
                "display_name": t.display_name,
                "definition": t.definition,
                "domain": t.domain,
                "status": t.status,
                "provenance": t.provenance,
            }
            for t in terms
        ],
        "count": len(terms),
    }


@router.get("/{session_id}/glossary/{name}")
async def get_glossary_term(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get a specific glossary term with physical resources."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    term = vs.get_glossary_term(name, session_id)

    # Build response from unified view or just entity
    from constat.discovery.glossary_generator import resolve_physical_resources, is_grounded

    resources = resolve_physical_resources(name, session_id, vs)
    grounded = is_grounded(name, session_id, vs)

    if term:
        return {
            "name": term.name,
            "display_name": term.display_name,
            "definition": term.definition,
            "domain": term.domain,
            "parent_id": term.parent_id,
            "aliases": term.aliases,
            "semantic_type": term.semantic_type,
            "cardinality": term.cardinality,
            "plural": term.plural,
            "list_of": term.list_of,
            "tags": term.tags,
            "owner": term.owner,
            "status": term.status,
            "provenance": term.provenance,
            "glossary_status": "defined",
            "grounded": grounded,
            "connected_resources": resources,
        }

    # No glossary term — check if entity exists (self-describing)
    entity = vs.find_entity_by_name(name, session_id=session_id)
    if entity:
        return {
            "name": entity.name,
            "display_name": entity.display_name,
            "definition": None,
            "domain": entity.domain_id,
            "semantic_type": entity.semantic_type,
            "glossary_status": "self_describing",
            "grounded": True,
            "connected_resources": resources,
        }

    raise HTTPException(status_code=404, detail=f"Term '{name}' not found")


@router.post("/{session_id}/glossary/generate")
async def generate_glossary(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Re-trigger LLM glossary generation."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Clear existing LLM-generated terms
    vs.clear_session_glossary(session_id)

    # Trigger async generation
    session_manager.refresh_entities_async(session_id)

    return {"status": "generating", "message": "Glossary generation started"}


@router.post("/{session_id}/glossary")
async def add_definition(
    session_id: str,
    request: GlossaryCreateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Add a definition to a term."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Check if term already has a definition
    existing = vs.get_glossary_term(request.name, session_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Term '{request.name}' already has a definition. Use PUT to update.")

    term = GlossaryTerm(
        id=_make_term_id(request.name, session_id, request.domain),
        name=request.name.lower(),
        display_name=display_entity_name(request.name),
        definition=request.definition,
        domain=request.domain,
        parent_id=request.parent_id,
        aliases=request.aliases,
        semantic_type=request.semantic_type or "concept",
        status="draft",
        provenance="human",
        session_id=session_id,
    )

    vs.add_glossary_term(term)

    return {
        "status": "created",
        "name": term.name,
        "display_name": term.display_name,
    }


@router.put("/{session_id}/glossary/{name}")
async def update_definition(
    session_id: str,
    name: str,
    request: GlossaryUpdateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Update a glossary term definition."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    existing = vs.get_glossary_term(name, session_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    updates = {}
    if request.definition is not None:
        updates["definition"] = request.definition
    if request.aliases is not None:
        updates["aliases"] = request.aliases
    if request.parent_id is not None:
        updates["parent_id"] = request.parent_id
    if request.status is not None:
        updates["status"] = request.status
    if request.domain is not None:
        updates["domain"] = request.domain
    if request.semantic_type is not None:
        updates["semantic_type"] = request.semantic_type

    # Update provenance to hybrid if was LLM-generated
    if existing.provenance == "llm" and updates:
        updates["provenance"] = "hybrid"

    if not updates:
        return {"status": "no_changes"}

    vs.update_glossary_term(name, session_id, updates)

    return {"status": "updated", "name": name}


@router.delete("/{session_id}/glossary/{name}")
async def delete_definition(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Remove a definition (term reverts to self-describing)."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    deleted = vs.delete_glossary_term(name, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    return {"status": "deleted", "name": name}


@router.post("/{session_id}/glossary/{name}/refine")
async def refine_definition(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """AI-assisted definition refinement."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    existing = vs.get_glossary_term(name, session_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    if not hasattr(session, '_router') or not session._router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    # Build context
    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    system = (
        "Improve this business glossary definition. Keep it concise. "
        "Describe meaning, not storage. Return only the improved definition text."
    )
    user_msg = (
        f"Term: {existing.display_name}\n"
        f"Current definition: {existing.definition}\n\n"
        f"Context:\n{context}"
    )

    from constat.core.models import TaskType
    result = session._router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=512,
        complexity="low",
    )

    if not result.success:
        raise HTTPException(status_code=500, detail="Refinement failed")

    refined = result.content.strip().strip('"')
    before = existing.definition

    # Update the definition
    vs.update_glossary_term(name, session_id, {
        "definition": refined,
        "provenance": "hybrid" if existing.provenance == "llm" else existing.provenance,
    })

    return {
        "status": "refined",
        "name": name,
        "before": before,
        "after": refined,
    }
