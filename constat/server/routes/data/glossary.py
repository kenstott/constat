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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from constat.server.persona_config import require_write

from constat.discovery.models import GlossaryTerm, display_entity_name
from constat.server.models import (
    GlossaryBulkStatusRequest,
    GlossaryCreateRequest,
    GlossaryListResponse,
    GlossaryTermResponse,
    GlossaryUpdateRequest,
    TaxonomySuggestion,
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


def _make_term_id(name: str, scope_id: str, domain: str | None = None) -> str:
    key = f"{name}:{scope_id}:{domain or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@router.get("/{session_id}/glossary", response_model=GlossaryListResponse)
async def list_glossary(
    session_id: str,
    scope: str = Query(default="all", description="all | defined | self_describing"),
    domain: str | None = Query(default=None, description="Filter by domain"),
    session_manager: SessionManager = Depends(get_session_manager),
) -> GlossaryListResponse:
    """Get the unified glossary for a session.

    Queries NER-extracted entities joined with glossary_terms from the DB.
    All entities come from NER extraction — every entity has chunk references.
    """
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    active_domains = getattr(managed, "active_domains", []) or []

    rows = vs.get_unified_glossary(session_id, scope=scope, active_domains=active_domains, user_id=managed.user_id)

    # Optionally filter by domain
    if domain:
        rows = [r for r in rows if r.get("domain") == domain]

    terms = []
    for row in sorted(rows, key=lambda r: r["name"]):
        terms.append(GlossaryTermResponse(
            name=row["name"],
            display_name=row["display_name"],
            definition=row.get("definition"),
            domain=row.get("domain"),
            parent_id=row.get("parent_id"),
            parent_verb=row.get("parent_verb") or "has",
            aliases=row.get("aliases") or [],
            semantic_type=row.get("semantic_type"),
            cardinality=row.get("cardinality") or "many",
            status=row.get("status"),
            provenance=row.get("provenance"),
            glossary_status=row.get("glossary_status") or "self_describing",
            entity_id=row.get("entity_id"),
            glossary_id=row.get("glossary_id"),
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

    active_domains = getattr(managed, "active_domains", []) or []
    terms = vs.get_deprecated_glossary(session_id, active_domains=active_domains, user_id=managed.user_id)
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

    term = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
    active_domains = getattr(managed, "active_domains", []) or []

    # Build response from unified view or just entity
    from constat.discovery.glossary_generator import resolve_physical_resources, is_grounded

    resources = resolve_physical_resources(name, session_id, vs, domain_ids=active_domains, user_id=managed.user_id)
    grounded = is_grounded(name, session_id, vs, user_id=managed.user_id)

    # Resolve parent — parent_id can be glossary_id or entity_id
    parent_info = None
    if term and term.parent_id:
        parent_term = vs.get_glossary_term_by_id(term.parent_id)
        if parent_term:
            parent_info = {"name": parent_term.name, "display_name": parent_term.display_name}
        else:
            parent_entity = vs.get_entity_by_id(term.parent_id)
            if parent_entity:
                parent_info = {"name": parent_entity.name, "display_name": parent_entity.display_name}

    # Resolve children — parent_id can be glossary_id or entity_id
    children = []
    lookup_name = term.name if term else name
    entity = vs.find_entity_by_name(lookup_name, session_id=session_id)
    candidate_ids = []
    if term:
        candidate_ids.append(term.id)
    if entity:
        candidate_ids.append(entity.id)
    if candidate_ids:
        child_terms = vs.get_child_terms(candidate_ids[0], *candidate_ids[1:])
        children = [{"name": c.name, "display_name": c.display_name, "parent_verb": c.parent_verb} for c in child_terms]

    # Resolve SVO relationships (keyed by name)
    rels = vs.get_relationships_for_entity(lookup_name, session_id)
    relationships = [
        {
            "id": r["id"],
            "subject": r["subject_name"],
            "verb": r["verb"],
            "object": r["object_name"],
            "confidence": r["confidence"],
        }
        for r in rels
    ]
    logger.debug(f"get_glossary_term({name}): {len(relationships)} relationships, parent={bool(parent_info)}, children={len(children)}")

    if term:
        return {
            "name": term.name,
            "display_name": term.display_name,
            "definition": term.definition,
            "domain": term.domain,
            "parent_id": term.parent_id,
            "parent_verb": term.parent_verb,
            "parent": parent_info,
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
            "children": children,
            "relationships": relationships,
        }

    # No glossary term — check if entity exists (self-describing)
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
            "children": children,
            "relationships": relationships,
        }

    raise HTTPException(status_code=404, detail=f"Term '{name}' not found")


@router.get("/{session_id}/glossary/{name}/relationships")
async def get_term_relationships(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get SVO relationships for a glossary term."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    svo_rels = vs.get_relationships_for_entity(name, session_id)

    # Also get co-occurrence suggestions
    from constat.discovery.glossary_generator import suggest_cooccurrence_relationships
    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    cooccurrence = suggest_cooccurrence_relationships(session_id, terms, vs, min_cooccurrence=2)
    # Filter to this entity
    cooccurrence = [
        c for c in cooccurrence
        if c["source"].lower() == name.lower() or c["target"].lower() == name.lower()
    ]

    return {
        "name": name,
        "svo_relationships": svo_rels,
        "cooccurrence_suggestions": cooccurrence,
    }


@router.post("/{session_id}/relationships", dependencies=[Depends(require_write("glossary"))])
async def create_relationship(
    session_id: str,
    request: Request,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Create a new SVO relationship between two entities."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    body = await request.json()
    subject_name = body.get("subject_name")
    verb = body.get("verb")
    object_name = body.get("object_name")

    if not subject_name or not verb or not object_name:
        raise HTTPException(status_code=422, detail="subject_name, verb, and object_name are required")

    import uuid
    from constat.discovery.models import EntityRelationship
    from constat.discovery.relationship_extractor import categorize_verb

    rel = EntityRelationship(
        id=str(uuid.uuid4()),
        subject_name=subject_name.lower(),
        verb=verb,
        object_name=object_name.lower(),
        confidence=1.0,
        verb_category=categorize_verb(verb.lower()),
        session_id=session_id,
    )
    vs.add_entity_relationship(rel)

    return {
        "status": "created",
        "id": rel.id,
        "subject": subject_name,
        "verb": verb,
        "object": object_name,
    }


@router.put("/{session_id}/relationships/{rel_id}", dependencies=[Depends(require_write("glossary"))])
async def update_relationship_verb(
    session_id: str,
    rel_id: str,
    request: Request,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Update the verb of an existing relationship."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    body = await request.json()
    verb = body.get("verb")
    if not verb:
        raise HTTPException(status_code=422, detail="verb is required")

    updated = vs.update_entity_relationship_verb(rel_id, verb)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Relationship '{rel_id}' not found")

    return {"status": "updated", "id": rel_id, "verb": verb}


@router.delete("/{session_id}/relationships/{rel_id}", dependencies=[Depends(require_write("glossary"))])
async def delete_relationship(
    session_id: str,
    rel_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Delete a relationship."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    deleted = vs.delete_entity_relationship(rel_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Relationship '{rel_id}' not found")

    return {"status": "deleted", "id": rel_id}


@router.post("/{session_id}/glossary/generate", dependencies=[Depends(require_write("glossary"))])
async def generate_glossary(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Re-trigger LLM glossary generation."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Cancel any in-progress generation before clearing and restarting
    managed._glossary_cancelled.set()

    # Clear existing LLM-generated terms and relationships (user-scoped)
    vs.clear_session_glossary(session_id, user_id=managed.user_id)
    vs.clear_session_relationships(session_id)

    # Run LLM glossary generation in background
    import threading

    def _run():
        session_manager._run_glossary_generation(session_id, managed)

    thread = threading.Thread(target=_run, name=f"glossary-gen-{session_id}", daemon=True)
    thread.start()

    return {"status": "generating", "message": "Glossary generation started"}


@router.post("/{session_id}/glossary", dependencies=[Depends(require_write("glossary"))])
async def add_definition(
    session_id: str,
    request: GlossaryCreateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Add a definition to a term."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Check if term already has a definition (user-scoped)
    existing = vs.get_glossary_term(request.name, session_id, user_id=managed.user_id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Term '{request.name}' already has a definition. Use PUT to update.")

    term = GlossaryTerm(
        id=_make_term_id(request.name, managed.user_id, request.domain),
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
        user_id=managed.user_id,
    )

    vs.add_glossary_term(term)

    return {
        "status": "created",
        "name": term.name,
        "display_name": term.display_name,
    }


@router.put("/{session_id}/glossary/{name}", dependencies=[Depends(require_write("glossary"))])
async def update_definition(
    session_id: str,
    name: str,
    request: GlossaryUpdateRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Update a glossary term definition."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    updates = {}
    if request.definition is not None:
        updates["definition"] = request.definition
    if request.aliases is not None:
        updates["aliases"] = request.aliases
    if request.parent_id is not None:
        updates["parent_id"] = request.parent_id
    if request.parent_verb is not None:
        updates["parent_verb"] = request.parent_verb
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

    logger.info(f"update_definition({name}): updates={updates}")
    result = vs.update_glossary_term(name, session_id, updates, user_id=managed.user_id)
    logger.info(f"update_definition({name}): result={result}")

    return {"status": "updated", "name": name}


@router.delete("/{session_id}/glossary", dependencies=[Depends(require_write("glossary"))])
async def delete_glossary_by_status(
    session_id: str,
    status: str = "draft",
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Delete all glossary terms matching a status (default: draft)."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Cancel any in-progress glossary generation so it doesn't re-create drafts
    managed._glossary_cancelled.set()

    count = vs.delete_glossary_by_status(session_id, status, user_id=managed.user_id)
    return {"status": "deleted", "count": count}


@router.delete("/{session_id}/glossary/{name}", dependencies=[Depends(require_write("glossary"))])
async def delete_definition(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Remove a definition (term reverts to self-describing)."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    deleted = vs.delete_glossary_term(name, session_id, user_id=managed.user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    return {"status": "deleted", "name": name}


@router.post("/{session_id}/glossary/{name}/draft-definition", dependencies=[Depends(require_write("glossary"))])
async def draft_definition(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """AI-generate a draft definition for a term that lacks one."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    if not session.router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    system = (
        "Write a concise business glossary definition for the given term. "
        "Describe its meaning and purpose, not its storage. "
        "Return only the definition text, nothing else."
    )
    user_msg = f"Term: {display_entity_name(name)}\n\nContext:\n{context}"

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success or not result.content:
        raise HTTPException(status_code=500, detail="Draft generation failed")

    return {
        "status": "ok",
        "name": name,
        "draft": result.content.strip().strip('"'),
    }


@router.post("/{session_id}/glossary/{name}/draft-aliases", dependencies=[Depends(require_write("glossary"))])
async def draft_aliases(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """AI-generate draft aliases for a glossary term."""
    import json as _json

    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    if not session.router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    # Collect all existing aliases across all terms in the domain for cross-term dedup
    all_aliases: set[str] = set()
    all_term_names: set[str] = set()
    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    for t in terms:
        all_term_names.add(t.name.lower())
        if t.display_name:
            all_term_names.add(t.display_name.lower())
        for a in (t.aliases or []):
            all_aliases.add(a.lower())

    existing_list = ", ".join(sorted(all_aliases | all_term_names)) if (all_aliases or all_term_names) else "none"

    system = (
        "Generate 3-5 alternative names or aliases for the given glossary term. "
        "These should be synonyms, abbreviations, or alternative phrasings that users might use. "
        "Do NOT include any of the existing aliases or term names listed below. "
        "Return ONLY a JSON array of strings, nothing else."
    )
    user_msg = (
        f"Term: {display_entity_name(name)}\n\n"
        f"Context:\n{context}\n\n"
        f"Existing aliases and term names (DO NOT reuse any of these):\n{existing_list}"
    )

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success or not result.content:
        raise HTTPException(status_code=500, detail="Alias generation failed")

    # Parse JSON array from response
    content = result.content.strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        aliases = _json.loads(content)
        if not isinstance(aliases, list):
            aliases = []
    except _json.JSONDecodeError:
        aliases = []

    # Dedup against all existing aliases (case-insensitive)
    forbidden = all_aliases | all_term_names
    aliases = [a for a in aliases if isinstance(a, str) and a.lower() not in forbidden]

    return {
        "status": "ok",
        "name": name,
        "aliases": aliases,
    }


@router.post("/{session_id}/glossary/{name}/refine", dependencies=[Depends(require_write("glossary"))])
async def refine_definition(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """AI-assisted definition refinement."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Term '{name}' not found")

    if not session.router:
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
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
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
    }, user_id=managed.user_id)

    return {
        "status": "refined",
        "name": name,
        "before": before,
        "after": refined,
    }


@router.post("/{session_id}/glossary/suggest-taxonomy", dependencies=[Depends(require_write("glossary"))])
async def suggest_taxonomy(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """LLM-assisted taxonomy suggestions for glossary terms."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    if not session.router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    if len(terms) < 2:
        return {"suggestions": [], "message": "Need at least 2 defined terms"}

    term_descriptions = "\n".join(
        f"- {t.display_name}: {t.definition}" for t in terms if t.definition
    )

    system = (
        "You are organizing business glossary terms into a taxonomy. "
        "Suggest parent-child relationships between terms. "
        "Only suggest relationships where a clear is-a or belongs-to relationship exists. "
        "Respond as a JSON array: [{\"child\": \"...\", \"parent\": \"...\", \"confidence\": \"high|medium|low\", \"reason\": \"...\"}]"
    )
    user_msg = f"Terms:\n{term_descriptions}"

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="medium",
    )

    if not result.success:
        raise HTTPException(status_code=500, detail="Taxonomy suggestion failed")

    from constat.discovery.glossary_generator import _parse_llm_response
    parsed = _parse_llm_response(result.content)

    suggestions = []
    for item in parsed:
        child = item.get("child", "")
        parent = item.get("parent", "")
        if child and parent:
            suggestions.append({
                "child": child,
                "parent": parent,
                "confidence": item.get("confidence", "medium"),
                "reason": item.get("reason", ""),
            })

    return {"suggestions": suggestions}


@router.patch("/{session_id}/glossary/bulk-status", dependencies=[Depends(require_write("glossary"))])
async def bulk_update_status(
    session_id: str,
    request: GlossaryBulkStatusRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Update status for multiple glossary terms."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    updated = []
    failed = []
    for name in request.names:
        ok = vs.update_glossary_term(name, session_id, {"status": request.status}, user_id=managed.user_id)
        if ok:
            updated.append(name)
        else:
            failed.append(name)

    return {
        "status": "updated",
        "updated": updated,
        "failed": failed,
        "count": len(updated),
    }


@router.post("/{session_id}/glossary/persist", dependencies=[Depends(require_write("glossary"))])
async def persist_glossary(
    session_id: str,
    request: Request,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Write approved glossary terms to domain YAML."""
    import yaml

    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    config = request.app.state.config

    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    domain_filter = body.get("domain")

    # Get approved terms (user-scoped)
    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    approved = [t for t in terms if t.status == "approved"]
    if domain_filter:
        approved = [t for t in approved if t.domain == domain_filter]

    if not approved:
        return {"status": "no_terms", "message": "No approved terms to persist"}

    # Group by domain
    by_domain: dict[str, list[GlossaryTerm]] = {}
    for t in approved:
        key = t.domain or "__base__"
        by_domain.setdefault(key, []).append(t)

    persisted_count = 0
    for domain_key, domain_terms in by_domain.items():
        if domain_key == "__base__":
            continue  # Skip base config — only persist to domain files

        domain_cfg = config.load_domain(domain_key)
        if not domain_cfg or not domain_cfg.source_path:
            continue

        domain_path = Path(domain_cfg.source_path)
        if not domain_path.exists():
            continue

        # Load existing YAML
        existing_yaml = yaml.safe_load(domain_path.read_text()) or {}

        # Build glossary dict
        glossary_dict = existing_yaml.get("glossary", {})
        for t in domain_terms:
            entry: dict[str, Any] = {"definition": t.definition}
            if t.aliases:
                entry["aliases"] = t.aliases
            if t.semantic_type and t.semantic_type != "concept":
                entry["category"] = t.semantic_type
            glossary_dict[t.name] = entry

        existing_yaml["glossary"] = glossary_dict

        # Write back
        domain_path.write_text(yaml.dump(existing_yaml, default_flow_style=False, sort_keys=False))
        persisted_count += len(domain_terms)

    return {
        "status": "persisted",
        "count": persisted_count,
    }
