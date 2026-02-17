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
    """Get the unified glossary for a session.

    Aggregates entities from all sources (NER-extracted, schema, API, config,
    domains) and joins each with glossary_terms to produce the unified view.
    """
    from constat.discovery.models import normalize_entity_name, display_entity_name as make_display_name

    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    active_domains = getattr(managed, "active_domains", []) or []

    # 1. NER-extracted entities joined with glossary_terms (from DB)
    rows = vs.get_unified_glossary(session_id, scope=scope, active_domains=active_domains)

    # Build lookup by normalized name to deduplicate across sources
    seen: dict[str, dict] = {}
    for row in rows:
        key = row["name"].lower()
        seen[key] = row

    # 2. Schema entities (tables from schema_manager)
    session = managed.session
    if session.schema_manager:
        for _full_name, table_meta in session.schema_manager.metadata_cache.items():
            norm = normalize_entity_name(table_meta.name)
            key = norm.lower()
            if key not in seen:
                seen[key] = {
                    "entity_id": None,
                    "name": norm,
                    "display_name": make_display_name(table_meta.name),
                    "semantic_type": "table",
                    "ner_type": "SCHEMA",
                    "session_id": session_id,
                    "glossary_id": None,
                    "domain": None,
                    "definition": None,
                    "parent_id": None,
                    "aliases": [],
                    "cardinality": "many",
                    "plural": None,
                    "list_of": None,
                    "status": None,
                    "provenance": None,
                    "glossary_status": "self_describing",
                }

    # 3. API entities from config
    if session.config and session.config.apis:
        for api_name in session.config.apis:
            norm = normalize_entity_name(api_name)
            key = norm.lower()
            if key not in seen:
                seen[key] = {
                    "entity_id": None,
                    "name": norm,
                    "display_name": make_display_name(api_name),
                    "semantic_type": "api",
                    "ner_type": "API",
                    "session_id": session_id,
                    "glossary_id": None,
                    "domain": None,
                    "definition": None,
                    "parent_id": None,
                    "aliases": [],
                    "cardinality": "many",
                    "plural": None,
                    "list_of": None,
                    "status": None,
                    "provenance": None,
                    "glossary_status": "self_describing",
                }

    # 4. Document entities from config
    if session.config and session.config.documents:
        for doc_name in session.config.documents:
            norm = normalize_entity_name(doc_name)
            key = norm.lower()
            if key not in seen:
                seen[key] = {
                    "entity_id": None,
                    "name": norm,
                    "display_name": make_display_name(doc_name),
                    "semantic_type": "concept",
                    "ner_type": None,
                    "session_id": session_id,
                    "glossary_id": None,
                    "domain": None,
                    "definition": None,
                    "parent_id": None,
                    "aliases": [],
                    "cardinality": "many",
                    "plural": None,
                    "list_of": None,
                    "status": None,
                    "provenance": None,
                    "glossary_status": "self_describing",
                }

    # 5. Domain entities (APIs + documents from active domains)
    if active_domains and session.config:
        for domain_filename in active_domains:
            domain_cfg = session.config.load_domain(domain_filename)
            if not domain_cfg:
                continue
            for api_name in domain_cfg.apis:
                norm = normalize_entity_name(api_name)
                key = norm.lower()
                if key not in seen:
                    seen[key] = {
                        "entity_id": None,
                        "name": norm,
                        "display_name": make_display_name(api_name),
                        "semantic_type": "api",
                        "ner_type": "API",
                        "session_id": session_id,
                        "glossary_id": None,
                        "domain": domain_filename,
                        "definition": None,
                        "parent_id": None,
                        "aliases": [],
                        "cardinality": "many",
                        "plural": None,
                        "list_of": None,
                        "status": None,
                        "provenance": None,
                        "glossary_status": "self_describing",
                    }
            for doc_name in domain_cfg.documents:
                norm = normalize_entity_name(doc_name)
                key = norm.lower()
                if key not in seen:
                    seen[key] = {
                        "entity_id": None,
                        "name": norm,
                        "display_name": make_display_name(doc_name),
                        "semantic_type": "concept",
                        "ner_type": None,
                        "session_id": session_id,
                        "glossary_id": None,
                        "domain": domain_filename,
                        "definition": None,
                        "parent_id": None,
                        "aliases": [],
                        "cardinality": "many",
                        "plural": None,
                        "list_of": None,
                        "status": None,
                        "provenance": None,
                        "glossary_status": "self_describing",
                    }

    # Check supplemental entities against glossary_terms table
    glossary_terms_by_name = {}
    supp_names = [k for k, v in seen.items() if v.get("glossary_id") is None and v.get("entity_id") is None]
    if supp_names:
        gt_rows = vs.get_glossary_terms_by_names(supp_names, session_id)
        for gt in gt_rows:
            glossary_terms_by_name[gt.name.lower()] = gt
        for key, gt in glossary_terms_by_name.items():
            if key in seen:
                row = seen[key]
                row["glossary_id"] = gt.id
                row["definition"] = gt.definition
                row["domain"] = gt.domain
                row["parent_id"] = gt.parent_id
                row["aliases"] = gt.aliases or []
                row["status"] = gt.status
                row["provenance"] = gt.provenance
                row["glossary_status"] = "defined"

    # Apply scope filter to supplemental entries
    if scope == "defined":
        seen = {k: v for k, v in seen.items() if v.get("glossary_id") is not None}
    elif scope == "self_describing":
        seen = {k: v for k, v in seen.items() if v.get("glossary_id") is None}

    # Optionally filter by domain
    all_rows = list(seen.values())
    if domain:
        all_rows = [r for r in all_rows if r.get("domain") == domain]

    terms = []
    for row in sorted(all_rows, key=lambda r: r["name"]):
        terms.append(GlossaryTermResponse(
            name=row["name"],
            display_name=row["display_name"],
            definition=row.get("definition"),
            domain=row.get("domain"),
            parent_id=row.get("parent_id"),
            aliases=row.get("aliases") or [],
            semantic_type=row.get("semantic_type"),
            cardinality=row.get("cardinality") or "many",
            status=row.get("status"),
            provenance=row.get("provenance"),
            glossary_status=row.get("glossary_status") or "self_describing",
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


@router.get("/{session_id}/glossary/{name}/relationships")
async def get_term_relationships(
    session_id: str,
    name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get SVO relationships for a glossary term."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)

    # Find entity by name
    entity = vs.find_entity_by_name(name, session_id=session_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")

    svo_rels = vs.get_relationships_for_entity(entity.id, session_id)

    # Also get co-occurrence suggestions
    from constat.discovery.glossary_generator import suggest_cooccurrence_relationships
    terms = vs.list_glossary_terms(session_id)
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

    # Run LLM glossary generation in background
    import threading

    def _run():
        session_manager._run_glossary_generation(session_id, managed)

    thread = threading.Thread(target=_run, name=f"glossary-gen-{session_id}", daemon=True)
    thread.start()

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


@router.post("/{session_id}/glossary/suggest-taxonomy")
async def suggest_taxonomy(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """LLM-assisted taxonomy suggestions for glossary terms."""
    managed = session_manager.get_session(session_id)
    vs = _get_vector_store(managed)
    session = managed.session

    if not hasattr(session, '_router') or not session._router:
        raise HTTPException(status_code=503, detail="LLM router not available")

    terms = vs.list_glossary_terms(session_id)
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
    result = session._router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=2048,
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


@router.patch("/{session_id}/glossary/bulk-status")
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
        ok = vs.update_glossary_term(name, session_id, {"status": request.status})
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


@router.post("/{session_id}/glossary/persist")
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

    # Get approved terms
    terms = vs.list_glossary_terms(session_id)
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
