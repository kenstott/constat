# Copyright (c) 2025 Kenneth Stott
# Canary: 459b942b-f58d-48a6-8400-a58e24c8315e
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL query and mutation resolvers for glossary and entity data."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Optional

import strawberry
from constat.server.graphql.session_context import GqlInfo as Info

from constat.server.entity_state import (
    _build_domain_maps,
    _resolve_entity_domain,
    _resolve_entity_domains,
    normalize_domain,
)
from strawberry.scalars import JSON

from constat.server.graphql.types import (
    ConnectedResourceSource,
    ConnectedResourceType,
    DraftAliasesType,
    DraftDefinitionType,
    DraftTagsType,
    EntityRelationshipType,
    GenerateResultType,
    GlossaryChangeAction,
    GlossaryChangeEvent,
    GlossaryChildType,
    GlossaryListType,
    GlossaryParentType,
    GlossaryTermInput,
    GlossaryTermType,
    GlossaryTermUpdateInput,
    RefineResultType,
    RenameResultType,
    TaxonomySuggestionType,
    TaxonomySuggestionsType,
)

logger = logging.getLogger(__name__)


def _get_managed(info: Info, session_id: str):
    """Get managed session from context, raising if not found."""
    sm = info.context.session_manager
    managed = sm.get_session_or_none(session_id)
    if not managed:
        raise ValueError(f"Session {session_id} not found")
    return managed


def _make_term_id(name: str, scope_id: str, domain: str | None = None) -> str:
    key = f"{name}:{scope_id}:{domain or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_vector_store(managed):
    """Get vector store from managed session."""
    session = managed.session
    if hasattr(session, "doc_tools") and session.doc_tools:
        vs = session.doc_tools._vector_store
        if vs:
            return vs
    raise ValueError("Vector store not available")


def _build_term_from_row(
    row: dict,
    user_id: str,
    domain_path_map: dict[str, str],
    source_to_domain: dict[str, str],
    vs,
) -> GlossaryTermType:
    """Build a GlossaryTermType from a unified glossary row."""
    effective_domain = row.get("domain")
    if not effective_domain:
        effective_domain = row.get("entity_domain_id")
    if not effective_domain:
        effective_domain = _resolve_entity_domain(
            row.get("entity_id"), vs, source_to_domain
        )
    effective_domain = normalize_domain(effective_domain, user_id, domain_path_map)
    domain_path = domain_path_map.get(effective_domain) if effective_domain else None
    tags = {
        k: v for k, v in (row.get("tags") or {}).items() if not k.startswith("_")
    }

    return GlossaryTermType(
        name=row["name"],
        display_name=row["display_name"],
        definition=row.get("definition"),
        domain=effective_domain,
        domain_path=domain_path,
        parent_id=row.get("parent_id"),
        parent_verb=row.get("parent_verb") or "HAS_KIND",
        aliases=row.get("aliases") or [],
        semantic_type=row.get("semantic_type"),
        ner_type=row.get("ner_type"),
        cardinality=row.get("cardinality") or "many",
        status=row.get("status"),
        provenance=row.get("provenance"),
        glossary_status=row.get("glossary_status") or "self_describing",
        entity_id=row.get("entity_id"),
        glossary_id=row.get("glossary_id"),
        tags=tags or None,
        ignored=row.get("ignored") or False,
        canonical_source=row.get("canonical_source"),
    )


@strawberry.type
class Query:
    @strawberry.field
    async def glossary(
        self,
        info: Info,
        session_id: str,
        scope: str = "all",
        domain: Optional[str] = None,
    ) -> GlossaryListType:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        active_domains = getattr(managed, "active_domains", []) or []
        rows = vs.get_unified_glossary(
            session_id, scope=scope, active_domains=active_domains, user_id=managed.user_id
        )

        config = managed.session.config
        domain_path_map, source_to_domain = _build_domain_maps(config, managed.session)
        user_id = managed.user_id

        # Auto-prune deprecated drafts (status=draft, no grounded entity)
        session_active_domains = (
            managed.session.active_domains
            if hasattr(managed.session, "active_domains")
            else None
        )
        keep_rows: list[dict] = []
        pruned = 0
        for row in rows:
            if (
                row.get("status") == "draft"
                and not row.get("entity_id")
                and not vs.entity_exists(row["name"], session_id)
            ):
                try:
                    vs.delete_glossary_term_cascade(
                        row["name"],
                        session_id,
                        session_active_domains,
                        user_id=managed.user_id,
                        domain=row.get("domain"),
                    )
                    pruned += 1
                except Exception:
                    logger.warning("Failed to prune deprecated draft: %s", row["name"], exc_info=True)
                    keep_rows.append(row)
            else:
                keep_rows.append(row)
        rows = keep_rows
        # Single event for all pruned terms (query result already reflects pruned state)
        if pruned:
            _publish(info, session_id, GlossaryChangeAction.DELETED, "auto-prune")

        # Pre-resolve effective domain per row for filtering
        for row in rows:
            d = row.get("domain")
            if not d:
                d = row.get("entity_domain_id")
            if not d:
                d = _resolve_entity_domain(row.get("entity_id"), vs, source_to_domain)
            row["_effective_domain"] = normalize_domain(d, user_id, domain_path_map)

        # Filter by domain (comma-separated)
        if domain:
            domain_set = set(domain.split(","))
            include_system = "system" in domain_set
            include_cross = "cross-domain" in domain_set
            explicit_domains: set[str] = set()
            for d in domain_set - {"system", "cross-domain"}:
                explicit_domains.add(d)
                explicit_domains.add(d.removesuffix(".yaml"))
                if not d.endswith(".yaml"):
                    explicit_domains.add(d + ".yaml")

            def _matches(r: dict) -> bool:
                eff = r.get("_effective_domain")
                if eff == "cross-domain":
                    return include_cross
                if include_system and eff == "__base__":
                    return True
                if eff in explicit_domains:
                    return True
                return False

            rows = [r for r in rows if _matches(r)]

        terms = [
            _build_term_from_row(row, user_id, domain_path_map, source_to_domain, vs)
            for row in sorted(rows, key=lambda r: r["name"])
        ]

        defined = sum(1 for t in terms if t.glossary_status == "defined")
        self_describing = sum(1 for t in terms if t.glossary_status == "self_describing")

        return GlossaryListType(
            terms=terms,
            total_defined=defined,
            total_self_describing=self_describing,
        )

    @strawberry.field
    async def glossary_term(
        self,
        info: Info,
        session_id: str,
        name: str,
        domain: Optional[str] = None,
    ) -> Optional[GlossaryTermType]:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        term = vs.get_glossary_term(name, session_id, user_id=managed.user_id, domain=domain)
        active_domains = getattr(managed, "active_domains", []) or []

        config = managed.session.config
        domain_path_map, source_to_domain = _build_domain_maps(config, managed.session)
        user_id = managed.user_id

        # Resolve physical resources
        from constat.discovery.glossary_generator import resolve_physical_resources

        doc_tools = getattr(managed.session, "doc_tools", None)
        resources = resolve_physical_resources(
            name, session_id, vs, domain_ids=active_domains,
            user_id=managed.user_id, doc_tools=doc_tools,
        )
        connected = [
            ConnectedResourceType(
                entity_name=r["entity_name"],
                entity_type=r["entity_type"],
                sources=[
                    ConnectedResourceSource(
                        document_name=s["document_name"],
                        source=s["source"],
                        section=s.get("section"),
                        url=s.get("url"),
                    )
                    for s in r.get("sources", [])
                ],
            )
            for r in resources
        ]

        # Resolve parent
        parent_info = None
        if term and term.parent_id:
            parent_term = vs.get_glossary_term_by_id(term.parent_id)
            if parent_term:
                parent_info = GlossaryParentType(name=parent_term.name, display_name=parent_term.display_name)
            else:
                parent_entity = vs.get_entity_by_id(term.parent_id)
                if parent_entity:
                    parent_info = GlossaryParentType(name=parent_entity.name, display_name=parent_entity.display_name)

        # Resolve children
        children = []
        lookup_name = term.name if term else name
        entity = vs.find_entity_by_name(
            lookup_name, domain_ids=active_domains, session_id=session_id, cross_session=True,
        )
        candidate_ids = []
        if term:
            candidate_ids.append(term.id)
        if entity:
            candidate_ids.append(entity.id)
        if candidate_ids:
            child_terms = vs.get_child_terms(candidate_ids[0], *candidate_ids[1:])
            children = [
                GlossaryChildType(name=c.name, display_name=c.display_name, parent_verb=c.parent_verb)
                for c in child_terms
            ]

        # Resolve relationships
        rels = vs.get_relationships_for_entity(lookup_name, session_id)
        relationships = [
            EntityRelationshipType(
                id=r["id"],
                subject=r["subject_name"],
                verb=r["verb"],
                object=r["object_name"],
                confidence=r["confidence"],
                user_edited=r.get("user_edited", False),
            )
            for r in rels
        ]

        # Resolve cluster siblings
        cluster_siblings: list[str] = []
        if hasattr(vs, "get_cluster_siblings"):
            try:
                cluster_siblings = vs.get_cluster_siblings(lookup_name, session_id)
            except Exception:
                logger.warning(f"glossary_term({name}): cluster siblings lookup failed", exc_info=True)

        # Build result from glossary term or entity
        if term:
            effective_domain = normalize_domain(
                term.domain
                or (entity.domain_id if entity else None)
                or _resolve_entity_domain(entity.id if entity else None, vs, source_to_domain),
                user_id, domain_path_map,
            )
            spanning = None
            if effective_domain == "cross-domain":
                spanning = [
                    normalize_domain(d, user_id, domain_path_map) or d
                    for d in _resolve_entity_domains(entity.id if entity else None, vs, source_to_domain)
                ]
            tags = {k: v for k, v in (term.tags or {}).items() if not k.startswith("_")}
            return GlossaryTermType(
                name=term.name,
                display_name=term.display_name,
                definition=term.definition,
                domain=effective_domain,
                domain_path=domain_path_map.get(effective_domain) if effective_domain else None,
                parent_id=term.parent_id,
                parent_verb=term.parent_verb or "HAS_KIND",
                parent=parent_info,
                aliases=term.aliases or [],
                semantic_type=term.semantic_type,
                cardinality=term.cardinality or "many",
                status=term.status,
                provenance=term.provenance,
                glossary_status="defined",
                entity_id=entity.id if entity else None,
                glossary_id=term.id if term else None,
                tags=tags or None,
                ignored=term.ignored or False,
                canonical_source=term.canonical_source,
                connected_resources=connected,
                spanning_domains=spanning,
                children=children or None,
                relationships=relationships or None,
                cluster_siblings=cluster_siblings or None,
            )

        # No glossary term — check entity (self-describing)
        if entity:
            effective_domain = normalize_domain(
                entity.domain_id or _resolve_entity_domain(entity.id, vs, source_to_domain),
                user_id, domain_path_map,
            )
            spanning = None
            if effective_domain == "cross-domain":
                spanning = [
                    normalize_domain(d, user_id, domain_path_map) or d
                    for d in _resolve_entity_domains(entity.id, vs, source_to_domain)
                ]
            return GlossaryTermType(
                name=entity.name,
                display_name=entity.display_name,
                domain=effective_domain,
                domain_path=domain_path_map.get(effective_domain) if effective_domain else None,
                semantic_type=entity.semantic_type,
                glossary_status="self_describing",
                entity_id=entity.id,
                connected_resources=connected,
                spanning_domains=spanning,
                children=children or None,
                relationships=relationships or None,
                cluster_siblings=cluster_siblings or None,
            )

        return None


def _publish(info: Info, session_id: str, action: GlossaryChangeAction, term_name: str, term: GlossaryTermType | None = None):
    sm = info.context.session_manager
    event = GlossaryChangeEvent(session_id=session_id, action=action, term_name=term_name, term=term)
    sm.publish_glossary_change(session_id, event)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_glossary_term(
        self, info: Info, session_id: str, input: GlossaryTermInput
    ) -> GlossaryTermType:
        from constat.discovery.models import GlossaryTerm, display_entity_name

        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        existing = vs.get_glossary_term(input.name, session_id, user_id=managed.user_id, domain=input.domain)
        if existing:
            raise ValueError(f"Term '{input.name}' already has a definition. Use updateGlossaryTerm to update.")

        if not input.parent_id and not input.is_abstract:
            if not vs.entity_exists(input.name, session_id):
                raise ValueError("Abstract term must have parent_id or match a grounded entity")

        # New terms are always drafts in the user's personal domain.
        # Promotion to a shared domain happens via update after review.
        draft_domain = managed.user_id

        term = GlossaryTerm(
            id=_make_term_id(input.name, managed.user_id, draft_domain),
            name=input.name.lower(),
            display_name=display_entity_name(input.name),
            definition=input.definition,
            domain=draft_domain,
            parent_id=input.parent_id,
            aliases=input.aliases,
            semantic_type=input.semantic_type or "concept",
            status="draft",
            provenance="human",
            session_id=session_id,
            user_id=managed.user_id,
        )
        vs.add_glossary_term(term)

        result = GlossaryTermType(
            name=term.name,
            display_name=term.display_name,
            definition=term.definition,
            domain=draft_domain,
            parent_id=term.parent_id,
            aliases=term.aliases or [],
            semantic_type=term.semantic_type,
            status="draft",
            provenance="human",
            glossary_status="defined",
            glossary_id=term.id,
        )
        _publish(info, session_id, GlossaryChangeAction.CREATED, term.name, result)
        return result

    @strawberry.mutation
    async def update_glossary_term(
        self, info: Info, session_id: str, name: str, input: GlossaryTermUpdateInput,
        domain: Optional[str] = None,
    ) -> GlossaryTermType:
        from constat.discovery.models import GlossaryTerm, display_entity_name

        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id, domain=domain)
        if not existing:
            active_domains = getattr(managed, "active_domains", []) or []
            entity = vs.find_entity_by_name(name, domain_ids=active_domains, session_id=session_id, cross_session=True)
            if not entity:
                raise ValueError(f"Term '{name}' not found")
            draft_domain = managed.user_id
            companion = GlossaryTerm(
                id=_make_term_id(name, managed.user_id, draft_domain),
                name=name.lower(),
                display_name=display_entity_name(name),
                definition="",
                domain=draft_domain,
                semantic_type=entity.semantic_type,
                status="draft",
                provenance="human",
                session_id=session_id,
                user_id=managed.user_id,
            )
            vs.add_glossary_term(companion)
            existing = companion

        updates = {}
        if input.definition is not None:
            updates["definition"] = input.definition
        if input.aliases is not None:
            updates["aliases"] = input.aliases
        if input.parent_id is not None:
            updates["parent_id"] = input.parent_id
        if input.parent_verb is not None:
            updates["parent_verb"] = input.parent_verb
        if input.status is not None:
            updates["status"] = input.status
        if input.domain is not None:
            updates["domain"] = input.domain or None
        if input.semantic_type is not None:
            updates["semantic_type"] = input.semantic_type
        if input.tags is not None:
            updates["tags"] = input.tags
        if input.ignored is not None:
            updates["ignored"] = input.ignored
        if input.canonical_source is not None:
            updates["canonical_source"] = input.canonical_source

        if existing.provenance == "llm" and updates:
            updates["provenance"] = "hybrid"

        if not updates:
            raise ValueError("No fields to update")

        vs.update_glossary_term(name, session_id, updates, user_id=managed.user_id, domain=domain)

        # Build response from existing + updates (avoids refetch that can cause
        # DuckDB write-write conflicts on rapid sequential updates)
        import json as _json
        config = managed.session.config
        domain_path_map, source_to_domain = _build_domain_maps(config, managed.session)
        merged_aliases = updates.get("aliases", existing.aliases) or []
        if isinstance(merged_aliases, str):
            merged_aliases = _json.loads(merged_aliases)
        merged_tags = updates.get("tags", existing.tags)
        if isinstance(merged_tags, str):
            merged_tags = _json.loads(merged_tags)
        row = {
            "entity_id": None, "name": existing.name,
            "display_name": updates.get("display_name", existing.display_name),
            "semantic_type": updates.get("semantic_type", existing.semantic_type),
            "ner_type": None, "session_id": session_id,
            "glossary_id": existing.id, "domain": updates.get("domain", existing.domain),
            "definition": updates.get("definition", existing.definition),
            "parent_id": updates.get("parent_id", existing.parent_id),
            "parent_verb": updates.get("parent_verb", existing.parent_verb),
            "aliases": merged_aliases,
            "cardinality": updates.get("cardinality", existing.cardinality),
            "plural": updates.get("plural", existing.plural),
            "status": updates.get("status", existing.status),
            "provenance": updates.get("provenance", existing.provenance),
            "glossary_status": "defined", "entity_domain_id": None,
            "ignored": updates.get("ignored", existing.ignored),
            "canonical_source": updates.get("canonical_source", existing.canonical_source),
            "tags": merged_tags,
        }
        term_type = _build_term_from_row(row, managed.user_id, domain_path_map, source_to_domain, vs)
        _publish(info, session_id, GlossaryChangeAction.UPDATED, name, term_type)
        return term_type

    @strawberry.mutation
    async def delete_glossary_term(
        self, info: Info, session_id: str, name: str,
        domain: Optional[str] = None,
    ) -> bool:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        sm = info.context.session_manager

        existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id, domain=domain)
        if not existing:
            return False

        active_domains = managed.session.active_domains if hasattr(managed.session, "active_domains") else None
        result = vs.delete_glossary_term_cascade(
            name, session_id, active_domains, user_id=managed.user_id, domain=domain,
        )
        if not result:
            raise ValueError(f"Term '{name}' not found for delete")

        sm.write_config_tombstone(session_id, "glossary", name)
        _publish(info, session_id, GlossaryChangeAction.DELETED, name)
        return True

    @strawberry.mutation
    async def delete_glossary_terms(
        self, info: Info, session_id: str, names: list[str],
    ) -> int:
        """Batch delete glossary terms, publishing a delta event per deletion."""
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        sm = info.context.session_manager
        active_domains = managed.session.active_domains if hasattr(managed.session, "active_domains") else None

        deleted = 0
        for name in names:
            existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
            if not existing:
                continue

            result = vs.delete_glossary_term_cascade(
                name, session_id, active_domains, user_id=managed.user_id, domain=existing.domain,
            )
            if not result:
                continue

            sm.write_config_tombstone(session_id, "glossary", name)
            deleted += 1

        # Single subscription event for the entire batch
        if deleted:
            _publish(info, session_id, GlossaryChangeAction.DELETED, names[0])

        return deleted

    @strawberry.mutation
    async def create_relationship(
        self, info: Info, session_id: str, subject: str, verb: str, object: str
    ) -> EntityRelationshipType:
        from constat.discovery.models import EntityRelationship
        from constat.discovery.relationship_extractor import categorize_verb

        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        verb = verb.strip().upper().replace(" ", "_").replace("-", "_")

        rel = EntityRelationship(
            id=str(uuid.uuid4()),
            subject_name=subject.lower(),
            verb=verb,
            object_name=object.lower(),
            confidence=1.0,
            verb_category=categorize_verb(verb.lower()),
            session_id=session_id,
            user_edited=True,
        )
        vs.add_entity_relationship(rel)

        return EntityRelationshipType(
            id=rel.id,
            subject=subject,
            verb=verb,
            object=object,
            confidence=1.0,
            user_edited=True,
        )

    @strawberry.mutation
    async def update_relationship(
        self, info: Info, session_id: str, rel_id: str, verb: str
    ) -> EntityRelationshipType:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        verb = verb.strip().upper().replace(" ", "_").replace("-", "_")
        updated = vs.update_entity_relationship_verb(rel_id, verb)
        if not updated:
            raise ValueError(f"Relationship '{rel_id}' not found")

        # Return with updated verb; subject/object from the store
        store = vs._relational
        rels = store.list_session_relationships(session_id)
        for r in rels:
            if r["id"] == rel_id:
                return EntityRelationshipType(
                    id=rel_id,
                    subject=r["subject_name"],
                    verb=verb,
                    object=r["object_name"],
                    confidence=r["confidence"],
                    user_edited=r.get("user_edited", False),
                )
        raise ValueError(f"Relationship '{rel_id}' not found after update")

    @strawberry.mutation
    async def delete_relationship(
        self, info: Info, session_id: str, rel_id: str
    ) -> bool:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        deleted = vs.delete_entity_relationship(rel_id)
        if not deleted:
            raise ValueError(f"Relationship '{rel_id}' not found")
        return True

    @strawberry.mutation
    async def approve_relationship(
        self, info: Info, session_id: str, rel_id: str
    ) -> EntityRelationshipType:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        if not vs.mark_relationship_user_edited(rel_id):
            raise ValueError(f"Relationship '{rel_id}' not found")

        store = vs._relational
        rels = store.list_session_relationships(session_id)
        for r in rels:
            if r["id"] == rel_id:
                return EntityRelationshipType(
                    id=rel_id,
                    subject=r["subject_name"],
                    verb=r["verb"],
                    object=r["object_name"],
                    confidence=r["confidence"],
                    user_edited=True,
                )
        raise ValueError(f"Relationship '{rel_id}' not found after approval")

    @strawberry.mutation
    async def bulk_update_status(
        self, info: Info, session_id: str, names: list[str], new_status: str
    ) -> int:
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)

        count = 0
        for name in names:
            term = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
            ok = vs.update_glossary_term(name, session_id, {"status": new_status}, user_id=managed.user_id, domain=term.domain if term else None)
            if ok:
                count += 1
        return count

    @strawberry.mutation
    async def generate_glossary(
        self, info: Info, session_id: str, phases: Optional[JSON] = None,
    ) -> GenerateResultType:
        from constat.server.routes.data.glossary import generate_glossary_op
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager
        result = await generate_glossary_op(session_id, managed, sm, phases=phases)
        return GenerateResultType(status=result["status"], message=result["message"])

    @strawberry.mutation
    async def draft_definition(
        self, info: Info, session_id: str, name: str,
    ) -> DraftDefinitionType:
        from constat.server.routes.data.glossary import draft_definition_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await draft_definition_op(session_id, managed, vs, name)
        return DraftDefinitionType(name=result["name"], draft=result["draft"])

    @strawberry.mutation
    async def draft_aliases(
        self, info: Info, session_id: str, name: str,
    ) -> DraftAliasesType:
        from constat.server.routes.data.glossary import draft_aliases_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await draft_aliases_op(session_id, managed, vs, name)
        return DraftAliasesType(name=result["name"], aliases=result["aliases"])

    @strawberry.mutation
    async def draft_tags(
        self, info: Info, session_id: str, name: str,
    ) -> DraftTagsType:
        from constat.server.routes.data.glossary import draft_tags_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await draft_tags_op(session_id, managed, vs, name)
        return DraftTagsType(name=result["name"], tags=result["tags"])

    @strawberry.mutation
    async def refine_definition(
        self, info: Info, session_id: str, name: str,
    ) -> RefineResultType:
        from constat.server.routes.data.glossary import refine_definition_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await refine_definition_op(session_id, managed, vs, name)
        # Publish update event since definition was persisted
        config = managed.session.config
        domain_path_map, source_to_domain = _build_domain_maps(config, managed.session)
        term = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
        if term:
            row = {
                "name": term.name, "display_name": term.display_name,
                "definition": result["after"], "domain": term.domain,
                "parent_id": term.parent_id, "parent_verb": term.parent_verb,
                "aliases": term.aliases, "semantic_type": term.semantic_type,
                "status": term.status, "provenance": term.provenance,
                "glossary_status": "defined", "glossary_id": term.id,
                "entity_id": None, "entity_domain_id": None,
                "cardinality": term.cardinality, "ignored": term.ignored,
                "canonical_source": term.canonical_source, "tags": term.tags,
                "ner_type": None,
            }
            term_type = _build_term_from_row(row, managed.user_id, domain_path_map, source_to_domain, vs)
            _publish(info, session_id, GlossaryChangeAction.UPDATED, name, term_type)
        return RefineResultType(name=result["name"], before=result["before"], after=result["after"])

    @strawberry.mutation
    async def suggest_taxonomy(
        self, info: Info, session_id: str,
    ) -> TaxonomySuggestionsType:
        from constat.server.routes.data.glossary import suggest_taxonomy_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await suggest_taxonomy_op(session_id, managed, vs)
        suggestions = [
            TaxonomySuggestionType(
                child=s["child"], parent=s["parent"],
                parent_verb=s.get("parent_verb", "HAS_KIND"),
                confidence=s["confidence"], reason=s["reason"],
            )
            for s in result.get("suggestions", [])
        ]
        return TaxonomySuggestionsType(
            suggestions=suggestions,
            message=result.get("message"),
        )

    @strawberry.mutation
    async def rename_term(
        self, info: Info, session_id: str, name: str, new_name: str,
    ) -> RenameResultType:
        from constat.server.routes.data.glossary import rename_term_op
        managed = _get_managed(info, session_id)
        vs = _get_vector_store(managed)
        result = await rename_term_op(session_id, managed, vs, name, new_name)
        _publish(info, session_id, GlossaryChangeAction.UPDATED, result["new_name"])
        return RenameResultType(
            old_name=result["old_name"],
            new_name=result["new_name"],
            display_name=result["display_name"],
            relationships_updated=result["relationships_updated"],
        )
