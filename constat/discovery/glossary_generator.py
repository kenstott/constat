# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""LLM-powered glossary generation from extracted entities.

Selectively generates business definitions for entities where physical
metadata is insufficient (ambiguous, contested, cross-referenced, opaque).
"""

import hashlib
import json
import logging
from typing import Callable, Optional

from constat.core.models import TaskType
from constat.discovery.models import GlossaryTerm, display_entity_name

logger = logging.getLogger(__name__)

# Generic actions that never need definitions
SKIP_ACTIONS = {"get", "create", "update", "delete", "list", "set", "post", "put", "patch"}

GLOSSARY_SYSTEM_PROMPT = """You are building a business glossary from extracted entities. Not every entity
needs a glossary entry — most physical metadata is self-describing (email_address,
created_at, first_name). The glossary exists only for terms that NEED semantic
help: ambiguous, contested, cross-referenced, opaque, or policy-defined terms.

For each entity below, first decide: does this term need a glossary entry?
- YES if: the name is ambiguous, the concept is contested across teams, it appears
  across multiple systems, the name is opaque without context, or a policy defines
  it with specific meaning
- NO if: the name is self-describing, it only matters within one scope, or the
  physical metadata already communicates its meaning

For entities that need a glossary entry, write a business definition that describes
what this concept means — not where it is stored or what system it comes from.

Good definition: "An individual or organization with an active commercial
relationship, identified by account number and tracked through the customer
lifecycle."

Bad definition: "Customer data stored in the CRM PostgreSQL database."

Build multi-level hierarchies where appropriate. A term's parent can itself
have a parent. For example: "Base Salary" → "Compensation" → "Employee".
Go as deep as the domain warrants — do not flatten to a single level.

For each entity, also suggest:
- A parent category (if one exists among the other entities)
- Aliases (other names people use for this concept)
- Confidence (high/medium/low) in the definition

Respond as a JSON array (omit entities that don't need glossary entries):
[{
  "name": "...",
  "definition": "...",
  "parent": "..." or null,
  "aliases": ["..."],
  "confidence": "high|medium|low"
}]

If no entities need definitions, return an empty array: []"""


def _make_term_id(name: str, session_id: str, domain: str | None = None) -> str:
    """Generate a deterministic ID for a glossary term."""
    key = f"{name}:{session_id}:{domain or ''}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _is_candidate(entity_row: dict) -> bool:
    """Pre-filter: should this entity be sent to the LLM for elevation decision?"""
    name = entity_row.get("name", "").lower()
    semantic_type = entity_row.get("semantic_type", "")
    ner_type = entity_row.get("ner_type", "")
    ref_count = entity_row.get("ref_count", 0)
    source_count = entity_row.get("source_count", 1)

    # Skip generic actions
    if semantic_type == "action" and name in SKIP_ACTIONS:
        return False

    # Skip single-mention schema entities with self-describing names
    if ref_count <= 1 and ner_type == "SCHEMA" and source_count <= 1:
        if len(name.split()) <= 2 and "_" not in name:
            return False

    # CONCEPT/TERM types are likely candidates
    if semantic_type in ("concept", "term"):
        return True

    # ATTRIBUTEs with enough mentions
    if semantic_type == "attribute" and ref_count >= 3:
        return True

    # Multi-source entities (appear in DB + API, DB + document, etc.)
    if source_count > 1:
        return True

    # High mention count entities
    if ref_count >= 3:
        return True

    return False


def _build_entity_context(
    entity_name: str,
    session_id: str,
    vector_store,
) -> str:
    """Build context string for an entity (chunks + co-occurring entities)."""
    # Find entity in vector store
    entity = vector_store.find_entity_by_name(entity_name, session_id=session_id)
    if not entity:
        return f"Entity: {entity_name}\nNo additional context available."

    # Get top chunks
    chunks = vector_store.get_chunks_for_entity(entity.id, limit=3)
    lines = [f"Entity: {entity_name} (type: {entity.semantic_type})"]

    if chunks:
        lines.append("Referenced in:")
        for _chunk_id, chunk, _confidence in chunks:
            # Truncate long content
            content = chunk.content[:200].replace("\n", " ")
            lines.append(f"  [{chunk.document_name}] {content}")

    return "\n".join(lines)


def _batch_entities(candidates: list[dict], batch_size: int = 20) -> list[list[dict]]:
    """Split candidates into batches for LLM calls."""
    return [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]


def _parse_llm_response(content: str) -> list[dict]:
    """Parse the LLM's JSON response."""
    # Try to extract JSON array from response
    content = content.strip()
    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        content = content.strip()

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        # Try to find JSON array in the response
        start = content.find("[")
        end = content.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

    logger.warning(f"Failed to parse glossary LLM response: {content[:200]}")
    return []


def generate_glossary(
    session_id: str,
    vector_store,
    router,
    domain: str | None = None,
    on_batch_complete: Callable[[list[GlossaryTerm]], None] | None = None,
) -> list[GlossaryTerm]:
    """Generate glossary definitions for entities that need them.

    Uses LLM to selectively write business definitions for candidate entities
    where physical metadata is insufficient. The unified glossary view already
    shows all entities; this adds curated definitions to the glossary_terms table
    for candidates that pass the elevation test.

    Args:
        session_id: Session ID
        vector_store: DuckDBVectorStore instance
        router: TaskRouter for LLM calls
        domain: Optional domain scope
        on_batch_complete: Optional callback invoked per batch of stored terms

    Returns:
        List of generated GlossaryTerm objects
    """
    # Get all entities for this session with their reference counts
    rows = vector_store._conn.execute("""
        SELECT
            e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
            (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as ref_count,
            (SELECT COUNT(DISTINCT em.source) FROM chunk_entities ce
             JOIN embeddings em ON ce.chunk_id = em.chunk_id
             WHERE ce.entity_id = e.id) as source_count
        FROM entities e
        WHERE e.session_id = ?
        ORDER BY e.name
    """, [session_id]).fetchall()

    if not rows:
        logger.info(f"No entities found for session {session_id}, skipping glossary generation")
        return []

    # Build candidate list (pre-filter obvious non-candidates)
    candidates = []
    for row in rows:
        entity_id, name, display_name, semantic_type, ner_type, ref_count, source_count = row
        entity_data = {
            "id": entity_id,
            "name": name,
            "display_name": display_name,
            "semantic_type": semantic_type,
            "ner_type": ner_type,
            "ref_count": ref_count,
            "source_count": source_count,
        }
        if _is_candidate(entity_data):
            candidates.append(entity_data)

    if not candidates:
        logger.info(f"No candidates for glossary generation in session {session_id}")
        return []

    logger.info(f"Glossary generation: {len(candidates)} candidates from {len(rows)} entities")

    # Batch candidates and call LLM
    generated_terms: list[GlossaryTerm] = []
    batches = _batch_entities(candidates)

    for batch_idx, batch in enumerate(batches):
        # Build context for each candidate
        context_parts = []
        for candidate in batch:
            ctx = _build_entity_context(candidate["name"], session_id, vector_store)
            context_parts.append(ctx)

        user_message = "Entities:\n\n" + "\n\n---\n\n".join(context_parts)

        try:
            result = router.execute(
                task_type=TaskType.GLOSSARY_GENERATION,
                system=GLOSSARY_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=4096,
                complexity="medium",
            )

            if not result.success:
                logger.warning(f"Glossary generation batch {batch_idx} failed: {result.content}")
                continue

            parsed = _parse_llm_response(result.content)
            logger.info(f"Batch {batch_idx}: LLM returned {len(parsed)} definitions")

            # Convert to GlossaryTerm objects
            batch_terms: list[GlossaryTerm] = []
            for item in parsed:
                name = item.get("name", "").strip()
                definition = item.get("definition", "").strip()
                if not name or not definition:
                    continue

                # Find matching candidate for display_name
                matching = next(
                    (c for c in batch if c["name"].lower() == name.lower()),
                    None,
                )

                term = GlossaryTerm(
                    id=_make_term_id(name, session_id, domain),
                    name=name.lower(),
                    display_name=matching["display_name"] if matching else display_entity_name(name),
                    definition=definition,
                    domain=domain,
                    parent_id=None,  # Parent linking done separately
                    aliases=item.get("aliases", []),
                    semantic_type=matching["semantic_type"] if matching else "concept",
                    status="draft",
                    provenance="llm",
                    session_id=session_id,
                )

                # Store parent name for later linking
                parent_name = item.get("parent")
                if parent_name:
                    term.tags = {"_suggested_parent": {"name": parent_name}}

                batch_terms.append(term)

            # Store batch terms immediately
            for term in batch_terms:
                try:
                    vector_store.add_glossary_term(term)
                except Exception as e:
                    logger.warning(f"Failed to store glossary term '{term.name}': {e}")

            generated_terms.extend(batch_terms)

            if on_batch_complete and batch_terms:
                on_batch_complete(batch_terms)

        except Exception as e:
            logger.exception(f"Glossary generation batch {batch_idx} error: {e}")

    # Link parents (second pass)
    _link_parents(generated_terms, session_id, vector_store)

    logger.info(f"Glossary generation complete: {len(generated_terms)} terms for session {session_id}")
    return generated_terms


def _link_parents(
    terms: list[GlossaryTerm],
    session_id: str,
    vector_store,
) -> None:
    """Link parent_id fields based on suggested parent names."""
    # Build name -> id lookup
    name_to_id = {}
    for term in terms:
        name_to_id[term.name.lower()] = term.id

    for term in terms:
        suggested = term.tags.get("_suggested_parent", {})
        parent_name = suggested.get("name", "").lower() if suggested else ""
        if parent_name and parent_name in name_to_id:
            parent_id = name_to_id[parent_name]
            vector_store.update_glossary_term(
                term.name, session_id,
                {"parent_id": parent_id, "tags": {}},
            )


def resolve_physical_resources(
    term_name: str,
    session_id: str,
    vector_store,
    _visited: set | None = None,
) -> list[dict]:
    """Walk from glossary term to physical resources, pruning ungrounded paths.

    Handles three grounding paths:
    1. Direct: term name matches entities -> return their sources
    2. Collection: term.list_of -> resolve the element type
    3. Taxonomy: term.children -> resolve each child recursively
    """
    if _visited is None:
        _visited = set()

    # Prevent infinite recursion
    key = f"{term_name}:{session_id}"
    if key in _visited:
        return []
    _visited.add(key)

    # Direct match — concrete term
    entity = vector_store.find_entity_by_name(term_name, session_id=session_id)
    if entity:
        chunks = vector_store.get_chunks_for_entity(entity.id, limit=5)
        sources = []
        for _chunk_id, chunk, _confidence in chunks:
            sources.append({
                "document_name": chunk.document_name,
                "source": chunk.source,
                "section": chunk.section,
            })
        return [{
            "entity_name": entity.display_name,
            "entity_type": entity.semantic_type,
            "sources": sources,
        }]

    # No direct entities — try indirect paths
    term = vector_store.get_glossary_term(term_name, session_id)
    if not term:
        return []

    # Collection: follow list_of to element type
    if term.list_of:
        target = vector_store.get_glossary_term_by_id(term.list_of)
        if target:
            resources = resolve_physical_resources(target.name, session_id, vector_store, _visited)
            if resources:
                return resources

    # Taxonomy: follow children, collect their resources
    resources = []
    children = vector_store.get_child_terms(term.id)
    for child in children:
        child_resources = resolve_physical_resources(child.name, session_id, vector_store, _visited)
        resources.extend(child_resources)

    return resources


def is_grounded(
    term_name: str,
    session_id: str,
    vector_store,
    _visited: set | None = None,
) -> bool:
    """Check if a term is grounded in physical reality."""
    if _visited is None:
        _visited = set()

    key = f"{term_name}:{session_id}"
    if key in _visited:
        return False
    _visited.add(key)

    # Direct entity match
    entity = vector_store.find_entity_by_name(term_name, session_id=session_id)
    if entity:
        return True

    term = vector_store.get_glossary_term(term_name, session_id)
    if not term:
        return False

    # Collection: follow list_of
    if term.list_of:
        target = vector_store.get_glossary_term_by_id(term.list_of)
        if target and is_grounded(target.name, session_id, vector_store, _visited):
            return True

    # Taxonomy: check children
    children = vector_store.get_child_terms(term.id)
    return any(is_grounded(c.name, session_id, vector_store, _visited) for c in children)


def suggest_fk_relationships(
    session_id: str,
    glossary_terms: list[GlossaryTerm],
    schema_manager,
) -> list[dict]:
    """Generate relationship suggestions from foreign key constraints."""
    suggestions = []

    # Build name -> term lookup
    term_by_name = {t.name.lower(): t for t in glossary_terms}

    try:
        for _full_name, table_meta in schema_manager.metadata_cache.items():
            if not hasattr(table_meta, "foreign_keys"):
                continue
            for fk in table_meta.foreign_keys:
                source_table = table_meta.name.lower()
                target_table = fk.get("referred_table", "").lower() if isinstance(fk, dict) else ""

                source_term = term_by_name.get(source_table)
                target_term = term_by_name.get(target_table)

                if source_term and target_term:
                    suggestions.append({
                        "source": source_term.name,
                        "target": target_term.name,
                        "relationship": f"{source_term.display_name} has {target_term.display_name}",
                        "confidence": "high",
                        "provenance": "fk",
                    })
    except Exception as e:
        logger.warning(f"FK relationship suggestion failed: {e}")

    return suggestions


def suggest_cooccurrence_relationships(
    session_id: str,
    glossary_terms: list[GlossaryTerm],
    vector_store,
    min_cooccurrence: int = 3,
) -> list[dict]:
    """Suggest relationships from entity co-occurrence in chunks."""
    term_names = {t.name.lower() for t in glossary_terms}

    try:
        pairs = vector_store._conn.execute("""
            SELECT e1.name, e2.name, COUNT(*) as co_count
            FROM chunk_entities ce1
            JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id AND ce1.entity_id < ce2.entity_id
            JOIN entities e1 ON ce1.entity_id = e1.id
            JOIN entities e2 ON ce2.entity_id = e2.id
            WHERE e1.session_id = ? AND e2.session_id = ?
            GROUP BY e1.name, e2.name
            HAVING COUNT(*) >= ?
            ORDER BY co_count DESC
        """, [session_id, session_id, min_cooccurrence]).fetchall()
    except Exception as e:
        logger.warning(f"Co-occurrence query failed: {e}")
        return []

    suggestions = []
    for name_a, name_b, count in pairs:
        if name_a.lower() in term_names and name_b.lower() in term_names:
            suggestions.append({
                "source": name_a,
                "target": name_b,
                "relationship": None,
                "evidence": f"Co-occur in {count} chunks",
                "confidence": "medium",
                "provenance": "cooccurrence",
            })

    return suggestions
