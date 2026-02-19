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
import time
from typing import Callable, Optional

from constat.core.models import TaskType
from constat.discovery.models import GlossaryTerm, display_entity_name

logger = logging.getLogger(__name__)


def _retry_on_conflict(fn, max_retries=3):
    """Retry a DuckDB operation on write-write conflict."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "write-write conflict" in str(e) and attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
                continue
            raise

# Generic actions that never need definitions
SKIP_ACTIONS = {"get", "create", "update", "delete", "list", "set", "post", "put", "patch"}

GLOSSARY_SYSTEM_PROMPT = """You are building a business glossary from extracted entities.
Write a business definition for each entity below. Definitions describe what the
concept means in business terms — not where it is stored or what system it comes from.

Skip ONLY trivially self-describing names where a definition adds no value
(e.g., first_name, email_address, created_at, phone_number). Define everything
else — even seemingly simple terms like "Currency", "Country", or "Status" benefit
from a definition that captures the business meaning.

Good: "An individual or organization with an active commercial relationship,
identified by account number and tracked through the customer lifecycle."
Bad: "Customer data stored in the CRM PostgreSQL database."

Build multi-level hierarchies where appropriate. A term's parent can itself
have a parent. For example: "Base Salary" → "Compensation" → "Employee".
Go as deep as the domain warrants — do not flatten to a single level.

For each entity, also provide:
- A parent category — either an existing entity from the list, or a new grouping
  term you create (e.g., "Customer Tier" as parent of "Platinum", "Gold", "Bronze").
  Include any new parent terms as entries in the output with their own definitions.
  Every term must ultimately ground to a physical entity through the hierarchy.
- Aliases (other names people use for this concept)
- Confidence (high/medium/low) in the definition

IMPORTANT: When you create a new parent grouping term, you MUST also include entries
for ALL its children — even if those children seem self-describing. Without the child
entries, the parent-child link cannot be established. For self-describing children,
keep the definition brief (e.g., "A premium customer tier level").

Respond as a JSON array with one entry per entity (plus any new parent terms):
[{
  "name": "...",
  "definition": "...",
  "parent": "..." or null,
  "aliases": ["..."],
  "confidence": "high|medium|low"
}]"""


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

    # Skip dot-notation schema paths (e.g., "Countries.Language.Rtl")
    if "." in name:
        return False

    # Skip colon-separated labels (NER extraction artifacts)
    if ":" in name:
        return False

    # Skip hex-like names (auto-generated IDs, hashes)
    clean = name.replace("-", "").replace("_", "")
    if len(clean) >= 8 and all(c in "0123456789abcdef" for c in clean):
        return False

    # Skip generic actions
    if semantic_type == "action" and name in SKIP_ACTIONS:
        return False

    # Skip generic actions
    if semantic_type == "action" and name in SKIP_ACTIONS:
        return False

    # Everything else is a candidate — the LLM decides
    return True


def _build_entity_context(
    entity_name: str,
    session_id: str,
    vector_store,
    domain_ids: list[str] | None = None,
) -> str:
    """Build context string for an entity (chunks + co-occurring entities)."""
    entity = vector_store.find_entity_by_name(
        entity_name, domain_ids=domain_ids, session_id=session_id,
    )
    if not entity:
        return f"Entity: {entity_name}\nNo additional context available."

    # Get top chunks
    chunks = vector_store.get_chunks_for_entity(entity.id, domain_ids=domain_ids, limit=3)
    lines = [f"Entity: {entity_name} (type: {entity.semantic_type})"]

    if chunks:
        lines.append("Referenced in:")
        for _chunk_id, chunk, _confidence in chunks:
            # Truncate long content
            content = chunk.content[:200].replace("\n", " ")
            lines.append(f"  [{chunk.document_name}] {content}")

    return "\n".join(lines)


def _batch_entities(candidates: list[dict], batch_size: int = 10) -> list[list[dict]]:
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
    active_domains: list[str] | None = None,
    on_batch_complete: Callable[[list[GlossaryTerm]], None] | None = None,
    on_progress: Callable[[str, int], None] | None = None,
    user_id: str | None = None,
) -> list[GlossaryTerm]:
    """Generate glossary definitions for entities that need them.

    Uses LLM to selectively write business definitions for candidate entities
    where physical metadata is insufficient. The unified glossary view already
    shows all entities; this adds curated definitions to the glossary_terms table
    for candidates that pass the elevation test.

    Args:
        session_id: Session ID (for entity visibility)
        vector_store: DuckDBVectorStore instance
        router: TaskRouter for LLM calls
        domain: Optional domain scope
        active_domains: Active domain IDs for entity visibility
        on_batch_complete: Optional callback invoked per batch of stored terms
        on_progress: Optional callback(stage, percent) for progress reporting
        user_id: User ID for glossary term ownership (user-scoped)

    Returns:
        List of generated GlossaryTerm objects
    """
    from constat.discovery.vector_store import DuckDBVectorStore

    if on_progress:
        on_progress("Collecting entities", 0)

    entity_where, params = DuckDBVectorStore.entity_visibility_filter(
        session_id, active_domains, alias="e",
    )

    rows = vector_store._conn.execute(f"""
        SELECT
            e.id, e.name, e.display_name, e.semantic_type, e.ner_type,
            (SELECT COUNT(*) FROM chunk_entities ce WHERE ce.entity_id = e.id) as ref_count,
            (SELECT COUNT(DISTINCT em.source) FROM chunk_entities ce
             JOIN embeddings em ON ce.chunk_id = em.chunk_id
             WHERE ce.entity_id = e.id) as source_count
        FROM entities e
        WHERE {entity_where}
        ORDER BY e.name
    """, params).fetchall()

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

    # Track claimed aliases across all batches to prevent duplicates
    # An alias cannot be: another term's primary name, or claimed by another term
    claimed_aliases: set[str] = set()  # all aliases assigned so far
    all_term_names: set[str] = {r[1].lower() for r in rows}  # all entity names

    # Also include existing glossary term names and their aliases
    existing_terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    for et in existing_terms:
        all_term_names.add(et.name.lower())
        for a in (et.aliases or []):
            if a:
                claimed_aliases.add(a.strip().lower())

    for batch_idx, batch in enumerate(batches):
        if on_progress:
            pct = 5 + int((batch_idx + 1) / len(batches) * 65)
            on_progress("Generating definitions", pct)

        # Build context for each candidate
        context_parts = []
        for candidate in batch:
            ctx = _build_entity_context(candidate["name"], session_id, vector_store, domain_ids=active_domains)
            context_parts.append(ctx)

        user_message = "Entities:\n\n" + "\n\n---\n\n".join(context_parts)

        try:
            result = router.execute(
                task_type=TaskType.GLOSSARY_GENERATION,
                system=GLOSSARY_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=router.max_output_tokens,
                complexity="medium",
            )

            if not result.success:
                logger.warning(f"Glossary generation batch {batch_idx} failed: {result.content}")
                continue

            parsed = _parse_llm_response(result.content)
            logger.info(f"Batch {batch_idx}: LLM returned {len(parsed)} definitions")

            # Grounding check: only create terms that are grounded to a
            # real entity OR are transitively grounded (ancestor of a
            # grounded term through the parent hierarchy).
            grounded_names: set[str] = set()
            parent_refs: dict[str, str] = {}  # child_name -> parent_name
            for item in parsed:
                name = item.get("name", "").strip().lower()
                if not name:
                    continue
                if any(c["name"].lower() == name for c in candidates):
                    grounded_names.add(name)
                parent_name = item.get("parent", "")
                if parent_name:
                    parent_refs[name] = parent_name.strip().lower()

            # Walk parent chains upward from grounded terms to find
            # transitively grounded parents (multi-level hierarchies).
            needed_parents: set[str] = set()
            frontier = set(grounded_names)
            while frontier:
                next_frontier: set[str] = set()
                for child in frontier:
                    parent = parent_refs.get(child)
                    if parent and parent not in grounded_names and parent not in needed_parents:
                        needed_parents.add(parent)
                        next_frontier.add(parent)
                frontier = next_frontier

            batch_terms: list[GlossaryTerm] = []
            for item in parsed:
                name = item.get("name", "").strip()
                definition = item.get("definition", "").strip()
                if not name or not definition:
                    continue

                name_lower = name.lower()

                # Skip terms not grounded or transitively grounded
                if name_lower not in grounded_names and name_lower not in needed_parents:
                    logger.debug(f"Skipping ungrounded LLM term: {name}")
                    continue

                matching = next(
                    (c for c in candidates if c["name"].lower() == name_lower),
                    None,
                )

                # Deduplicate aliases: remove self-references, other term names, already-claimed
                raw_aliases = item.get("aliases", [])
                deduped_aliases = []
                for a in raw_aliases:
                    if not a:
                        continue
                    a_lower = a.strip().lower()
                    if a_lower == name_lower:
                        continue  # Self-reference
                    if a_lower in all_term_names:
                        continue  # Another term's primary name
                    if a_lower in claimed_aliases:
                        continue  # Already claimed by another term
                    deduped_aliases.append(a.strip())
                    claimed_aliases.add(a_lower)

                all_term_names.add(name_lower)

                scope_id = user_id or session_id
                term = GlossaryTerm(
                    id=_make_term_id(name, scope_id, domain),
                    name=name_lower,
                    display_name=matching["display_name"] if matching else display_entity_name(name),
                    definition=definition,
                    domain=domain,
                    parent_id=None,  # Parent linking done separately
                    aliases=deduped_aliases,
                    semantic_type=matching["semantic_type"] if matching else "concept",
                    status="draft",
                    provenance="llm",
                    session_id=session_id,
                    user_id=scope_id,
                )

                # Store parent name for later linking
                parent_name = item.get("parent")
                if parent_name:
                    term.tags = {"_suggested_parent": {"name": parent_name}}

                batch_terms.append(term)

            # Store batch terms immediately
            for term in batch_terms:
                try:
                    _retry_on_conflict(lambda t=term: vector_store.add_glossary_term(t))
                except Exception as e:
                    logger.warning(f"Failed to store glossary term '{term.name}': {e}")

            generated_terms.extend(batch_terms)

            if on_batch_complete and batch_terms:
                on_batch_complete(batch_terms)

        except Exception as e:
            logger.exception(f"Glossary generation batch {batch_idx} error: {e}")

    # Link parents (second pass)
    if on_progress:
        on_progress("Linking hierarchy", 72)
    _link_parents(generated_terms, session_id, vector_store, active_domains, domain, user_id=user_id)

    # Prune ungrounded terms — LLM-created parents that failed to link children
    if on_progress:
        on_progress("Pruning ungrounded", 75)
    pruned = _prune_ungrounded(session_id, vector_store, user_id=user_id)
    if pruned:
        generated_terms = [t for t in generated_terms if t.name not in pruned]

    logger.info(f"Glossary generation complete: {len(generated_terms)} terms for session {session_id}")
    return generated_terms


def _prune_ungrounded(session_id: str, vector_store, *, user_id: str | None = None) -> set[str]:
    """Delete LLM-generated glossary terms that aren't grounded.

    A term is grounded if it has a matching visible entity with physical
    document chunks. Abstract parent terms are kept only if they are
    ancestors of a grounded term — every branch must terminate at a
    grounded leaf.

    Returns:
        Set of pruned term names
    """
    terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    if not terms:
        return set()

    by_id: dict[str, GlossaryTerm] = {t.id: t for t in terms}

    # Find terms directly grounded to a visible entity with physical chunks
    grounded: set[str] = set()
    for t in terms:
        entity = vector_store.find_entity_by_name(t.name, session_id=session_id)
        if entity and _entity_has_physical_chunks(entity, vector_store):
            grounded.add(t.id)

    # A term is valid if:
    # - It is directly grounded (has entity with physical chunks), OR
    # - It is an ancestor of a grounded term (abstract grouping parent)
    # Key rule: every branch must terminate at a grounded leaf.
    # Abstract parents with NO grounded descendants are pruned.

    # Walk upward from grounded terms — mark ancestors valid
    valid = set(grounded)
    for tid in grounded:
        current = by_id.get(tid)
        while current and current.parent_id:
            parent = by_id.get(current.parent_id)
            if not parent or parent.id in valid:
                break
            valid.add(parent.id)
            current = parent

    # Prune LLM-generated terms that aren't valid
    pruned: set[str] = set()
    for term in terms:
        if term.provenance != "llm":
            continue
        if term.id in valid:
            continue

        try:
            _retry_on_conflict(lambda n=term.name: vector_store.delete_glossary_term(n, session_id, user_id=user_id))
            pruned.add(term.name)
            logger.debug(f"Pruned ungrounded term: {term.name}")
        except Exception as e:
            logger.warning(f"Failed to prune term '{term.name}': {e}")

    if pruned:
        logger.info(f"Pruned {len(pruned)} ungrounded glossary terms")
    return pruned


def _link_parents(
    terms: list[GlossaryTerm],
    session_id: str,
    vector_store,
    active_domains: list[str] | None = None,
    domain: str | None = None,
    *,
    user_id: str | None = None,
) -> None:
    """Link parent_id fields based on suggested parent names.

    Looks up parents in: generated terms, existing glossary terms, then entities.
    Creates stub glossary entries for parents that don't exist anywhere —
    abstract parent terms are valid as long as their hierarchy terminates
    at concrete entities.
    """
    # Build name -> glossary_id lookup from generated + existing terms
    name_to_id = {}
    for term in terms:
        name_to_id[term.name.lower()] = term.id

    existing = vector_store.list_glossary_terms(session_id, user_id=user_id)
    for et in existing:
        if et.name.lower() not in name_to_id:
            name_to_id[et.name.lower()] = et.id

    scope_id = user_id or session_id

    # First pass: create stub glossary entries for referenced parents that
    # don't exist as glossary terms. This ensures parent_id always points
    # to a glossary term ID (not an entity ID), which keeps the hierarchy
    # traversable for grounding checks.
    for term in terms:
        suggested = term.tags.get("_suggested_parent", {})
        parent_name = suggested.get("name", "").lower() if suggested else ""
        if not parent_name or parent_name in name_to_id:
            continue

        # Parent doesn't exist as a glossary term — create a stub
        stub_id = _make_term_id(parent_name, scope_id, domain)
        stub = GlossaryTerm(
            id=stub_id,
            name=parent_name,
            display_name=display_entity_name(parent_name),
            definition=f"Parent category for related terms.",
            domain=domain,
            parent_id=None,
            aliases=[],
            semantic_type="concept",
            status="draft",
            provenance="llm",
            session_id=session_id,
            user_id=scope_id,
        )
        try:
            _retry_on_conflict(lambda s=stub: vector_store.add_glossary_term(s))
            name_to_id[parent_name] = stub_id
            logger.info(f"Created stub parent term: {parent_name}")
        except Exception as e:
            logger.warning(f"Failed to create stub parent '{parent_name}': {e}")

    # Second pass: link children to parents using glossary term IDs
    linked = 0
    for term in terms:
        suggested = term.tags.get("_suggested_parent", {})
        parent_name = suggested.get("name", "").lower() if suggested else ""
        if not parent_name:
            continue

        parent_id = name_to_id.get(parent_name)
        if not parent_id:
            logger.warning(f"Parent '{parent_name}' not found for term '{term.name}'")
            continue

        try:
            _retry_on_conflict(lambda: vector_store.update_glossary_term(
                term.name, session_id,
                {"parent_id": parent_id, "tags": {}},
                user_id=user_id,
            ))
            linked += 1
        except Exception as e:
            logger.warning(f"Failed to link parent for '{term.name}': {e}")

    if linked:
        logger.info(f"Linked {linked} parent relationships")


def reconcile_alias_entities(
    session_id: str,
    vector_store,
    *,
    user_id: str | None = None,
) -> int:
    """Reconcile entities that match glossary term aliases.

    When a glossary term has an alias that matches an existing entity name,
    rename that entity to the canonical term name and delete the glossary
    term if the term itself has no direct entity (was ungrounded).

    Example: entity "platinum" exists, glossary term "platinum tier" has
    alias "platinum" → rename entity to "platinum tier", remove the
    ungrounded glossary term if "platinum tier" had no entity.

    Returns:
        Number of entities reconciled
    """
    terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    reconciled = 0

    for term in terms:
        if not term.aliases:
            continue

        # Skip if a direct entity already exists for this term
        direct = vector_store.find_entity_by_name(term.name, session_id=session_id)
        if direct:
            # Check it has physical chunks (not just glossary-grounded)
            if _entity_has_physical_chunks(direct, vector_store):
                continue

        # Check each alias for a matching entity
        for alias in term.aliases:
            if not alias:
                continue
            alias_entity = vector_store.find_entity_by_name(alias, session_id=session_id)
            if not alias_entity:
                continue
            if not _entity_has_physical_chunks(alias_entity, vector_store):
                continue

            # Rename the alias entity to the canonical term name
            canonical = term.name.lower()
            canonical_display = term.display_name
            try:
                _retry_on_conflict(lambda: vector_store._conn.execute(
                    "UPDATE entities SET name = ?, display_name = ? WHERE id = ?",
                    [canonical, canonical_display, alias_entity.id],
                ))
                reconciled += 1
                logger.info(
                    f"Reconciled entity '{alias}' → '{canonical}' (id={alias_entity.id})"
                )
            except Exception as e:
                logger.warning(f"Failed to reconcile entity '{alias}' → '{canonical}': {e}")
            break  # Only reconcile one alias per term

    if reconciled:
        logger.info(f"Reconciled {reconciled} alias entities")

    # Deduplicate aliases across all terms
    _deduplicate_aliases(session_id, vector_store, user_id=user_id)

    return reconciled


def _deduplicate_aliases(session_id: str, vector_store, *, user_id: str | None = None) -> None:
    """Remove duplicate aliases across glossary terms.

    Rules:
    - An alias matching another term's primary name is removed
    - An alias matching an entity name is removed (the entity is canonical)
    - An alias claimed by an earlier term wins; later duplicates are stripped
    """
    terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    primary_names = {t.name.lower() for t in terms}

    # Collect all entity names
    entity_where, entity_params = vector_store.entity_visibility_filter(
        session_id, None, alias="e",
    )
    entity_rows = vector_store._conn.execute(
        f"SELECT LOWER(e.name) FROM entities e WHERE {entity_where}",
        entity_params,
    ).fetchall()
    entity_names = {r[0] for r in entity_rows}

    claimed: set[str] = set()
    for term in terms:
        if not term.aliases:
            continue

        clean = []
        changed = False
        for a in term.aliases:
            if not a:
                changed = True
                continue
            a_lower = a.strip().lower()
            if a_lower == term.name.lower():
                changed = True
                continue  # Self-reference
            if a_lower in primary_names:
                changed = True
                continue  # Another term's primary name
            if a_lower in entity_names and a_lower != term.name.lower():
                changed = True
                continue  # An entity's primary name
            if a_lower in claimed:
                changed = True
                continue  # Already claimed
            claimed.add(a_lower)
            clean.append(a.strip())

        if changed:
            try:
                _retry_on_conflict(lambda n=term.name, c=clean: vector_store.update_glossary_term(
                    n, session_id, {"aliases": c},
                    user_id=user_id,
                ))
            except Exception as e:
                logger.warning(f"Failed to deduplicate aliases for '{term.name}': {e}")

    logger.debug(f"Alias deduplication complete for session {session_id}")


def resolve_physical_resources(
    term_name: str,
    session_id: str,
    vector_store,
    domain_ids: list[str] | None = None,
    _visited: set | None = None,
    *,
    user_id: str | None = None,
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

    def _collect_entity_sources(ent):
        """Collect physical sources for an entity."""
        chunks = vector_store.get_chunks_for_entity(ent.id, domain_ids=domain_ids)
        seen_docs: set[str] = set()
        sources = []
        for _chunk_id, chunk, _confidence in chunks:
            if chunk.document_name.startswith("glossary:") or chunk.document_name.startswith("relationship:"):
                continue
            if chunk.document_name in seen_docs:
                continue
            seen_docs.add(chunk.document_name)
            sources.append({
                "document_name": chunk.document_name,
                "source": chunk.source,
                "section": chunk.section,
            })
        return sources

    # Direct match — concrete term
    entity = vector_store.find_entity_by_name(term_name, session_id=session_id)
    if entity:
        sources = _collect_entity_sources(entity)
        if sources:
            return [{
                "entity_name": entity.display_name,
                "entity_type": entity.semantic_type,
                "sources": sources,
            }]

    # Check aliases for entity matches
    term = vector_store.get_glossary_term(term_name, session_id, user_id=user_id)
    if term:
        for alias in (term.aliases or []):
            alias_entity = vector_store.find_entity_by_name(alias, session_id=session_id)
            if alias_entity:
                sources = _collect_entity_sources(alias_entity)
                if sources:
                    return [{
                        "entity_name": alias_entity.display_name,
                        "entity_type": alias_entity.semantic_type,
                        "sources": sources,
                    }]
    if not term:
        return []

    # Collection: follow list_of to element type
    if term.list_of:
        target = vector_store.get_glossary_term_by_id(term.list_of)
        if target:
            resources = resolve_physical_resources(
                target.name, session_id, vector_store, domain_ids, _visited,
                user_id=user_id,
            )
            if resources:
                return resources

    # Taxonomy: children are shown in the UI's Children section.
    # Don't duplicate their resources here — user navigates to them directly.
    return []


def _entity_has_physical_chunks(entity, vector_store) -> bool:
    """Check if an entity has non-glossary/relationship chunks."""
    chunks = vector_store.get_chunks_for_entity(entity.id)
    return any(
        not c.document_name.startswith("glossary:") and not c.document_name.startswith("rel:")
        for _, c, _ in chunks
    )


def is_grounded(
    term_name: str,
    session_id: str,
    vector_store,
    _visited: set | None = None,
    *,
    user_id: str | None = None,
) -> bool:
    """Check if a term is grounded in physical reality."""
    if _visited is None:
        _visited = set()

    key = f"{term_name}:{session_id}"
    if key in _visited:
        return False
    _visited.add(key)

    # Direct entity match — only grounded if it has non-glossary chunks
    entity = vector_store.find_entity_by_name(term_name, session_id=session_id)
    if entity and _entity_has_physical_chunks(entity, vector_store):
        return True

    # Check aliases — term may be grounded through an alias entity
    term = vector_store.get_glossary_term(term_name, session_id, user_id=user_id)
    if term:
        for alias in (term.aliases or []):
            alias_key = f"{alias.lower()}:{session_id}"
            if alias_key in _visited:
                continue
            _visited.add(alias_key)
            alias_entity = vector_store.find_entity_by_name(alias, session_id=session_id)
            if alias_entity and _entity_has_physical_chunks(alias_entity, vector_store):
                return True

    if not term:
        return False

    # Collection: follow list_of
    if term.list_of:
        target = vector_store.get_glossary_term_by_id(term.list_of)
        if target and is_grounded(target.name, session_id, vector_store, _visited, user_id=user_id):
            return True

    # Taxonomy: check children (by glossary ID and entity ID)
    entity_for_term = vector_store.find_entity_by_name(term_name, session_id=session_id)
    extra_ids = [entity_for_term.id] if entity_for_term else []
    children = vector_store.get_child_terms(term.id, *extra_ids)
    return any(is_grounded(c.name, session_id, vector_store, _visited, user_id=user_id) for c in children)


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
                target_table = fk.to_table.lower()

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
