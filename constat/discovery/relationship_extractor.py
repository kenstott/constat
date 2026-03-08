# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVO relationship extraction from document chunks using spaCy dependency parsing
and LLM-based refinement.

Multi-phase extraction:
Phase -1: Structural seeding — FK constraints (no glossary dependency) + Neo4j graph patterns
Phase 1:  SpaCy pass — extracts candidate SVOs from co-occurring entity pairs (cheap, fast)
Phase 2:  LLM pass — validates/rejects candidates, fixes verbs, infers implicit relationships
"""

import hashlib
import json
import logging
import re
from typing import Callable, Optional

from constat.core.models import TaskType

logger = logging.getLogger(__name__)

# Limits
MAX_RELATIONSHIPS_PER_ENTITY = 50
MAX_CO_OCCURRING_PAIRS = 50
MAX_CHUNKS_PER_PAIR = 10

# LLM refinement limits
LLM_BATCH_SIZE = 8
MAX_LLM_PAIRS = 75
MAX_EXCERPTS_PER_PAIR = 5
MAX_EXCERPT_LENGTH = 500
LLM_CONFIDENCE_MAP = {"high": 0.95, "medium": 0.8, "low": 0.6}

# Technical nouns that spaCy mis-tags as VERB in structured/schema text
_NON_VERBS = frozenset({
    "endpoint", "api", "query", "type", "field", "schema", "table",
    "column", "database", "server", "client", "model", "name", "value",
    "key", "index", "node", "config", "code", "file", "path", "route",
    "parameter", "object", "class", "method", "function", "module",
    "service", "resource", "record", "entry", "item", "data",
})

# Canonical verb vocabulary — small, constrained set for graph consistency.
# All verbs use Cypher-standard UPPER_SNAKE_CASE:
#   (Manager)-[:MANAGES]->(Employee)
PREFERRED_VERBS: dict[str, set[str]] = {
    "hierarchy": {"MANAGES", "REPORTS_TO"},
    "action": {"CREATES", "PROCESSES", "APPROVES", "PLACES"},
    "flow": {"SENDS", "RECEIVES", "TRANSFERS"},
    "causation": {"DRIVES", "REQUIRES", "ENABLES"},
    "temporal": {"PRECEDES", "FOLLOWS", "TRIGGERS"},
    "association": {"REFERENCES", "WORKS_IN", "PARTICIPATES_IN"},
}
# Verbs that overlap with the taxonomy (HAS_ONE, HAS_KIND, HAS_MANY).
# Rejected from SVO only when the pair already has a parent-child edge.
_HIERARCHY_VERBS = {"HAS", "HAS_ONE", "HAS_KIND", "HAS_MANY", "USES"}
ALL_PREFERRED = set().union(*PREFERRED_VERBS.values())

VERB_CATEGORIES = list(PREFERRED_VERBS.keys()) + ["other"]

_CANONICAL_VERB_LIST = ", ".join(sorted(ALL_PREFERRED))

# Bare "HAS" is never allowed — must be qualified
_VERB_REPLACEMENTS = {
    "HAS": "HAS_ONE",
    "CONTAIN": "CONTAINS",
    "CONTAINS": "HAS_MANY",
    "BELONG_TO": "BELONGS_TO",
    "BELONGS_TO": "HAS_ONE",
    "IS_TYPE_OF": "HAS_KIND",
}
# Verbs whose direction is inverted (child→parent becomes parent→child after replacement)
_SWAP_DIRECTION_VERBS = {"BELONGS_TO", "IS_TYPE_OF"}


def _normalize_verb(verb: str) -> tuple[str, bool]:
    """Normalize verb to UPPER_SNAKE_CASE, replacing disallowed bare forms.

    Returns:
        (normalized_verb, swap_direction) — swap_direction is True when
        subject and object should be swapped (e.g. BELONGS_TO → HAS_ONE).
    """
    if "_" not in verb.lower():
        verb = _to_third_person(verb.lower())
    verb = verb.upper()
    swap = verb in _SWAP_DIRECTION_VERBS
    verb = _VERB_REPLACEMENTS.get(verb, verb)
    return verb, swap


def _to_third_person(verb: str) -> str:
    """Convert a verb (lemma or already-inflected) to third-person singular present.

    Uses lemminflect: lemmatize first, then inflect to VBZ. Handles both
    base forms ("use") and already-inflected forms ("uses") correctly.

    For underscore compounds ("belong_to"), inflects only the first word.
    """
    if not verb:
        return verb
    # Underscore compound — inflect first word only
    if "_" in verb:
        parts = verb.split("_", 1)
        return _to_third_person(parts[0]) + "_" + parts[1]
    try:
        from lemminflect import getLemma, getInflection
        lemma = getLemma(verb, upos="VERB")
        base = lemma[0] if lemma else verb
        inflected = getInflection(base, tag="VBZ")
        return inflected[0] if inflected else verb
    except ImportError:
        # Fallback: simple rules if lemminflect unavailable
        return verb + "s" if not verb.endswith("s") else verb

RELATIONSHIP_SYSTEM_PROMPT = f"""You are extracting relationships between entity pairs from document excerpts.

For each pair of entities, you receive:
- The two entity names and their types
- Up to 3 shared document excerpts where both entities appear
- Any spaCy-suggested relationships (may be empty or inaccurate)

Return a JSON array of relationships found. For each relationship:
- subject: must be exactly one of the two entity names provided
- object: must be the other entity name
- verb: Cypher-standard UPPER_SNAKE_CASE (e.g., "MANAGES", "WORKS_IN", "TRIGGERS")
- verb_category: one of: ownership, hierarchy, action, flow, causation, temporal, association, other
- evidence: brief quote or paraphrase from the excerpt supporting this relationship
- confidence: high, medium, or low

VERB SELECTION — Choose from this canonical list: {_CANONICAL_VERB_LIST}
If none of these accurately describes the relationship, you may use a different UPPER_SNAKE_CASE verb, but strongly prefer the canonical list.

NEVER use bare "HAS" — always qualify as HAS_ONE (composition), HAS_MANY (collection), or HAS_KIND (taxonomy).
NEVER use CONTAINS (use HAS_MANY), BELONGS_TO (use HAS_ONE with swapped direction), or IS_TYPE_OF (use HAS_KIND with swapped direction).

DIRECTION RULE — The subject performs the action on the object:
- CORRECT: "Customer PLACES Order" (Customer is the actor)
- WRONG: "Order RECEIVES Customer" (Order is not the actor)

Examples of good relationships:
- {{"subject": "Manager", "verb": "MANAGES", "object": "Employee", "verb_category": "hierarchy"}}
- {{"subject": "Employee", "verb": "WORKS_IN", "object": "Department", "verb_category": "association"}}
- {{"subject": "Employee", "verb": "REPORTS_TO", "object": "Manager", "verb_category": "hierarchy"}}
- {{"subject": "Customer", "verb": "PLACES", "object": "Order", "verb_category": "action"}}
- {{"subject": "Invoice", "verb": "TRIGGERS", "object": "Payment", "verb_category": "temporal"}}
- {{"subject": "Policy", "verb": "REQUIRES", "object": "Approval", "verb_category": "causation"}}

Rules:
- Return EXACTLY ONE relationship per entity pair — pick the most specific, meaningful verb
- Avoid hierarchy/ownership verbs (HAS_ONE, HAS_MANY, HAS_KIND) when the pair already has a parent-child relationship — these duplicate the taxonomy
- Prefer spaCy suggestions when the verb is accurate; override when text supports a better verb
- You may infer implicit relationships even if no sentence states them directly
  (e.g., "Employee" WORKS_IN "Department" can be inferred from organizational context)
- Use Cypher-standard UPPER_SNAKE_CASE for all verbs (e.g., "MANAGES" not "manages")
- Use underscores for compound verbs (e.g., "REPORTS_TO", "WORKS_IN")
- subject and object must be exactly the entity names provided (no modifications)

Respond ONLY with a JSON array:
[{{"subject": "...", "verb": "...", "object": "...", "verb_category": "...", "evidence": "...", "confidence": "high|medium|low"}}]

Return [] if no relationships can be determined."""


def categorize_verb(verb: str) -> str:
    """Return the verb category name or 'other'."""
    v = verb.upper()
    for category, verbs in PREFERRED_VERBS.items():
        if v in verbs:
            return category
    return "other"


# ---------------------------------------------------------------------------
# Phase -1: Structural relationship seeding (FK + Neo4j graph patterns)
# ---------------------------------------------------------------------------

_GRAPH_PATTERN_RE = re.compile(r'\(:(\w+)\)-\[:(\w+)\]->\(:(\w+)\)')


def _parse_graph_pattern(pattern: str) -> tuple[str, str, str] | None:
    """Parse a Neo4j graph pattern like '(:Person)-[:ACTED_IN]->(:Movie)'.

    Returns:
        (source, relationship, target) or None if pattern doesn't match.
    """
    m = _GRAPH_PATTERN_RE.search(pattern)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def seed_structural_relationships(
    session_id: str,
    schema_manager,
    vector_store,
    on_batch: Callable[[list], None] | None = None,
) -> list:
    """Seed high-confidence structural relationships before SpaCy/LLM inference.

    Two sub-steps:
    A. FK constraints — creates REFERENCES relationships from TableMetadata.foreign_keys
       (no glossary dependency, uses raw table names)
    B. Neo4j graph patterns — parses graph_schema() patterns into EntityRelationships

    Args:
        session_id: Session ID
        schema_manager: SchemaManager with metadata_cache and nosql_connections
        vector_store: DuckDBVectorStore instance
        on_batch: Optional callback per batch of relationships

    Returns:
        List of EntityRelationship objects created.
    """
    from constat.catalog.nosql.base import NoSQLType
    from constat.discovery.models import EntityRelationship

    all_rels: list[EntityRelationship] = []

    # --- A. FK constraint seeding ---
    for table_meta in schema_manager.metadata_cache.values():
        for fk in table_meta.foreign_keys:
            subject = table_meta.name
            obj = fk.to_table

            rel_id = hashlib.sha256(
                f"{subject}:REFERENCES:{obj}:{session_id}".encode()
            ).hexdigest()[:16]

            rel = EntityRelationship(
                id=rel_id,
                subject_name=subject,
                verb="REFERENCES",
                object_name=obj,
                sentence=f"FK: {table_meta.name}.{fk.from_column} \u2192 {fk.to_table}.{fk.to_column}",
                confidence=0.95,
                verb_category="association",
                session_id=session_id,
            )

            try:
                vector_store.add_entity_relationship(rel)
            except Exception as insert_err:
                if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                    continue
                raise
            all_rels.append(rel)

    if all_rels:
        logger.info(f"FK structural seeding: {len(all_rels)} relationships for {session_id}")

    # --- B. Neo4j graph pattern seeding ---
    neo4j_count = 0
    for _db_name, connector in schema_manager.nosql_connections.items():
        if connector.nosql_type != NoSQLType.GRAPH:
            continue

        try:
            schema = connector.graph_schema()
        except Exception as e:
            logger.warning(f"Failed to get graph schema from {_db_name}: {e}")
            continue

        for pattern in schema.get("patterns", []):
            parsed = _parse_graph_pattern(pattern)
            if not parsed:
                continue

            src, rel_type, tgt = parsed

            rel_id = hashlib.sha256(
                f"{src}:{rel_type}:{tgt}:{session_id}".encode()
            ).hexdigest()[:16]

            rel = EntityRelationship(
                id=rel_id,
                subject_name=src,
                verb=rel_type,
                object_name=tgt,
                sentence=f"Graph pattern: {pattern}",
                confidence=0.98,
                verb_category=categorize_verb(rel_type),
                session_id=session_id,
            )

            try:
                vector_store.add_entity_relationship(rel)
            except Exception as insert_err:
                if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                    continue
                raise
            all_rels.append(rel)
            neo4j_count += 1

    if neo4j_count:
        logger.info(f"Neo4j structural seeding: {neo4j_count} relationships for {session_id}")

    if all_rels and on_batch:
        on_batch(all_rels)

    return all_rels


def _get_ancestors(token):
    """Yield ancestor tokens up to the root."""
    seen = set()
    current = token
    while current.head != current and current.i not in seen:
        seen.add(current.i)
        current = current.head
        yield current


def find_entity_span(sent, entity_name: str):
    """Find entity mention in a sentence span. Returns spaCy Span or None."""
    text_lower = sent.text.lower()
    name_lower = entity_name.lower()
    idx = text_lower.find(name_lower)
    if idx < 0:
        return None
    # Map character offset to token indices
    start_char = sent.start_char + idx
    end_char = start_char + len(name_lower)
    doc = sent.doc
    start_tok = None
    end_tok = None
    for token in sent:
        if token.idx <= start_char < token.idx + len(token.text):
            start_tok = token.i
        if token.idx < end_char <= token.idx + len(token.text):
            end_tok = token.i + 1
    if start_tok is not None and end_tok is not None:
        return doc[start_tok:end_tok]
    return None


def find_connecting_verb(sent, span_a, span_b):
    """Find the verb connecting two entity spans via LCA in dependency tree."""
    head_a = span_a.root
    head_b = span_b.root

    ancestors_a = set()
    for anc in _get_ancestors(head_a):
        ancestors_a.add(anc.i)
    ancestors_a.add(head_a.i)

    ancestors_b = set()
    for anc in _get_ancestors(head_b):
        ancestors_b.add(anc.i)
    ancestors_b.add(head_b.i)

    # Walk from head_a upward, find first ancestor also in ancestors_b
    lca = None
    for anc in _get_ancestors(head_a):
        if anc.i in ancestors_b:
            lca = anc
            break
    if lca is None:
        # Try head tokens directly
        if head_a.i in ancestors_b:
            lca = head_a
        elif head_b.i in ancestors_a:
            lca = head_b

    if lca is None:
        return None

    if lca.pos_ == "VERB":
        return lca

    # Walk up from LCA to find nearest verb ancestor
    for anc in _get_ancestors(lca):
        if anc.pos_ == "VERB":
            return anc

    return None


def determine_svo_direction(span_a, span_b, verb, entity_a_name: str, entity_b_name: str):
    """Determine subject/object based on dependency labels. Returns (subject_name, object_name)."""
    head_a = span_a.root
    head_b = span_b.root

    def is_subject_of(token, verb_token):
        """Check if token is a subject of the verb via dep chain."""
        if token.head.i == verb_token.i and token.dep_ in ("nsubj", "nsubjpass"):
            return True
        # Walk up
        for anc in _get_ancestors(token):
            if anc.i == verb_token.i:
                break
            if anc.dep_ in ("nsubj", "nsubjpass") and anc.head.i == verb_token.i:
                return True
        return False

    a_is_subj = is_subject_of(head_a, verb)
    b_is_subj = is_subject_of(head_b, verb)

    if a_is_subj and not b_is_subj:
        return entity_a_name, entity_b_name
    if b_is_subj and not a_is_subj:
        return entity_b_name, entity_a_name
    # Fallback: word order
    if span_a.start < span_b.start:
        return entity_a_name, entity_b_name
    return entity_b_name, entity_a_name


def extract_svo_relationships(
    session_id: str,
    vector_store,
    nlp,
    min_cooccurrence: int = 2,
    on_batch: Callable[[list], None] | None = None,
) -> list:
    """Extract SVO relationships from co-occurring entity pairs.

    Args:
        session_id: Session ID
        vector_store: DuckDBVectorStore instance
        nlp: spaCy Language model
        min_cooccurrence: Minimum co-occurrence count to process
        on_batch: Optional callback per entity-pair batch

    Returns:
        List of EntityRelationship objects
    """
    from constat.discovery.models import EntityRelationship

    # Get co-occurring entity pairs
    pairs = vector_store.get_cooccurrence_pairs(
        session_id, min_count=min_cooccurrence, limit=MAX_CO_OCCURRING_PAIRS,
    )

    if not pairs:
        logger.info(f"No co-occurring pairs found for session {session_id}")
        return []

    logger.info(f"SVO extraction: processing {len(pairs)} co-occurring pairs")

    all_relationships: list[EntityRelationship] = []
    relationship_count_by_entity: dict[str, int] = {}

    for e1_id, e1_name, e2_id, e2_name, _co_count in pairs:
        # Check per-entity limits (by name)
        if relationship_count_by_entity.get(e1_name, 0) >= MAX_RELATIONSHIPS_PER_ENTITY:
            continue
        if relationship_count_by_entity.get(e2_name, 0) >= MAX_RELATIONSHIPS_PER_ENTITY:
            continue

        # Get shared chunk IDs
        shared_chunk_ids = vector_store.get_shared_chunk_ids(e1_id, e2_id, MAX_CHUNKS_PER_PAIR)

        batch_rels: list[EntityRelationship] = []

        for chunk_id in shared_chunk_ids:
            # Load chunk text
            chunk_text = vector_store.get_chunk_content(chunk_id)
            if not chunk_text:
                continue

            try:
                doc = nlp(chunk_text)
            except Exception:
                continue

            for sent in doc.sents:
                span_a = find_entity_span(sent, e1_name)
                span_b = find_entity_span(sent, e2_name)
                if not (span_a and span_b):
                    continue

                verb = find_connecting_verb(sent, span_a, span_b)
                if not verb:
                    continue

                lemma = verb.lemma_.lower()
                if lemma in _NON_VERBS:
                    continue
                verb_form, swap = _normalize_verb(lemma)
                category = categorize_verb(verb_form)
                confidence = 1.0 if verb_form in ALL_PREFERRED else 0.5

                subject_name, object_name = determine_svo_direction(
                    span_a, span_b, verb, e1_name, e2_name,
                )
                if swap:
                    subject_name, object_name = object_name, subject_name

                rel_id = hashlib.sha256(
                    f"{subject_name}:{verb_form}:{object_name}:{session_id}".encode()
                ).hexdigest()[:16]

                rel = EntityRelationship(
                    id=rel_id,
                    subject_name=subject_name,
                    verb=verb_form,
                    object_name=object_name,
                    sentence=sent.text.strip(),
                    confidence=confidence,
                    verb_category=category,
                    session_id=session_id,
                )

                try:
                    vector_store.add_entity_relationship(rel)
                except Exception as insert_err:
                    if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                        continue
                    raise
                batch_rels.append(rel)

                # Track per-entity counts
                relationship_count_by_entity[subject_name] = relationship_count_by_entity.get(subject_name, 0) + 1
                relationship_count_by_entity[object_name] = relationship_count_by_entity.get(object_name, 0) + 1

        if batch_rels:
            all_relationships.extend(batch_rels)
            if on_batch:
                on_batch(batch_rels)

    logger.info(f"SVO extraction complete: {len(all_relationships)} relationships for session {session_id}")
    return all_relationships


# ---------------------------------------------------------------------------
# LLM-based relationship refinement (Phase 2)
# ---------------------------------------------------------------------------

def get_co_occurring_pairs(
    session_id: str,
    vector_store,
    min_cooccurrence: int = 2,
    limit: int = MAX_LLM_PAIRS,
) -> list[tuple]:
    """Query co-occurring entity pairs with their semantic types.

    Returns:
        List of (e1_id, e1_name, e1_type, e2_id, e2_name, e2_type, co_count)
    """
    return vector_store.get_cooccurrence_pairs(
        session_id, min_count=min_cooccurrence, limit=limit, include_types=True,
    )


def _get_shared_excerpts(
    vector_store,
    e1_id: str,
    e2_id: str,
    max_excerpts: int = MAX_EXCERPTS_PER_PAIR,
    max_length: int = MAX_EXCERPT_LENGTH,
) -> list[str]:
    """Get shared chunk excerpts for an entity pair."""
    contents = vector_store.get_shared_chunk_content(e1_id, e2_id, max_excerpts)
    return [c[:max_length] for c in contents]


def _parse_llm_response(content: str) -> list[dict]:
    """Parse LLM JSON response, stripping markdown fences."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
        content = content.strip()

    raw = None
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("[")
        end = content.rfind("]")
        if start >= 0 and end > start:
            try:
                raw = json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass

    if not isinstance(raw, list):
        logger.warning(f"Failed to parse relationship LLM response: {content[:200]}")
        return []

    # Flatten nested lists (LLM sometimes returns [[{...}], [{...}]])
    flat: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(i for i in item if isinstance(i, dict))
    return flat


def _validate_relationship(item: dict, e1_name: str, e2_name: str) -> bool:
    """Validate that subject/object match the entity pair names."""
    subj = item.get("subject", "")
    obj = item.get("object", "")
    verb = item.get("verb", "")
    if not (subj and obj and verb):
        return False
    pair_names = {e1_name.lower(), e2_name.lower()}
    return subj.lower() in pair_names and obj.lower() in pair_names


def refine_relationships_with_llm(
    session_id: str,
    vector_store,
    router,
    svo_candidates: list,
    co_occurring_pairs: list[tuple],
    on_batch: Callable[[list], None] | None = None,
) -> list:
    """Refine relationships using LLM, using spaCy SVOs as suggestions.

    Args:
        session_id: Session ID
        vector_store: DuckDBVectorStore instance
        router: LLM router with .execute()
        svo_candidates: SpaCy-extracted EntityRelationship objects
        co_occurring_pairs: From get_co_occurring_pairs()
        on_batch: Optional callback per LLM batch

    Returns:
        List of EntityRelationship objects created by LLM
    """
    from constat.discovery.models import EntityRelationship

    # Build spaCy suggestion map: (lower_name_a, lower_name_b) -> [{verb, verb_category, sentence}]
    svo_map: dict[tuple[str, str], list[dict]] = {}
    for rel in svo_candidates:
        key = tuple(sorted([rel.subject_name.lower(), rel.object_name.lower()]))
        svo_map.setdefault(key, []).append({
            "subject": rel.subject_name,
            "verb": rel.verb,
            "object": rel.object_name,
            "verb_category": rel.verb_category,
            "sentence": rel.sentence,
        })

    # Build pair contexts for LLM
    pair_contexts: list[dict] = []
    for e1_id, e1_name, e1_type, e2_id, e2_name, e2_type, co_count in co_occurring_pairs:
        excerpts = _get_shared_excerpts(vector_store, e1_id, e2_id)
        if not excerpts:
            continue

        key = tuple(sorted([e1_name.lower(), e2_name.lower()]))
        suggestions = svo_map.get(key, [])

        pair_contexts.append({
            "e1_name": e1_name,
            "e1_type": e1_type or "unknown",
            "e2_name": e2_name,
            "e2_type": e2_type or "unknown",
            "excerpts": excerpts,
            "spacy_suggestions": suggestions,
        })

    if not pair_contexts:
        logger.info(f"LLM refinement: no pair contexts to process for session {session_id}")
        return []

    logger.info(f"LLM refinement: processing {len(pair_contexts)} pairs in batches of {LLM_BATCH_SIZE}")

    all_llm_rels: list[EntityRelationship] = []

    # Process in batches
    for batch_start in range(0, len(pair_contexts), LLM_BATCH_SIZE):
        batch = pair_contexts[batch_start:batch_start + LLM_BATCH_SIZE]

        # Build user message
        parts = []
        for i, ctx in enumerate(batch):
            part = f"Pair {i + 1}: {ctx['e1_name']} ({ctx['e1_type']}) <-> {ctx['e2_name']} ({ctx['e2_type']})\n"
            part += "Excerpts:\n"
            for j, excerpt in enumerate(ctx["excerpts"]):
                part += f"  [{j + 1}] {excerpt}\n"
            if ctx["spacy_suggestions"]:
                part += "SpaCy suggestions:\n"
                for s in ctx["spacy_suggestions"]:
                    part += f"  - {s['subject']} → {s['verb']} → {s['object']} ({s['verb_category']})\n"
            parts.append(part)

        user_message = "\n---\n".join(parts)

        try:
            result = router.execute(
                task_type=TaskType.RELATIONSHIP_EXTRACTION,
                system=RELATIONSHIP_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=2048,
                complexity="low",
            )

            if not result.success:
                logger.warning(f"LLM relationship batch failed: {result.content}")
                continue

            items = _parse_llm_response(result.content)
            batch_rels: list[EntityRelationship] = []

            # Build name lookup for this batch
            batch_pair_names: set[tuple[str, str]] = set()
            for ctx in batch:
                batch_pair_names.add((ctx["e1_name"], ctx["e2_name"]))

            for item in items:
                # Find which pair this relationship belongs to
                subj = item.get("subject", "")
                obj = item.get("object", "")
                matched_pair = None
                for ctx in batch:
                    if _validate_relationship(item, ctx["e1_name"], ctx["e2_name"]):
                        matched_pair = ctx
                        break

                if not matched_pair:
                    continue

                verb = item.get("verb", "").strip()
                if not verb:
                    continue
                verb, swap = _normalize_verb(verb)

                # Use LLM-provided category, fall back to our own categorization
                verb_category = item.get("verb_category", "")
                if verb_category not in VERB_CATEGORIES:
                    verb_category = categorize_verb(verb)

                confidence_str = item.get("confidence", "medium")
                confidence = LLM_CONFIDENCE_MAP.get(confidence_str, 0.8)

                evidence = item.get("evidence", "")

                # Use exact entity names from the pair context
                # (LLM might return slightly different casing)
                subject_name = matched_pair["e1_name"] if subj.lower() == matched_pair["e1_name"].lower() else matched_pair["e2_name"]
                object_name = matched_pair["e2_name"] if subject_name == matched_pair["e1_name"] else matched_pair["e1_name"]
                if swap:
                    subject_name, object_name = object_name, subject_name

                rel_id = hashlib.sha256(
                    f"{subject_name}:{verb}:{object_name}:{session_id}".encode()
                ).hexdigest()[:16]

                rel = EntityRelationship(
                    id=rel_id,
                    subject_name=subject_name,
                    verb=verb,
                    object_name=object_name,
                    sentence=evidence,
                    confidence=confidence,
                    verb_category=verb_category,
                    session_id=session_id,
                )

                try:
                    vector_store.add_entity_relationship(rel)
                except Exception as insert_err:
                    if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                        continue
                    raise
                batch_rels.append(rel)

            if batch_rels:
                all_llm_rels.extend(batch_rels)
                if on_batch:
                    on_batch(batch_rels)

            logger.info(f"LLM batch {batch_start // LLM_BATCH_SIZE + 1}: {len(batch_rels)} relationships")

        except Exception as e:
            logger.exception(f"LLM relationship batch failed: {e}")
            continue

    logger.info(f"LLM refinement complete: {len(all_llm_rels)} relationships for session {session_id}")
    return all_llm_rels


# ---------------------------------------------------------------------------
# Phase 0: FK-based relationships
# ---------------------------------------------------------------------------

def store_fk_relationships(
    session_id: str,
    glossary_terms: list,
    schema_manager,
    vector_store,
    on_batch: Callable[[list], None] | None = None,
) -> list:
    """Create relationships from foreign key constraints.

    Calls suggest_fk_relationships() from glossary_generator and converts
    each suggestion to an EntityRelationship stored in the vector store.
    """
    from constat.discovery.glossary_generator import suggest_fk_relationships
    from constat.discovery.models import EntityRelationship

    suggestions = suggest_fk_relationships(session_id, glossary_terms, schema_manager)
    if not suggestions:
        return []

    all_rels: list[EntityRelationship] = []
    for s in suggestions:
        source = s["source"]
        target = s["target"]

        rel_id = hashlib.sha256(
            f"{source}:HAS_ONE:{target}:{session_id}".encode()
        ).hexdigest()[:16]

        # Build evidence from FK metadata
        evidence = s.get("relationship", f"FK: {source} -> {target}")

        rel = EntityRelationship(
            id=rel_id,
            subject_name=source,
            verb="HAS_ONE",
            object_name=target,
            sentence=evidence,
            confidence=0.95,
            verb_category="ownership",
            session_id=session_id,
        )

        try:
            vector_store.add_entity_relationship(rel)
        except Exception as insert_err:
            if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                continue
            raise
        all_rels.append(rel)

    if all_rels and on_batch:
        on_batch(all_rels)

    logger.info(f"FK relationships: {len(all_rels)} for session {session_id}")
    return all_rels


# ---------------------------------------------------------------------------
# Phase 3: Glossary-informed LLM relationship inference
# ---------------------------------------------------------------------------

GLOSSARY_RELATIONSHIP_PROMPT = f"""You are inferring cross-cutting relationships between business glossary terms.

For each group of terms below, you receive:
- Term name, definition, semantic type, and parent (if any)

Infer **cross-cutting** relationships between terms based on their definitions and business semantics.

VERB SELECTION — Choose from this canonical list: {_CANONICAL_VERB_LIST}
If none of these accurately describes the relationship, you may use a different UPPER_SNAKE_CASE verb, but strongly prefer the canonical list.

NEVER use bare "HAS" — always qualify as HAS_ONE (composition), HAS_MANY (collection), or HAS_KIND (taxonomy).
NEVER use CONTAINS (use HAS_MANY), BELONGS_TO (use HAS_ONE with swapped direction), or IS_TYPE_OF (use HAS_KIND with swapped direction).

DIRECTION RULE — The subject performs the action on the object:
- CORRECT: "Customer PLACES Order" (Customer is the actor)
- WRONG: "Order RECEIVES Customer" (Order is not the actor)

IMPORTANT RULES:
- Return EXACTLY ONE relationship per entity pair — pick the most specific, meaningful verb
- Do NOT return relationships that duplicate an existing parent-child edge between the same pair
  (e.g., if "Order" is parent of "Line Item" with HAS_MANY, do not return "Order HAS_MANY Line Item")
- Hierarchy/ownership verbs (HAS_ONE, HAS_MANY, HAS_KIND) are fine between terms that are NOT in a parent-child relationship
- Focus on: action relationships, causation, temporal ordering, association between entities in different branches
- Use Cypher-standard UPPER_SNAKE_CASE for all verbs (e.g., "MANAGES", "TRIGGERS", "APPROVES")
- Use underscores for compound verbs (e.g., "BELONGS_TO", "WORKS_IN")
- Only return relationships where you have reasonable confidence from the definitions

Examples of good cross-cutting relationships:
- {{"subject": "Employee", "verb": "WORKS_IN", "object": "Department", "verb_category": "association"}}
- {{"subject": "Invoice", "verb": "TRIGGERS", "object": "Payment", "verb_category": "temporal"}}
- {{"subject": "Customer", "verb": "PLACES", "object": "Order", "verb_category": "action"}}
- {{"subject": "Manager", "verb": "APPROVES", "object": "Expense", "verb_category": "action"}}
- {{"subject": "Policy", "verb": "REQUIRES", "object": "Claim", "verb_category": "causation"}}

verb_category must be one of: hierarchy, action, flow, causation, temporal, association, other

Respond ONLY with a JSON array:
[{{"subject": "...", "verb": "...", "object": "...", "verb_category": "...", "evidence": "...", "confidence": "high|medium|low"}}]

Return [] if no cross-cutting relationships can be inferred."""


def infer_glossary_relationships(
    session_id: str,
    vector_store,
    router,
    on_batch: Callable[[list], None] | None = None,
    *,
    user_id: str | None = None,
) -> list:
    """Infer cross-cutting relationships from glossary term definitions.

    Loads all glossary terms, groups them into batches, and uses an LLM
    to infer relationships from definition semantics.
    """
    from constat.discovery.models import EntityRelationship

    terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    if not terms:
        return []

    # Build term lookup and name set
    term_names = {t.name.lower() for t in terms}
    term_by_name = {t.name.lower(): t for t in terms}
    term_by_id = {t.id: t for t in terms}

    # Build parent-child verb lookup: (child_name_lower, parent_name_lower) -> parent_verb
    # Used to skip SVO relationships that duplicate the taxonomy edge
    _parent_child_verbs: dict[tuple[str, str], str] = {}
    for t in terms:
        if t.parent_id and t.parent_id in term_by_id:
            parent = term_by_id[t.parent_id]
            _parent_child_verbs[(t.name.lower(), parent.name.lower())] = t.parent_verb.upper()

    # Group terms into batches of ~12, preferring related terms together
    # Build parent -> children map
    children_of: dict[str, list] = {}
    for t in terms:
        if t.parent_id:
            children_of.setdefault(t.parent_id, []).append(t)

    batches: list[list] = []
    used = set()

    # First pass: group siblings under same parent
    for parent_id, children in children_of.items():
        group = []
        for c in children:
            if c.name.lower() not in used:
                group.append(c)
                used.add(c.name.lower())
            if len(group) >= 12:
                batches.append(group)
                group = []
        if group:
            batches.append(group)

    # Second pass: remaining terms
    remaining = [t for t in terms if t.name.lower() not in used]
    for i in range(0, len(remaining), 12):
        batches.append(remaining[i:i + 12])

    if not batches:
        return []

    logger.info(f"Glossary relationship inference: {len(terms)} terms in {len(batches)} batches")

    all_rels: list[EntityRelationship] = []

    for batch_idx, batch in enumerate(batches):
        # Build user message with term details
        parts = []
        for t in batch:
            parent_name = ""
            if t.parent_id:
                # Look up parent display name
                parent_term = next(
                    (pt for pt in terms if pt.id == t.parent_id), None
                )
                if parent_term:
                    parent_name = parent_term.display_name

            part = f"- {t.display_name}"
            if t.semantic_type:
                part += f" ({t.semantic_type})"
            if parent_name:
                part += f" [parent: {parent_name}]"
            part += f"\n  Definition: {t.definition or 'N/A'}"
            parts.append(part)

        user_message = "Terms:\n" + "\n".join(parts)

        try:
            result = router.execute(
                task_type=TaskType.RELATIONSHIP_EXTRACTION,
                system=GLOSSARY_RELATIONSHIP_PROMPT,
                user_message=user_message,
                max_tokens=2048,
                complexity="low",
            )

            if not result.success:
                logger.warning(f"Glossary relationship batch {batch_idx} failed: {result.content}")
                continue

            items = _parse_llm_response(result.content)
            batch_rels: list[EntityRelationship] = []

            batch_names = {t.name.lower() for t in batch}

            for item in items:
                subj = item.get("subject", "").strip()
                obj = item.get("object", "").strip()
                verb = item.get("verb", "").strip()
                if not (subj and obj and verb):
                    continue

                # Validate subject and object exist in glossary
                subj_lower = subj.lower()
                obj_lower = obj.lower()
                if subj_lower not in term_names or obj_lower not in term_names:
                    continue

                # Normalize verb to Cypher UPPER_SNAKE_CASE
                verb, swap = _normalize_verb(verb)
                if swap:
                    subj_lower, obj_lower = obj_lower, subj_lower

                # Skip if this verb duplicates an existing parent-child edge
                if verb in _HIERARCHY_VERBS:
                    pc_verb = _parent_child_verbs.get((subj_lower, obj_lower)) or _parent_child_verbs.get((obj_lower, subj_lower))
                    if pc_verb:
                        continue

                verb_category = item.get("verb_category", "")
                if verb_category not in VERB_CATEGORIES:
                    verb_category = categorize_verb(verb)

                confidence_str = item.get("confidence", "medium")
                confidence = LLM_CONFIDENCE_MAP.get(confidence_str, 0.8)

                evidence = item.get("evidence", "")

                # Use canonical names from glossary
                subject_name = term_by_name[subj_lower].name
                object_name = term_by_name[obj_lower].name

                rel_id = hashlib.sha256(
                    f"{subject_name}:{verb}:{object_name}:{session_id}".encode()
                ).hexdigest()[:16]

                rel = EntityRelationship(
                    id=rel_id,
                    subject_name=subject_name,
                    verb=verb,
                    object_name=object_name,
                    sentence=evidence,
                    confidence=confidence,
                    verb_category=verb_category,
                    session_id=session_id,
                )

                try:
                    vector_store.add_entity_relationship(rel)
                except Exception as insert_err:
                    if "Duplicate key" in str(insert_err) or "UNIQUE constraint" in str(insert_err):
                        logger.debug(f"Skipping duplicate relationship {rel_id}: {subject_name} {verb} {object_name}")
                        continue
                    raise
                batch_rels.append(rel)

            if batch_rels:
                all_rels.extend(batch_rels)
                if on_batch:
                    on_batch(batch_rels)

            logger.info(f"Glossary relationship batch {batch_idx}: {len(batch_rels)} relationships")

        except Exception as e:
            logger.exception(f"Glossary relationship batch {batch_idx} failed: {e}")
            continue

    logger.info(f"Glossary relationship inference complete: {len(all_rels)} relationships for session {session_id}")
    return all_rels


# ---------------------------------------------------------------------------
# Deduplication: keep only the best relationship per entity pair
# ---------------------------------------------------------------------------

def deduplicate_relationships(session_id: str, vector_store) -> int:
    """Keep only the highest-confidence relationship per entity pair.

    For each unordered pair (A, B) — regardless of which is subject/object —
    keeps the one with the highest confidence and deletes the rest.

    Returns:
        Number of relationships deleted.
    """
    # Build set of parent-child pairs from glossary taxonomy.
    # Any SVO relationship duplicating a taxonomy edge is redundant.
    parent_child_pairs: set[tuple[str, str]] = set()
    pc_rows = vector_store.get_glossary_parent_child_pairs(session_id)
    for child_name, parent_name in pc_rows:
        parent_child_pairs.add(tuple(sorted([child_name.lower(), parent_name.lower()])))

    rows = vector_store.list_session_relationships(session_id)

    # Group by unordered pair
    best_by_pair: dict[tuple[str, str], tuple] = {}
    duplicates: list[str] = []

    for row in rows:
        rel_id, subj, obj = row["id"], row["subject_name"], row["object_name"]
        confidence, user_edited = row["confidence"], row["user_edited"]
        pair_key = tuple(sorted([subj.lower(), obj.lower()]))
        # Remove SVO relationships that duplicate a parent-child taxonomy edge
        # but never delete user-edited relationships
        if pair_key in parent_child_pairs and not user_edited:
            duplicates.append(rel_id)
            continue
        if pair_key not in best_by_pair:
            best_by_pair[pair_key] = (rel_id, confidence)
        else:
            # Never delete user-edited rows as duplicates
            if user_edited:
                # Evict the previous winner
                old_id, _ = best_by_pair[pair_key]
                duplicates.append(old_id)
                best_by_pair[pair_key] = (rel_id, confidence)
            else:
                duplicates.append(rel_id)

    # Delete duplicates
    for rel_id in duplicates:
        vector_store.delete_entity_relationship(rel_id)

    if duplicates:
        logger.info(f"Deduplicated relationships for {session_id}: removed {len(duplicates)}, kept {len(best_by_pair)}")

    return len(duplicates)


# ---------------------------------------------------------------------------
# HAS_* promotion: convert orphan HAS relationships to taxonomy edges
# ---------------------------------------------------------------------------

_HAS_VERBS = {"HAS_ONE", "HAS_MANY", "HAS_KIND"}


def promote_has_relationships(
    session_id: str,
    vector_store,
    user_id: str | None = None,
) -> int:
    """Promote non-user-edited HAS_ONE/HAS_MANY/HAS_KIND relationships to taxonomy edges.

    For each matching relationship where the object term has no parent,
    sets the subject as its parent (with the HAS verb) and deletes the relationship.

    Returns:
        Number of relationships promoted.
    """
    rows = vector_store.get_promotable_relationships(session_id)

    if not rows:
        return 0

    # Build lookup of glossary terms by lowercase name
    terms = vector_store.list_glossary_terms(session_id, user_id=user_id)
    term_by_name: dict[str, object] = {t.name.lower(): t for t in terms}

    promoted = 0
    for rel_id, subject_name, verb, object_name in rows:
        child = term_by_name.get(object_name.lower())
        if not child or child.parent_id:
            continue
        parent = term_by_name.get(subject_name.lower())
        if not parent:
            continue

        vector_store.update_glossary_term(
            child.name, session_id,
            {"parent_id": parent.id, "parent_verb": verb},
            user_id=user_id,
        )
        vector_store.delete_entity_relationship(rel_id)
        promoted += 1

    if promoted:
        logger.info(f"Promoted {promoted} HAS relationships to taxonomy edges for {session_id}")

    return promoted
