# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SVO relationship extraction from document chunks using spaCy dependency parsing.

Extracts Subject-Verb-Object triples scoped to co-occurring entity pairs.
Preferred verbs get confidence 1.0; others get 0.5 but are not excluded.
"""

import hashlib
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Limits
MAX_RELATIONSHIPS_PER_ENTITY = 50
MAX_CO_OCCURRING_PAIRS = 20
MAX_CHUNKS_PER_PAIR = 10

# Preferred verb categories — verbs here get confidence 1.0
PREFERRED_VERBS: dict[str, set[str]] = {
    "ownership": {"own", "have", "contain", "include", "hold", "belong"},
    "action": {"receive", "submit", "approve", "reject", "send", "create", "process"},
    "causation": {"determine", "affect", "influence", "cause", "drive", "require"},
    "temporal": {"precede", "follow", "trigger", "initiate", "complete"},
    "association": {"relate", "associate", "link", "connect", "correspond", "reference"},
}
ALL_PREFERRED = set().union(*PREFERRED_VERBS.values())


def categorize_verb(lemma: str) -> str:
    """Return the verb category name or 'other'."""
    for category, verbs in PREFERRED_VERBS.items():
        if lemma in verbs:
            return category
    return "other"


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


def determine_svo_direction(span_a, span_b, verb, entity_a_id: str, entity_b_id: str):
    """Determine subject/object based on dependency labels. Returns (subject_id, object_id)."""
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
        return entity_a_id, entity_b_id
    if b_is_subj and not a_is_subj:
        return entity_b_id, entity_a_id
    # Fallback: word order
    if span_a.start < span_b.start:
        return entity_a_id, entity_b_id
    return entity_b_id, entity_a_id


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
    pairs = vector_store._conn.execute("""
        SELECT e1.id, e1.name, e2.id, e2.name, COUNT(*) as co_count
        FROM chunk_entities ce1
        JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id AND ce1.entity_id < ce2.entity_id
        JOIN entities e1 ON ce1.entity_id = e1.id
        JOIN entities e2 ON ce2.entity_id = e2.id
        WHERE e1.session_id = ? AND e2.session_id = ?
        GROUP BY e1.id, e1.name, e2.id, e2.name
        HAVING COUNT(*) >= ?
        ORDER BY co_count DESC
        LIMIT ?
    """, [session_id, session_id, min_cooccurrence, MAX_CO_OCCURRING_PAIRS]).fetchall()

    if not pairs:
        logger.info(f"No co-occurring pairs found for session {session_id}")
        return []

    logger.info(f"SVO extraction: processing {len(pairs)} co-occurring pairs")

    all_relationships: list[EntityRelationship] = []
    relationship_count_by_entity: dict[str, int] = {}

    for e1_id, e1_name, e2_id, e2_name, _co_count in pairs:
        # Check per-entity limits
        if relationship_count_by_entity.get(e1_id, 0) >= MAX_RELATIONSHIPS_PER_ENTITY:
            continue
        if relationship_count_by_entity.get(e2_id, 0) >= MAX_RELATIONSHIPS_PER_ENTITY:
            continue

        # Get shared chunk IDs
        shared_chunks = vector_store._conn.execute("""
            SELECT DISTINCT ce1.chunk_id
            FROM chunk_entities ce1
            JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
            WHERE ce1.entity_id = ? AND ce2.entity_id = ?
            LIMIT ?
        """, [e1_id, e2_id, MAX_CHUNKS_PER_PAIR]).fetchall()

        batch_rels: list[EntityRelationship] = []

        for (chunk_id,) in shared_chunks:
            # Load chunk text
            chunk_row = vector_store._conn.execute(
                "SELECT content FROM embeddings WHERE chunk_id = ?",
                [chunk_id],
            ).fetchone()
            if not chunk_row:
                continue

            try:
                doc = nlp(chunk_row[0])
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
                category = categorize_verb(lemma)
                confidence = 1.0 if lemma in ALL_PREFERRED else 0.5

                subject_id, object_id = determine_svo_direction(
                    span_a, span_b, verb, e1_id, e2_id,
                )

                rel_id = hashlib.sha256(
                    f"{subject_id}:{lemma}:{object_id}:{chunk_id}".encode()
                ).hexdigest()[:16]

                rel = EntityRelationship(
                    id=rel_id,
                    subject_entity_id=subject_id,
                    verb=lemma,
                    object_entity_id=object_id,
                    chunk_id=chunk_id,
                    sentence=sent.text.strip(),
                    confidence=confidence,
                    verb_category=category,
                    session_id=session_id,
                )

                vector_store.add_entity_relationship(rel)
                batch_rels.append(rel)

                # Track per-entity counts
                relationship_count_by_entity[subject_id] = relationship_count_by_entity.get(subject_id, 0) + 1
                relationship_count_by_entity[object_id] = relationship_count_by_entity.get(object_id, 0) + 1

        if batch_rels:
            all_relationships.extend(batch_rels)
            if on_batch:
                on_batch(batch_rels)

    logger.info(f"SVO extraction complete: {len(all_relationships)} relationships for session {session_id}")
    return all_relationships
