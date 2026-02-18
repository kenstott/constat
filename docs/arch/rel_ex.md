# Relationship Extraction Architecture

> **Status:** Pass 1 (co-occurrence) is implemented and used in production. Pass 2 (SVO extraction) is **future work** — too noisy for automated relationship creation. SVO extraction is viable as a **suggestion method**: extract candidate triples, surface in UI, user confirms or discards, confirmed relationships persist to the curated `relationships` config (see `config_merge.md` section 6). Same pattern as glossary auto-suggest bindings.

## Overview

Extract semantic relationships between entities using a two-pass approach:
1. **Pass 1 (cheap)**: Co-occurrence detection via shared chunks — **automated, reliable**
2. **Pass 2 (targeted)**: SVO extraction scoped to co-occurring pairs — **suggestion-only, user confirms**

This approach avoids expensive full-text parsing by using co-occurrence as a filter.

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Indexing                         │
├─────────────────────────────────────────────────────────────┤
│  Chunks  ──►  Entities  ──►  Co-occurrence  ──►  Relations  │
│  (text)      (NER/regex)    (chunk_entities)   (SVO parse)  │
└─────────────────────────────────────────────────────────────┘
                                    │
                              Scopes Pass 2
                              (only parse pairs
                               that co-occur)
```

## Data Model

### Existing Tables (unchanged)

| Table | Purpose |
|-------|---------|
| `entities` | Named entities (people, tables, concepts, etc.) |
| `chunk_entities` | Junction: which entities appear in which chunks |

### New Table: `entity_relationships`

```sql
CREATE TABLE entity_relationships (
    id VARCHAR PRIMARY KEY,
    subject_name VARCHAR NOT NULL,   -- Entity/term name (natural key)
    verb VARCHAR NOT NULL,           -- Lemmatized verb (e.g., "receive", "determine")
    object_name VARCHAR NOT NULL,    -- Entity/term name (natural key)
    sentence TEXT,                   -- Original sentence (for UI display)
    confidence FLOAT DEFAULT 1.0,
    verb_category VARCHAR DEFAULT 'other',
    session_id VARCHAR,              -- NULL = base, else session-scoped
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(subject_name, verb, object_name, session_id)
);

CREATE INDEX idx_rel_subject ON entity_relationships(subject_name);
CREATE INDEX idx_rel_object ON entity_relationships(object_name);
CREATE INDEX idx_rel_verb ON entity_relationships(verb);
```

## Two-Pass Algorithm

### Pass 1: Co-occurrence (Already Implemented)

Entities that appear in the same chunk are candidates for relationships.

```sql
-- Find co-occurring entities for entity A
SELECT e2.id, e2.name, COUNT(*) as co_occurrences
FROM chunk_entities ce1
JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
JOIN entities e2 ON ce2.entity_id = e2.id
WHERE ce1.entity_id = :entity_a_id
  AND ce2.entity_id != ce1.entity_id
GROUP BY e2.id, e2.name
ORDER BY co_occurrences DESC;
```

**Output**: List of (entity_a, entity_b, shared_chunk_ids) tuples

### Pass 2: SVO Extraction (Targeted)

For each co-occurring pair, parse their shared chunks to find connecting verbs.

```python
def extract_relationships_for_pair(
    entity_a: Entity,
    entity_b: Entity,
    shared_chunk_ids: list[str],
    nlp: spacy.Language,
) -> list[Relationship]:
    """
    Extract verb relationships between two co-occurring entities.

    Uses spaCy dependency parsing to find Subject-Verb-Object patterns
    where Subject and Object match our known entities.
    """
    relationships = []

    for chunk_id in shared_chunk_ids:
        chunk_text = get_chunk_text(chunk_id)
        doc = nlp(chunk_text)

        for sent in doc.sents:
            # Find entity mentions in sentence
            a_span = find_entity_span(sent, entity_a.name)
            b_span = find_entity_span(sent, entity_b.name)

            if not (a_span and b_span):
                continue

            # Extract connecting verb via dependency path
            verb = find_connecting_verb(sent, a_span, b_span)

            if verb:
                # Determine direction (who is subject, who is object)
                subject, obj = determine_svo_direction(sent, a_span, b_span, verb)

                relationships.append(Relationship(
                    subject_name=subject.name,
                    verb=verb.lemma_,
                    object_name=obj.name,
                    chunk_id=chunk_id,
                    sentence=sent.text,
                    confidence=compute_confidence(verb, a_span, b_span),
                ))

    return relationships
```

### Verb Extraction Logic

```python
def find_connecting_verb(
    sent: spacy.Span,
    span_a: spacy.Span,
    span_b: spacy.Span,
) -> Optional[spacy.Token]:
    """
    Find the verb that connects two entity spans.

    Strategy:
    1. Find the lowest common ancestor (LCA) in dependency tree
    2. If LCA is a verb, return it
    3. Otherwise, traverse up from LCA to find nearest verb
    """
    # Get head tokens of each span
    head_a = span_a.root
    head_b = span_b.root

    # Find ancestors of each
    ancestors_a = set(get_ancestors(head_a))
    ancestors_b = set(get_ancestors(head_b))

    # LCA is first common ancestor
    for ancestor in get_ancestors(head_a):
        if ancestor in ancestors_b:
            lca = ancestor
            break
    else:
        return None

    # If LCA is verb, use it
    if lca.pos_ == "VERB":
        return lca

    # Otherwise find nearest verb ancestor
    for ancestor in get_ancestors(lca):
        if ancestor.pos_ == "VERB":
            return ancestor

    return None


def determine_svo_direction(
    sent: spacy.Span,
    span_a: spacy.Span,
    span_b: spacy.Span,
    verb: spacy.Token,
) -> tuple[Entity, Entity]:
    """
    Determine which entity is subject and which is object.

    Uses dependency labels:
    - nsubj, nsubjpass -> Subject
    - dobj, pobj, attr -> Object
    """
    head_a = span_a.root
    head_b = span_b.root

    # Check dependency relation to verb
    a_is_subject = is_subject_of(head_a, verb)
    b_is_subject = is_subject_of(head_b, verb)

    if a_is_subject and not b_is_subject:
        return (entity_a, entity_b)
    elif b_is_subject and not a_is_subject:
        return (entity_b, entity_a)
    else:
        # Fallback: use word order
        if span_a.start < span_b.start:
            return (entity_a, entity_b)
        else:
            return (entity_b, entity_a)
```

## Execution Strategy

### Option A: Lazy Extraction (Recommended)

Extract relationships on-demand when user views entity details.

| Pros | Cons |
|------|------|
| Fast indexing | First view is slow |
| Only compute what's needed | Repeated computation if not cached |
| Memory efficient | |

```python
@router.get("/{session_id}/entities/{entity_id}/relationships")
async def get_entity_relationships(session_id: str, entity_id: str):
    # Check cache first
    cached = get_cached_relationships(entity_id)
    if cached:
        return cached

    # Get co-occurring entities (Pass 1 - already computed)
    co_occurring = get_co_occurring_entities(entity_id)

    # Extract relationships (Pass 2 - on demand)
    relationships = []
    for other_entity, shared_chunks in co_occurring:
        rels = extract_relationships_for_pair(
            entity_id, other_entity, shared_chunks
        )
        relationships.extend(rels)

    # Cache and return
    cache_relationships(entity_id, relationships)
    return relationships
```

### Option B: Background Extraction

Extract relationships in background after session starts.

```python
async def extract_all_relationships_background(session_id: str):
    """Run as background task after session initialization."""
    entities = get_session_entities(session_id)

    for entity in entities:
        co_occurring = get_co_occurring_entities(entity.id)
        for other, chunks in co_occurring:
            # Skip if already extracted (bidirectional)
            if relationship_exists(entity.id, other.id):
                continue

            rels = extract_relationships_for_pair(entity.id, other, chunks)
            store_relationships(rels)
```

### Option C: Index-Time Extraction

Extract during document indexing (highest latency, most complete).

```python
def index_document(document: Document):
    # Existing: chunk and extract entities
    chunks = chunk_document(document)
    entities = extract_entities(chunks)
    link_chunk_entities(chunks, entities)

    # New: extract relationships for this document's entities
    for chunk in chunks:
        chunk_entities = get_entities_in_chunk(chunk.id)
        for i, entity_a in enumerate(chunk_entities):
            for entity_b in chunk_entities[i+1:]:
                rels = extract_relationships_for_pair(
                    entity_a, entity_b, [chunk.id]
                )
                store_relationships(rels)
```

## Relationship Types

### Verb Categories

Group verbs into semantic categories for UI display:

| Category | Verbs | Example |
|----------|-------|---------|
| **Ownership** | own, have, contain, include | Employee **has** Performance Review |
| **Action** | receive, submit, approve, reject | Manager **approves** Raise |
| **Causation** | determine, affect, influence, cause | Rating **determines** Merit Increase |
| **Temporal** | precede, follow, trigger | Review **precedes** Compensation Decision |
| **Association** | relate, associate, link, connect | Salary Band **relates to** Job Level |

```python
VERB_CATEGORIES = {
    "ownership": {"own", "have", "contain", "include", "hold", "possess"},
    "action": {"receive", "submit", "approve", "reject", "send", "create", "update", "delete"},
    "causation": {"determine", "affect", "influence", "cause", "drive", "impact"},
    "temporal": {"precede", "follow", "trigger", "initiate", "complete"},
    "association": {"relate", "associate", "link", "connect", "correspond"},
}

def categorize_verb(verb_lemma: str) -> str:
    for category, verbs in VERB_CATEGORIES.items():
        if verb_lemma in verbs:
            return category
    return "other"
```

## API Response

### Entity Detail with Relationships

```json
{
  "id": "ent_123",
  "name": "Performance Review",
  "type": "concept",
  "sources": ["schema:hr.performance_reviews", "business_rules"],
  "related_entities": [
    {
      "entity": {"id": "ent_456", "name": "Employee", "type": "concept"},
      "relationships": [
        {"verb": "receive", "category": "action", "direction": "object", "count": 5},
        {"verb": "have", "category": "ownership", "direction": "subject", "count": 3}
      ]
    },
    {
      "entity": {"id": "ent_789", "name": "Rating", "type": "column"},
      "relationships": [
        {"verb": "determine", "category": "causation", "direction": "subject", "count": 4}
      ]
    }
  ]
}
```

### UI Display

```
Performance Review
━━━━━━━━━━━━━━━━━━

Sources: schema:hr.performance_reviews, business_rules

Relationships:
┌─────────────┬──────────────┬───────────────────┬───────┐
│ Related To  │ Relationship │ Direction         │ Count │
├─────────────┼──────────────┼───────────────────┼───────┤
│ Employee    │ receives     │ Employee → Review │   5   │
│ Employee    │ has          │ Review → Employee │   3   │
│ Rating      │ determines   │ Rating → Review   │   4   │
│ Manager     │ approves     │ Manager → Review  │   2   │
└─────────────┴──────────────┴───────────────────┴───────┘
```

## Implementation Plan

### Phase 1: Schema and Storage

1. Add `entity_relationships` table to `vector_store.py`
2. Add relationship CRUD methods
3. Add migration for existing databases

### Phase 2: Extraction Logic

1. Implement `find_connecting_verb()` in `discovery/relationship_extractor.py`
2. Implement `determine_svo_direction()`
3. Implement `extract_relationships_for_pair()`
4. Add verb categorization

### Phase 3: API Integration

1. Add lazy extraction endpoint
2. Add caching layer
3. Update entity detail response to include relationships

### Phase 4: UI Integration

1. Update EntityAccordion to show relationships
2. Add relationship table/graph visualization
3. Enable click-through to related entities

## Performance Considerations

| Operation | Cost | Mitigation |
|-----------|------|------------|
| Co-occurrence query | O(chunks) | Already indexed, fast |
| SVO parsing | O(sentences) | Scope to shared chunks only |
| spaCy load | ~500MB RAM | Already loaded for NER |
| Relationship storage | O(pairs) | Deduplicate, limit per pair |

### Limits

```python
MAX_RELATIONSHIPS_PER_ENTITY = 50      # Cap stored relationships
MAX_CO_OCCURRING_PAIRS = 20            # Only process top N pairs
MAX_CHUNKS_PER_PAIR = 10               # Limit chunks to parse per pair
```

## Testing

```python
def test_svo_extraction():
    text = "The manager approves the performance review for the employee."

    relationships = extract_relationships(text, ["manager", "performance review", "employee"])

    assert len(relationships) == 2
    assert relationships[0] == ("manager", "approve", "performance review")
    assert relationships[1] == ("performance review", "for", "employee")  # prepositional


def test_passive_voice():
    text = "The raise was approved by the manager."

    relationships = extract_relationships(text, ["raise", "manager"])

    # Should correctly identify manager as subject despite passive voice
    assert relationships[0] == ("manager", "approve", "raise")
```
