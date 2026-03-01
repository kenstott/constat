# Relationship Extraction Architecture

> **Status:** Implemented. Two-phase extraction (spaCy + LLM refinement) is in production. Relationships are extracted during session initialization and stored in `entity_relationships`.

## Overview

Extract semantic relationships between entities using a two-phase approach:
1. **Phase 1 (spaCy)**: SVO candidate extraction from co-occurring entity pairs — fast, cheap
2. **Phase 2 (LLM)**: Validates/rejects candidates, fixes verbs, infers implicit relationships — accurate

Co-occurrence acts as a filter: only entity pairs sharing chunks are considered.

```
┌─────────────────────────────────────────────────────────────┐
│                    Document Indexing                         │
├─────────────────────────────────────────────────────────────┤
│  Chunks  ──►  Entities  ──►  Co-occurrence  ──►  spaCy SVO │
│  (text)      (NER/regex)    (chunk_entities)      │         │
│                                                    ▼         │
│                                              LLM Refinement  │
│                                              (batch verify)  │
└─────────────────────────────────────────────────────────────┘
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

Relationships are extracted during session initialization via `extract_relationships()`:

1. Find co-occurring entity pairs (shared chunks)
2. For top pairs (up to `MAX_CO_OCCURRING_PAIRS=50`), run spaCy SVO extraction
3. Batch pairs to LLM for refinement (up to `MAX_LLM_PAIRS=75`, batches of `LLM_BATCH_SIZE=8`)
4. Store validated relationships in `entity_relationships`

### Limits

```python
MAX_RELATIONSHIPS_PER_ENTITY = 50
MAX_CO_OCCURRING_PAIRS = 50
MAX_CHUNKS_PER_PAIR = 10
LLM_BATCH_SIZE = 8
MAX_LLM_PAIRS = 75
MAX_EXCERPTS_PER_PAIR = 5
MAX_EXCERPT_LENGTH = 500
```

## Relationship Types

### Verb Vocabulary

All verbs use Cypher-standard UPPER_SNAKE_CASE for graph consistency:

```
(Manager)-[:MANAGES]->(Employee)
```

### Verb Categories

| Category | Preferred Verbs | Example |
|----------|----------------|---------|
| **hierarchy** | MANAGES, REPORTS_TO | Manager MANAGES Employee |
| **action** | CREATES, PROCESSES, APPROVES, PLACES | Manager APPROVES Raise |
| **flow** | SENDS, RECEIVES, TRANSFERS | System SENDS Notification |
| **causation** | DRIVES, REQUIRES, ENABLES | Rating DRIVES Merit Increase |
| **temporal** | PRECEDES, FOLLOWS, TRIGGERS | Review PRECEDES Compensation Decision |
| **association** | REFERENCES, WORKS_IN, PARTICIPATES_IN | Employee WORKS_IN Department |
| **ownership** | (LLM prompt category) | Employee HAS_ONE Performance Review |
| **other** | Anything not in preferred set | |

### Hierarchy Verb Filtering

Verbs that overlap with the taxonomy are filtered when the pair already has a parent-child edge:

```python
_HIERARCHY_VERBS = {"HAS", "HAS_ONE", "HAS_KIND", "HAS_MANY", "USES"}
```

Bare `HAS` is never allowed — it must be qualified via replacement:

```python
_VERB_REPLACEMENTS = {
    "HAS": "HAS_ONE",
    "CONTAINS": "HAS_MANY",
    "BELONGS_TO": "HAS_ONE",    # direction swapped
    "IS_TYPE_OF": "HAS_KIND",   # direction swapped
}
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

## LLM Refinement

The LLM sees co-occurring pairs with text excerpts and returns validated relationships:

```
For each pair of entities, analyze the excerpts and identify relationships.

Return JSON array with:
- subject: one of the two entity names
- object: the other entity name
- verb: Cypher UPPER_SNAKE_CASE (e.g., "MANAGES", "WORKS_IN")
- verb_category: ownership | hierarchy | action | flow | causation | temporal | association | other
- evidence: brief quote from excerpt
- confidence: high | medium | low
```

Confidence maps: `high=0.95, medium=0.8, low=0.6`.

### Non-Verb Filtering

Technical nouns that spaCy mis-tags as verbs are filtered:

```python
_NON_VERBS = {"endpoint", "api", "query", "type", "field", "schema", "table",
              "column", "database", "server", "client", "model", ...}
```
