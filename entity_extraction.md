# Entity Extraction for Plan Context Enhancement

## Goal

Augment vector search with entity awareness to give the LLM richer context during plan development. Enable exploration of related information through entity linkage.

## Schema

```sql
-- Extracted entities with optional embeddings for semantic search
CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,                    -- e.g., 'table', 'column', 'concept', 'business_term'
    embedding FLOAT[],            -- optional: for semantic entity search
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Many-to-many link between chunks and entities
CREATE TABLE chunk_entities (
    chunk_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    mention_count INTEGER DEFAULT 1,
    confidence FLOAT,
    PRIMARY KEY (chunk_id, entity_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id),
    FOREIGN KEY (entity_id) REFERENCES entities(id)
);

-- Index for efficient entity lookups
CREATE INDEX idx_chunk_entities_entity ON chunk_entities(entity_id);
```

## Extraction Approach

Extract entities during document ingestion:

1. **Schema entities** - Tables, columns, relationships from database metadata (high confidence)
2. **Named entities** - NER for business terms, proper nouns (medium confidence)
3. **Domain concepts** - LLM-assisted extraction for domain-specific terms (configurable)

```python
class EntityExtractor:
    def extract(self, chunk: Chunk) -> list[Entity]:
        """Extract entities from a chunk during ingestion."""
        pass

    def link_chunk(self, chunk_id: str, entities: list[Entity]) -> None:
        """Create chunk_entities records."""
        pass
```

## Search Enrichment

When similarity search returns chunks, include entity context:

```python
@dataclass
class EnrichedChunk:
    chunk: Chunk
    score: float
    entities: list[Entity]  # entities mentioned in this chunk

def search(query: str, k: int = 10) -> list[EnrichedChunk]:
    """Vector search with entity enrichment."""
    chunks = vector_store.search(query, k)
    return [
        EnrichedChunk(
            chunk=c,
            score=c.score,
            entities=get_entities_for_chunk(c.id)
        )
        for c in chunks
    ]
```

## LLM Tool: explore_entity

Single tool for the LLM to explore related context:

```python
def explore_entity(entity_name: str, limit: int = 5) -> list[EnrichedChunk]:
    """
    Find chunks mentioning the given entity.

    Use when the LLM notices a relevant entity and wants more context.
    Returns chunks ordered by relevance (mention count, recency).
    """
    entity = find_entity_by_name(entity_name)
    if not entity:
        return []

    chunk_ids = get_chunks_for_entity(entity.id, limit)
    return [enrich_chunk(cid) for cid in chunk_ids]
```

Tool definition for LLM:
```json
{
    "name": "explore_entity",
    "description": "Find additional context about an entity (table, concept, term) mentioned in the retrieved chunks. Use when you need more information about something specific.",
    "parameters": {
        "entity_name": {
            "type": "string",
            "description": "Name of the entity to explore"
        }
    }
}
```

## Workflow

```
1. User requests a plan
2. Vector search returns chunks + entities for each
3. LLM reads chunks, sees entities like [Customer, Order, PricingEngine]
4. LLM decides PricingEngine needs more context
5. LLM calls explore_entity("PricingEngine")
6. Returns additional chunks mentioning PricingEngine
7. LLM incorporates into plan
```

## Relationships: Deferred

Explicit relationship extraction (X *owns* Y, X *depends on* Z) is deferred. Rationale:

- Co-occurrence in chunks provides implicit relationship signal
- Relationship extraction is error-prone without fine-tuning
- LLMs infer relationships well from raw text
- Simpler schema, faster iteration

If needed later, add:
```sql
CREATE TABLE entity_relationships (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    rel_type TEXT,              -- 'references', 'contains', 'depends_on'
    confidence FLOAT,
    PRIMARY KEY (source_id, target_id, rel_type)
);
```

## Entity Types

Initial supported types:

| Type | Source | Example |
|------|--------|---------|
| `table` | Schema introspection | `customers`, `orders` |
| `column` | Schema introspection | `customer_id`, `created_at` |
| `concept` | LLM extraction | `churn rate`, `lifetime value` |
| `business_term` | NER / glossary | `SKU`, `fulfillment` |

## Open Questions

- [ ] Entity deduplication strategy (fuzzy matching? embeddings?)
- [ ] Confidence threshold for linking
- [ ] Whether to embed entities for semantic entity search
- [ ] Rate limiting on explore_entity calls