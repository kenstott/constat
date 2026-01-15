# Entity Extraction Design

## Overview

Enhance document retrieval by extracting entities and relationships at index time, then enriching semantic search results with entity context. The LLM receives richer context without needing to make separate tool calls to explore an entity graph.

## Goals

1. Improve answer quality by providing entity context alongside chunk text
2. Enable the LLM to understand relationships between concepts mentioned in documents
3. Keep the query path simple - no additional tool calls during inference

## Architecture

### Current Flow
```
query → semantic search → chunks → LLM
```

### Proposed Flow
```
query → semantic search → chunks → enrich with entities → LLM
```

## Index Time Processing

### 1. Existing: Chunk and Embed
```
Document → Chunks → Embeddings → Vector Store
```

### 2. New: Entity Extraction
```
Document → LLM Extraction → Entities + Relationships → SQLite
```

### 3. New: Entity-Chunk Mapping
```
For each chunk, record which entities are mentioned
```

## Database Schema

```sql
-- Entities extracted from documents
CREATE TABLE entities (
    id TEXT PRIMARY KEY,              -- hash of (name, type) for dedup
    name TEXT NOT NULL,               -- "Q4 Revenue", "John Smith", "Refund Policy"
    type TEXT NOT NULL,               -- person, metric, policy, term, org, date, etc.
    canonical_name TEXT,              -- normalized form for matching
    description TEXT,                 -- LLM-generated summary (1 sentence)
    first_seen_doc TEXT,              -- document where first extracted
    mention_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Relationships between entities
CREATE TABLE relationships (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES entities(id),
    target_id TEXT NOT NULL REFERENCES entities(id),
    relationship TEXT NOT NULL,       -- "owns", "reports_to", "defined_in", "measures"
    confidence REAL DEFAULT 1.0,
    evidence TEXT,                    -- chunk text that supports this relationship
    doc_name TEXT,                    -- source document
    UNIQUE(source_id, target_id, relationship)
);

-- Entity mentions in chunks (for retrieval enrichment)
CREATE TABLE entity_mentions (
    entity_id TEXT NOT NULL REFERENCES entities(id),
    doc_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    PRIMARY KEY (entity_id, doc_name, chunk_index)
);

-- Indexes
CREATE INDEX idx_entities_type ON entities(type);
CREATE INDEX idx_entities_canonical ON entities(canonical_name);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
CREATE INDEX idx_mentions_doc ON entity_mentions(doc_name);
```

## Extraction Prompt

Used once per document during indexing:

```
Extract entities and relationships from this document.

Document: {doc_name}
Content:
{doc_content}

Extract:
1. ENTITIES: Named things (people, metrics, policies, terms, orgs, systems)
2. RELATIONSHIPS: How entities relate (owns, defines, reports_to, measures, etc.)

Output JSON:
{
  "entities": [
    {"name": "Q4 Revenue Target", "type": "metric", "description": "Quarterly revenue goal"},
    {"name": "Sarah Chen", "type": "person", "description": "VP of Sales"},
    ...
  ],
  "relationships": [
    {"source": "Sarah Chen", "relationship": "owns", "target": "Q4 Revenue Target"},
    ...
  ]
}

Focus on business-relevant entities. Skip generic terms.
Keep descriptions to one sentence.
```

## Query Time Enrichment

### Retrieval Function

```python
def search_documents(self, query: str, limit: int = 5) -> str:
    """Search documents and return chunks enriched with entity context."""

    # 1. Semantic search (existing)
    chunks = self._vector_search(query, limit)

    # 2. Enrich each chunk with entity context
    enriched_results = []
    for chunk in chunks:
        entities = self._get_chunk_entities(chunk.doc_name, chunk.index)
        entity_context = self._format_entity_context(entities)

        enriched_results.append({
            "document": chunk.doc_name,
            "text": chunk.text,
            "entities": entity_context,
        })

    # 3. Format for LLM
    return self._format_search_results(enriched_results)
```

### Entity Context Query

```sql
SELECT e.name, e.type, e.description,
       GROUP_CONCAT(r.relationship || ' ' || t.name) as relationships
FROM entity_mentions m
JOIN entities e ON m.entity_id = e.id
LEFT JOIN relationships r ON r.source_id = e.id
LEFT JOIN entities t ON r.target_id = t.id
WHERE m.doc_name = ? AND m.chunk_index = ?
GROUP BY e.id
LIMIT 5  -- Max entities per chunk
```

## Output Format

### Before (current)
```
[Chunk 1]
Sarah mentioned the Q4 revenue target is at risk due to pipeline issues.

[Chunk 2]
The forecast shows we're 15% behind the annual goal.
```

### After (with entity enrichment)
```
[Chunk 1]
Sarah mentioned the Q4 revenue target is at risk due to pipeline issues.

Entities:
- Sarah Chen (person): VP of Sales, owns Q4 Revenue Target
- Q4 Revenue Target (metric): $4.2M quarterly goal, measured by Sales Pipeline

[Chunk 2]
The forecast shows we're 15% behind the annual goal.

Entities:
- Annual Revenue Goal (metric): $15M FY target, parent of Q4 Revenue Target
```

## Constraints

To keep context size bounded:
- **Max 5 entities per chunk** in output
- **Direct relationships only** - no graph traversal
- **1 sentence descriptions** - aggressive summarization
- **Max 3 relationships per entity** in output

## Entity Types

Standard types to extract:
- `person` - Named individuals
- `org` - Organizations, teams, departments
- `metric` - KPIs, measurements, targets
- `policy` - Rules, procedures, guidelines
- `term` - Domain-specific terminology
- `system` - Software, tools, platforms
- `date` - Specific dates or periods
- `location` - Places, regions

## Implementation Phases

### Phase 1: Entity Extraction Only
- Extract entities during document indexing
- Map entities to chunks
- Enrich retrieval results with entity names + descriptions
- No relationships yet

### Phase 2: Relationships
- Extract relationships during indexing
- Include key relationships in entity context
- Direct relationships only

### Phase 3: Deduplication & Merging
- Canonical name matching for entity dedup
- Merge entity descriptions across documents
- Confidence scoring for relationships

## Integration Points

### Document Indexing (`discovery/doc_tools.py`)
- Add `_extract_entities()` after chunking
- Store entities, relationships, mentions

### Document Search (`discovery/doc_tools.py`)
- Modify `search_documents()` to join entity context
- Format enriched results

### Storage
- New SQLite tables in existing document store
- Or separate `~/.constat/entities.db`

## Open Questions

1. **Extraction granularity**: Per-document or per-chunk extraction?
   - Per-document: Fewer LLM calls, but entity-chunk mapping is fuzzy
   - Per-chunk: More accurate mapping, but more LLM calls and potential duplicates

2. **Entity deduplication**: How to handle "Sarah" vs "Sarah Chen" vs "S. Chen"?
   - Canonical name normalization
   - LLM-based entity resolution (expensive)

3. **Relationship confidence**: How to score relationship reliability?
   - Multiple mentions = higher confidence
   - Explicit statements vs implied relationships
