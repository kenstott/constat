# Golden Questions: Domain Regression Testing

> **Status:** Design. No implementation yet.

## Problem

A domain's discovery pipeline — entity extraction, glossary generation, relationship extraction, grounding — is non-deterministic. Model updates, config changes, new documents, or schema migrations can silently degrade quality. There's no way to know if "customer" still resolves to the right table, or if the taxonomy still holds, without manually re-checking.

## Core Idea

Per-domain curated question sets with structured assertions at each pipeline stage. Run on demand or CI. No full LLM round-trip required for most checks — the expensive path is opt-in.

```
golden_questions:
  - question: "What are the top 5 customers by revenue?"
    expect:
      entities: [customer, revenue, order]
      grounding:
        - entity: customer
          resolves_to: ["schema:sales.customers"]
        - entity: revenue
          resolves_to: ["schema:sales.order_items"]
      relationships:
        - subject: customer
          verb: PLACES
          object: order
      glossary:
        - name: customer
          has_definition: true
          domain: sales-analytics
```

## What Gets Tested

Five layers, each independently assertable:

| Layer | What | Cost | Speed |
|-------|------|------|-------|
| **Entity extraction** | Expected entities appear in NER output | Free | Fast |
| **Grounding** | Entities resolve to expected document/schema sources | Free | Fast |
| **Glossary** | Terms have definitions, correct domain, parent hierarchy | Free | Fast |
| **Relationships** | Expected SVO triples exist with minimum confidence | Free | Fast |
| **End-to-end** | LLM generates plan, executes, answer matches reference | LLM call | Slow |

The first four layers are pure database lookups against the existing vector store. No LLM calls, no cost. End-to-end is the expensive one and should be opt-in per question.

## Storage

Golden questions live in the domain YAML alongside the resources they test:

```yaml
# domains/sales-analytics.yaml
name: Sales Analytics
databases:
  sales:
    uri: sqlite:///data/sales.db

golden_questions:
  - question: "What are the top 5 customers by revenue?"
    tags: [smoke, revenue]
    expect:
      entities: [customer, revenue, order]
      grounding:
        - entity: customer
          resolves_to: ["schema:sales.customers"]
      relationships:
        - subject: customer
          verb: PLACES
          object: order
      glossary:
        - name: customer
          has_definition: true
          parent: account holder

  - question: "How many orders were placed last month?"
    tags: [smoke]
    expect:
      entities: [order]
      grounding:
        - entity: order
          resolves_to: ["schema:sales.orders"]

  - question: "Which product categories have declining sales?"
    tags: [deep]
    expect:
      entities: [product, category, sale]
      end_to_end:
        plan_min_steps: 2
        result_type: dataframe
        result_contains: ["category"]
```

## Assertion Model

```python
@dataclass
class GoldenQuestion:
    question: str
    tags: list[str] = field(default_factory=list)
    expect: GoldenExpectations

@dataclass
class GoldenExpectations:
    entities: list[str] = field(default_factory=list)
    grounding: list[GroundingAssertion] = field(default_factory=list)
    relationships: list[RelationshipAssertion] = field(default_factory=list)
    glossary: list[GlossaryAssertion] = field(default_factory=list)
    end_to_end: EndToEndAssertion | None = None

@dataclass
class GroundingAssertion:
    entity: str
    resolves_to: list[str]          # Any of these document_names

@dataclass
class RelationshipAssertion:
    subject: str
    verb: str
    object: str
    min_confidence: float = 0.5

@dataclass
class GlossaryAssertion:
    name: str
    has_definition: bool = True
    domain: str | None = None
    parent: str | None = None

@dataclass
class EndToEndAssertion:
    plan_min_steps: int = 1
    result_type: str | None = None   # dataframe, scalar, chart
    result_contains: list[str] = field(default_factory=list)
    semantic_match: str | None = None  # Reference answer for similarity
```

## Runner

```
constat test [--domain sales-analytics] [--tags smoke] [--e2e]
```

### Execution Flow

```
Load domain config
  → Parse golden_questions section
  → For each question:
      1. Entity check: query entities table, assert expected names exist
      2. Grounding check: join chunk_entities → embeddings, assert document_names
      3. Glossary check: query glossary_terms, assert definitions/parents/domains
      4. Relationship check: query entity_relationships, assert SVO triples
      5. [--e2e only] End-to-end: submit question to session, assert plan + result
  → Aggregate results per question, per layer
  → Report pass/fail with details
```

### Result Format

```
sales-analytics: 3 questions, 2 passed, 1 failed

  ✓ "What are the top 5 customers by revenue?"
    entities: 3/3  grounding: 2/2  relationships: 1/1  glossary: 1/1

  ✓ "How many orders were placed last month?"
    entities: 1/1  grounding: 1/1

  ✗ "Which product categories have declining sales?"
    entities: 2/3  grounding: 1/1
    FAIL entity: missing "category" (found: product, sale)
```

## Implementation Plan

### Phase 1: Config + Fast Assertions

1. Add `golden_questions` to domain config schema (`constat/core/config.py`)
2. Add assertion dataclasses (`constat/testing/models.py`)
3. Runner that checks entities, grounding, glossary, relationships against live vector store
4. CLI command: `constat test`
5. JSON + text output formatters

### Phase 2: End-to-End

6. Session-based runner that submits questions through the full pipeline
7. Plan structure assertions (step count, step types)
8. Result assertions (type, column presence, row count)
9. Optional semantic similarity check against reference answers (uses embeddings, no extra LLM call)

### Phase 3: CI Integration

10. Exit code reflects pass/fail for CI pipelines
11. JUnit XML output for test reporters
12. `--baseline` flag to snapshot current state as golden expectations
13. `--update` flag to regenerate expectations from current pipeline output

## Design Decisions

**Why per-domain, not global?** Each domain has its own schema, documents, and glossary. A question only makes sense in the context of the domain it was written for. Cross-domain questions are a domain composition problem, not a testing problem.

**Why structured assertions, not semantic similarity?** Semantic similarity is fuzzy and requires a threshold. "customer resolves to sales.customers" is binary — it either does or doesn't. Structured assertions give precise, actionable failure messages. Semantic similarity is available as an opt-in for end-to-end answer comparison.

**Why not pytest?** The runner is a first-class CLI command, not a test framework plugin. Golden questions are authored by domain builders and SMEs, not developers. The output is a domain health report, not a test suite. That said, nothing prevents wrapping it in pytest for CI.

**Why YAML, not a separate file?** Golden questions are part of the domain definition. They test the domain's resources. Keeping them colocated makes it obvious what questions belong to what domain, and they travel with the domain config when exported or shared.
