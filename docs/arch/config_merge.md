# 5-Tier Configuration Architecture Plan

## Overview

Hierarchical configuration with five precedence levels:

1. **System** (lowest) — core + llm
2. **System Domain** — core + llm
3. **User** — core + preferences
4. **User Domain** — core + llm
5. **Session** (highest) — core

**Merge order:** system → system domain → user → user domain → session

### Core Config

Every tier includes the **core config** shape:

| Section    | Format                        | Merge Behavior                      |
|-----------|-------------------------------|-------------------------------------|
| skills     | Directory-based (SKILL.md)    | Later tier replaces by dir name     |
| sources    | `$ref` (apis, databases, documents) | Deep merge by key            |
| rights     | `$ref`                        | Deep merge by key                   |
| roles      | `$ref`                        | Deep merge by key                   |
| facts      | `$ref`                        | Deep merge by key                   |
| learnings  | `$ref`                        | Deep merge by key                   |
| glossary   | `$ref`                        | Deep merge by key                   |
| relationships | `$ref`                     | Deep merge by key                   |

**Additional sections by tier:**

| Section     | Available at         |
|------------|---------------------|
| llm         | system, system domain, user domain |
| preferences | user only            |

### Fundamental Rules

- **No arrays.** Everything is maps. Maps always deep merge.
- **`null` deletes.** Setting a key to `null` removes it from that tier, revealing the lower-tier value (or nothing).
- **Skills are directories** following the Anthropic model (`skill-name/SKILL.md`).
- **Everything else uses `$ref`** for composition within YAML files.
- **`$ref` resolves before merge.** Each tier's refs are fully resolved, then tiers merge in precedence order.
- **Resolve once, consume flat.** Tiers are merged into a single `ResolvedConfig` at startup and on any tier mutation. All downstream code sees one flat config — no tiers, no `$ref`, no `null` sentinels. Tier machinery is internal to the loader.

## Key Design Decisions

### 1. Domains (formerly Projects)

A **domain** is a bounded business context — a recursive container with its own sources, roles, skills, facts, learnings, glossary, and rights. Domains replace "projects" to better reflect that they represent permanent business infrastructure, not temporary deliverables.

**Domains can compose sub-domains.** A parent domain merges its children's configs (deep merge, parent overrides children), then the tier system applies on top.

```
<config_dir>/                           # System tier
├── config.yaml                         # core + llm (uses $ref throughout)
├── skills/                             # System skills (directory-based)
│   └── data-analysis/
│       └── SKILL.md
└── domains/                            # System domains
    └── customer/                       # Parent domain
        ├── config.yaml                 # core + llm ($ref)
        ├── skills/
        └── domains/                    # Sub-domains (recursive)
            ├── retail/
            │   └── config.yaml
            └── enterprise/
                └── config.yaml

.constat/<user_id>/                     # User tier
├── config.yaml                         # core + preferences ($ref)
├── preferences.yaml                    # User preferences (referenced from config)
├── skills/                             # User skills (directory-based)
└── domains/                            # User domains
    └── customer/                       # User domain tier (same name = overlay)
        ├── config.yaml                 # core + llm ($ref)
        └── skills/

.constat/<user_id>/sessions/<id>/       # Session tier
├── config.yaml                         # core only ($ref)
└── skills/                             # Session skills
```

**Domain config includes identity:**

```yaml
# domains/customer/config.yaml
owner: customer-team@acme.com
definition: "Customer lifecycle data and analytics"

sources:
  databases:
    crm:
      $ref: ./sources/crm.yaml

glossary:
  $ref: ./glossary.yaml

roles:
  customer_analyst:
    $ref: ./roles/analyst.yaml
```

**Sub-domain merge order:** Sub-domain configs merge alphabetically, then parent domain overlays. This happens before the tier merge.

```
sub-domain configs (merged by name)
  → parent domain config overlays
    → tier merge continues (system → system domain → user → ...)
```

**Domains are managed manually** (file-based). Too structural for UI CRUD in v1.

### 2. $ref Composition

Each `config.yaml` is thin — mostly `$ref` pointers:

```yaml
# system config.yaml
sources:
  databases:
    hr:
      $ref: ./sources/hr.yaml
    warehouse:
      $ref: ./sources/warehouse.yaml
  apis:
    stripe:
      $ref: ./sources/stripe-api.yaml
  documents:
    policy:
      $ref: ./sources/compensation-policy.yaml

roles:
  analyst:
    $ref: ./roles/analyst.yaml

rights:
  execute_sql:
    $ref: ./rights/sql-access.yaml

facts:
  $ref: ./facts.yaml

learnings:
  $ref: ./learnings.yaml

glossary:
  $ref: ./glossary.yaml

llm:
  $ref: ./llm.yaml
```

### 3. Rights Model (OPA-Compatible Subset)

Permission map structured as valid OPA **data**. Decision (`allow`) is separated from conditions (`constraints`). v1 evaluates in Python; later can drop in OPA sidecar with Rego policies against the same data — zero migration.

```yaml
rights:
  execute_sql:
    allow: true
    constraints:
      max_rows: 10000
      allowed_databases:
        hr: {}
        warehouse: {}
  write_files:
    allow: false
  access_api:
    allow: true
    constraints:
      rate_limit: 100
      allowed_apis:
        stripe: {}
```

**v1 evaluation:** Simple Python — `if rights[action].allow and check_constraints(...)`.

**Future:** OPA sidecar evaluates same YAML data with Rego for complex cross-cutting rules (e.g., time-based access, role intersections).

Rights should only **narrow** as you go down the precedence chain. A session can tighten a right but not loosen one set by a higher tier.

### 4. Learnings: Two-Tier Model

Learnings have two tiers — raw corrections and compacted rules — with a folding process that promotes instances into generalized patterns.

#### Raw Learnings (Corrections)

Individual corrections captured during sessions. Each records what went wrong and how to fix it:

```yaml
learnings:
  corrections:
    learn_001:
      category: codegen_error
      correction: "GraphQL API returns data.customers, not response.data.customers"
      context:
        error: "KeyError: 'response'"
        code_snippet: "df = response.data.customers"
      source: auto_capture
      created: "2024-01-15T10:30:00Z"
      applied_count: 0
      promoted_to: null

    learn_002:
      category: user_correction
      correction: "Always aggregate salary data by department, never show individual salaries"
      context:
        query: "show me employee salaries"
      source: explicit_command
      created: "2024-01-16T09:00:00Z"
      applied_count: 0
      promoted_to: rule_001
```

**Categories:** `user_correction`, `codegen_error`, `external_api_error`, `http_error`, `nl_correction`

**Sources:** `auto_capture` (system detected error), `explicit_command` (user corrected), `nl_detection` (detected from conversation)

#### Rules (Compacted Patterns)

Generalized patterns derived from multiple corrections. Rules are what the LLM actually uses during query generation:

```yaml
learnings:
  rules:
    rule_001:
      category: user_correction
      summary: "Aggregate salary and compensation data by department with minimum 5 employees. Never expose individual salary figures."
      confidence: 0.85
      source_learnings:
        learn_002: {}
        learn_005: {}
        learn_008: {}
      tags:
        salary: {}
        compensation: {}
        privacy: {}
      applied_count: 12
      created: "2024-01-20T14:00:00Z"
```

#### Compaction (Folding Corrections into Rules)

The compaction process uses the LLM to promote raw corrections into rules:

1. **Group** — LLM analyzes unpromoted corrections and groups by common pattern
2. **Check overlap** — compares groups against existing rules
3. **Strengthen or create** — if overlap found, strengthens existing rule with new evidence; otherwise creates new rule
4. **Archive** — source corrections marked as `promoted_to: <rule_id>`

**Auto-compact triggers** when unpromoted corrections exceed a threshold (currently 50).

**Confidence threshold:** Rules below 0.60 confidence are not applied.

#### Tier Behavior

Corrections are always session- or user-scoped. Rules can exist at any tier:

- **Session corrections** — captured during a session, promoted to user rules via compaction
- **User rules** — generalized patterns for this user
- **Domain rules** — shared rules for a business domain (promoted by domain owner)
- **System rules** — organization-wide rules (promoted by admin)

Rules merge by key across tiers (deep merge). A lower tier can override a rule's `confidence` or add `tags`, but the `summary` from the highest-precedence tier wins.

### 5. Glossary: Business Terms with Physical Grounding

The glossary maps business concepts to definitions, relationships, and optionally to physical resources. It serves as the cross-source Rosetta Stone — giving the LLM authoritative definitions instead of guesses.

#### Structure

```yaml
glossary:
  # Metrics with classification
  churn_rate:
    definition: "Percentage of customers who cancelled their subscription in a period"
    is_a: customer_metric
    aliases:
      attrition_rate: {}
      cancellation_rate: {}

  customer_metric:
    definition: "Metrics that measure customer behavior and lifecycle"
    aliases:
      customer_kpi: {}

  # Dimensional hierarchy
  day:
    definition: "Calendar day"
    part_of: month
  month:
    definition: "Calendar month"
    part_of: quarter
  quarter:
    definition: "Fiscal or calendar quarter"
    part_of: year
  year:
    definition: "Calendar or fiscal year"

  # Composition hierarchy
  line_item:
    definition: "Individual product entry on an order"
    part_of: order
  order:
    definition: "A customer purchase transaction"
    part_of: customer
    is_a: transaction

  # Cross-domain inheritance
  customer_lifetime_value:
    definition: "Total predicted revenue from a customer over the relationship"
    is_a: customer_metric
    extends:
      finance.revenue_metric: {}
      forecasting.predictive_metric: {}
```

#### Three Relationship Types

| Relationship | Meaning | Cardinality | Example |
|---|---|---|---|
| `is_a` | Classification within a taxonomy | Single parent (string) | `churn_rate` is a `customer_metric` |
| `part_of` | Composition / dimensional hierarchy | Single parent (string) | `day` is part of `month` |
| `extends` | Cross-domain inheritance | Multiple parents (map) | CLV extends `finance.revenue_metric` + `forecasting.predictive_metric` |

`is_a` and `part_of` are strings pointing to another glossary key. `extends` is a map (multiple cross-domain parents, no-arrays rule). A term can have all three.

Cross-domain `extends` is natural (different domains answer different questions). Within-domain multiple inheritance is a smell — flag for review.

#### Optional Physical Bindings

Bindings connect glossary terms to specific physical resources. Without bindings, the LLM uses semantic search + best guess. With bindings, the LLM gets authoritative mappings.

Bindings use `source/resource` as the key. The glossary always defines a **single concept** — cardinality (one row vs. many, single document vs. collection) is a property of the resource, not the term. The LLM already knows this from schema metadata, API specs, and document config.

Binding maps are recursive — they mirror the source's own structure (flat for SQL, nested for JSON/NoSQL/API responses). Leaf nodes are either `{}` (field exists here) or `{ values: {...} }` (field has enum mappings). `values` is only for enum/coded fields where physical codes differ from business terms.

```yaml
glossary:
  churn_rate:
    definition: "Percentage of customers who cancelled in a period"
    is_a: customer_metric
    aliases:
      attrition_rate: {}
    bindings:
      # SQL — flat field map
      sales_db/customers:
        status:
          values:
            churned: "C"
            active: "A"
            inactive: "I"
        cancelled_at: {}

      # NoSQL — nested, mirrors document structure
      mongo_db/customer_events:
        metadata:
          type:
            values:
              churned: "account_closed"

      # API — mirrors response shape
      stripe/customers:
        data:
          status:
            values:
              churned: "canceled"

      # Document — reference only
      kpi_guide:
        section: "Retention Metrics"
```

#### Three Tiers of Glossary Usefulness

| Level | What's provided | LLM behavior |
|---|---|---|
| Definition only | Term + meaning + aliases | Semantic search + best guess |
| Definition + hierarchy | + `is_a`, `part_of`, `extends` | Better discovery, cross-domain context |
| Definition + bindings | + resource/field/enum mappings | Authoritative — no guessing |

All three are useful. Bindings are only needed where guessing gets it wrong — terms that are cross-referenced across systems, contested/ambiguous, or where retrieval alone isn't reliable.

#### Integration with Discovery Pipeline

Glossary integrates with the existing discovery pipeline at three levels: NER entity extraction, vector search, and LLM tool results.

##### 1. NER Entity Extraction (Session Start)

The `EntityExtractor` uses spaCy with a custom `EntityRuler` that matches terms from three sources. Glossary provides the third:

```python
EntityExtractor(
    schema_terms=["customers", "orders", "status"],        # from schema metadata
    api_terms=["customers_list", "get_order"],              # from API specs
    business_terms=["churn_rate", "attrition_rate",         # glossary keys + alias keys
                    "customer_metric", "customer_kpi",
                    "active_customer", "current_customer",
                    "purchase", "buy", "order",             # relationship keys + alias keys
                    "approval", "approve", "authorize"],
)
```

The EntityRuler already handles variant forms (singular/plural, underscore/space/camelCase). So when a user query contains "attrition rate" or "AttritionRate", it gets detected as a TERM entity and linked to the canonical glossary term `churn_rate`. Similarly, "buy" matches the relationship `purchase`.

**Term collection at session start:**
```
For each glossary entry:
  → add the key as a business_term (e.g., "churn_rate")
  → add every alias key as a business_term (e.g., "attrition_rate", "cancellation_rate")
For each relationship entry:
  → add the key as a business_term (e.g., "purchase")
  → add every alias key as a business_term (e.g., "buy", "order")
EntityRuler generates pattern variants for each (underscore, space, camelCase, singular/plural)
```

When NER detects a glossary term in a query or document chunk, it creates an Entity with `ner_type="TERM"` and `semantic_type="TERM"`, linked to the chunk via `ChunkEntity`. This enables follow-the-entity searches — find all chunks that mention a business concept.

##### 2. Vector Search (Semantic Similarity)

Glossary entries are embedded as chunks in the existing vector store alongside schema, API, and document chunks:

```
Schema metadata → chunks (source="schema", chunk_type=DB_TABLE/DB_COLUMN)
API endpoints   → chunks (source="api", chunk_type=API_ENDPOINT)
Documents       → chunks (source="document", chunk_type=DOCUMENT)
Glossary terms  → chunks (source="glossary", chunk_type=GLOSSARY_TERM)   ← NEW
```

Each glossary entry becomes a chunk with the full semantic context embedded:

```
"churn_rate: Percentage of customers who cancelled their subscription
in a given period. Aliases: attrition_rate, cancellation_rate.
Is a: customer_metric. Extends: finance.revenue_metric."
```

Embedding the hierarchy in the chunk text means semantic search naturally surfaces related terms. A query about "customer metrics" matches `customer_metric` directly, and because `churn_rate` mentions "Is a: customer_metric" in its chunk, it also gets a similarity boost.

##### 3. LLM Tool Results (search_all)

`search_all` returns glossary matches as a 4th category alongside tables, APIs, and documents:

```python
results = {
    "tables": [...],
    "apis": [...],
    "documents": [...],
    "glossary": [...],   # ← matched terms with definitions + bindings
}
```

When a glossary term has bindings, they are included in the result — giving the LLM authoritative field/value mappings without additional tool calls.

##### Full Flow Example

```
User: "What's our attrition rate by region?"
  │
  ├─ NER: EntityRuler matches "attrition rate" → TERM entity
  │       (alias of canonical term "churn_rate")
  │
  ├─ search_all("attrition rate by region"):
  │   ├─ Vector search embeds query, cosine similarity across ALL chunks
  │   ├─ glossary: [{term: "churn_rate", definition: "...",
  │   │              bindings: {sales_db/customers: {status: {values: {churned: "C"}}}}}]
  │   ├─ tables:   [{name: "customers", relevance: 0.82}]
  │   └─ documents: [{name: "kpi_guide", section: "Retention"}]
  │
  ├─ LLM calls get_table_schema("sales", "customers")
  │   → columns: status (varchar), region (varchar), cancelled_at (date)
  │
  └─ LLM generates SQL — KNOWS status='C' means churned (from binding)
     rather than guessing from sample values
```

##### Auto-suggest: Incremental Glossary Generation (Future)

The glossary can be bootstrapped automatically via LLM prompts, processed **one source at a time** against the current glossary state. All output is suggestion-only — user confirms, edits, or discards via the right-hand panel UI.

**Incremental loop:**

```
Source A connected → prompt with Source A metadata + empty glossary
  → candidates surfaced in UI → user confirms/edits
  → glossary updated

Source B connected → prompt with Source B metadata + glossary from A
  → LLM sees existing terms, proposes new + extends + aliases + bindings
  → candidates surfaced → user confirms
  → glossary updated

Document C indexed → prompt with doc chunks + glossary from A+B
  → LLM finds terms used in prose, connects to existing glossary
  → candidates surfaced → user confirms
```

**Prompt inputs per source:**

| Input | Purpose | Scoping |
|---|---|---|
| Source metadata | Schema DDL, API spec summary, or doc chunks | One source per prompt |
| Existing glossary | Avoid duplication, build hierarchy | Only terms sharing entities with this source + their ancestors |
| Sample values | Enable enum binding proposals | From `get_sample_values` |
| Entity co-occurrences | Connect source entities to known glossary terms | Compact list |

**Prompt template (database source):**

```
You are building a business glossary incrementally. A new database has been connected.

## New Source: sales_db

Tables and columns:
- customers (id PK, name, email, status varchar, region varchar, created_at, cancelled_at)
  FK: none
  Sample values — status: ["A", "C", "I"], region: ["US", "EU", "APAC"]
- orders (id PK, customer_id FK→customers.id, product_id FK→products.id, amount decimal, order_date)
- products (id PK, name, category varchar, price decimal)
  Sample values — category: ["Electronics", "Clothing", "Food"]

## Existing Glossary (relevant terms only)

customer_metric:
  definition: "Metrics measuring customer behavior and lifecycle"
churn_rate:
  definition: "Percentage of customers who cancelled in a period"
  is_a: customer_metric

## Task

For each business concept you identify in this source:

1. Check if it already exists in the glossary (possibly under a different name)
   → If yes, propose it as an alias and suggest bindings
2. If new, propose a glossary entry with:
   - key (snake_case)
   - definition (one sentence, what it means to the business)
   - is_a (if it classifies under an existing term)
   - part_of (if it composes into an existing term)
   - aliases (alternative names a user might say)
   - bindings to this source (source/resource format, include values for enums)

Output valid YAML. Only propose terms that represent business concepts —
skip technical columns (id, created_at) and generic names (name, email).

## Task 2: Propose Relationships

Using the glossary terms above (existing + newly proposed), propose relationships
that connect them. Each relationship is an activity (verb) with subject/object pairs.

For each relationship:
- key (snake_case verb)
- definition (one sentence)
- aliases (alternative verbs a user might say)
- pairs: each pair has subject (glossary term), object (glossary term), cardinality

Infer from: FK paths, source structure, and domain semantics.
Output valid YAML under a `relationships:` key.
```

**Expected output:**

```yaml
customer:
  definition: "A person or entity that purchases products"
  aliases:
    buyer: {}
    client: {}
  bindings:
    sales_db/customers:
      status:
        values:
          active: "A"
          churned: "C"
          inactive: "I"
      region: {}
      cancelled_at: {}

order:
  definition: "A customer purchase transaction"
  part_of: customer
  aliases:
    purchase: {}
  bindings:
    sales_db/orders:
      amount: {}
      order_date: {}

product:
  definition: "An item available for purchase"
  aliases:
    item: {}
  bindings:
    sales_db/products:
      category:
        values:
          electronics: "Electronics"
          clothing: "Clothing"
          food: "Food"
      price: {}
```

**Expected relationship output (from same prompt):**

```yaml
relationships:
  purchase:
    definition: "A customer buying a product"
    aliases:
      buy: {}
      order: {}
    pairs:
      customer_product:
        subject: customer
        object: product
        cardinality: many-to-many
```

**Three sources of relationship suggestions:**

| Source | Signal | Confidence |
|---|---|---|
| FK paths | `orders.customer_id FK→customers.id` + `orders.product_id FK→products.id` → junction table → many-to-many | High |
| Co-occurrence + SVO | Glossary terms co-occur in doc chunks, SVO extraction finds connecting verb (see `rel_ex.md`) | Medium |
| LLM-inferred | Schema structure + glossary definitions → LLM proposes domain-appropriate relationships | Medium |

The per-source prompt combines all three: FK paths are in the schema metadata, co-occurrences are in the entity data, and the LLM infers from context. One prompt yields both glossary entries and relationship candidates.

**Design choices:**

- **Scoped context**: only include glossary terms sharing entities with the current source + their `is_a`/`part_of` ancestors. Keeps the prompt small.
- **Sample values are critical**: without them, the LLM can't propose enum bindings. `get_sample_values` already exists.
- **Skip technical columns**: prompt instructs this; UI confirmation step catches false positives.
- **One source per prompt**: bounded token budget, focused output.
- **One prompt, two outputs**: glossary entries + relationship candidates from the same prompt pass. Reduces LLM calls and gives the LLM full context to propose relationships between terms it just identified.

**Auto-suggest bindings** for existing glossary terms follows the same pattern: embed glossary definitions, semantic search against schema metadata, use `get_sample_values` for candidate field/value mappings, surface suggestions for user confirmation.

### 6. Relationships: Activities Connecting Concepts

Relationships define how glossary concepts connect via activities. While the glossary defines **what things mean** (nouns), relationships define **how things connect** (verbs). Relationships are purely conceptual — physical join paths are inferred from glossary bindings + schema FK metadata, not declared on the relationship itself.

#### Structure

```yaml
relationships:
  purchase:
    definition: "An entity buying a product or service"
    aliases:
      buy: {}
      order: {}
    pairs:
      customer_product:
        subject: customer
        object: product
        cardinality: many-to-many
      customer_subscription:
        subject: customer
        object: subscription
        cardinality: one-to-many
      partner_product:
        subject: partner
        object: product
        cardinality: many-to-many

  approval:
    definition: "An authorized person approving a request"
    aliases:
      approve: {}
      authorize: {}
    pairs:
      manager_expense:
        subject: manager
        object: expense_report
        cardinality: one-to-many
      director_purchase_order:
        subject: director
        object: purchase_order
        cardinality: one-to-many

  transfer:
    definition: "Movement of a resource between locations"
    pairs:
      employee_department:
        subject: employee
        object: department
        cardinality: many-to-one
      inventory_warehouse:
        subject: inventory
        object: warehouse
        cardinality: many-to-one
```

#### Cardinality

Cardinality lives on each pair, not on the relationship — different subject/object combinations may have different cardinalities.

| Cardinality | LLM implication |
|---|---|
| `one-to-one` | Direct lookup, no aggregation needed |
| `one-to-many` | Subject has multiple objects, aggregate on object side |
| `many-to-one` | Multiple subjects share one object, aggregate on subject side |
| `many-to-many` | Junction resource, aggregation on either side |

#### Physical Join Path Inference

Relationships contain no physical details. Join paths are inferred at query time:

1. Resolve subject glossary term → physical resources (from glossary bindings)
2. Resolve object glossary term → physical resources (from glossary bindings)
3. Find resources that FK-connect both (from schema metadata)
4. That's the join path — inferred, not declared

When inference fails (no FKs, cross-source, ambiguous paths), that's a signal the glossary terms need better bindings — not that the relationship needs physical details.

#### Integration with Discovery Pipeline

Relationship keys + alias keys feed into `business_terms` for NER, alongside glossary terms:

```
Term collection at session start:
  For each glossary entry:
    → add key + all alias keys as business_terms
  For each relationship entry:
    → add key + all alias keys as business_terms
```

Both tagged as `ner_type="TERM"` by EntityRuler. The distinction comes from `search_all` results, which return them in separate categories.

Relationship entries are embedded as chunks in the vector store (`source="relationship"`):

```
"purchase: An entity buying a product or service. Aliases: buy, order.
Pairs: customer→product (many-to-many), customer→subscription (one-to-many)."
```

`search_all` returns five categories:

```python
results = {
    "tables": [...],
    "apis": [...],
    "documents": [...],
    "glossary": [...],
    "relationships": [...],
}
```

#### Full Flow Example

```
User: "What did customers buy last quarter?"
  │
  ├─ NER: EntityRuler matches "buy" → TERM entity
  │       (alias of relationship "purchase")
  │
  ├─ search_all("customers buy last quarter"):
  │   ├─ relationships: [{name: "purchase", pairs: {customer_product:
  │   │                    {subject: "customer", object: "product",
  │   │                     cardinality: "many-to-many"}}}]
  │   ├─ glossary: [{term: "customer", bindings: {sales_db/customers: {...}}}]
  │   ├─ glossary: [{term: "product", bindings: {sales_db/products: {...}}}]
  │   └─ tables:   [{name: "orders", relevance: 0.78}]
  │
  ├─ LLM infers: customer→product is many-to-many
  │   Both bound to sales_db → FK metadata reveals orders as junction
  │
  └─ LLM generates SQL with correct joins and aggregation
```

### 7. Enable/Disable via `null`

Setting a key to `null` removes it from that tier:

```yaml
# system config.yaml
sources:
  databases:
    legacy_db:
      $ref: ./sources/legacy.yaml

# user domain config.yaml — disable legacy_db for this user's domain
sources:
  databases:
    legacy_db: null
```

### 8. Source Attribution

Every resolved item tracks its source tier:

```python
class ConfigSource(Enum):
    SYSTEM = "system"
    SYSTEM_DOMAIN = "system_domain"
    USER = "user"
    USER_DOMAIN = "user_domain"
    SESSION = "session"
```

### 9. Tier Loading and ResolvedConfig

Tier loading produces a single `ResolvedConfig` — the only config object downstream code ever sees. The tier machinery is internal to `TieredConfigLoader`. No other module needs to understand tiers, `$ref`, `null`-deletion, or merge order.

#### Load Order

When a session selects a domain:

1. Load system config (resolve `$ref`)
2. Load system domain (sub-domains merge first, then parent overlays; resolve `$ref`)
3. Load user config (resolve `$ref`)
4. Load user domain (same domain name under user; resolve `$ref`)
5. Load session config (resolve `$ref`)
6. Deep merge tiers in order (higher tier wins; `null` deletes key)
7. Strip all `null` keys from merged result
8. Produce `ResolvedConfig`

#### ResolvedConfig

```python
@dataclass
class ResolvedConfig:
    """Single merged config. All tiers resolved. No $ref, no null, no tier awareness needed."""
    sources: SourcesConfig          # databases + apis + documents (flat maps)
    roles: dict[str, RoleConfig]
    rights: dict[str, RightConfig]
    facts: dict[str, str]
    learnings: LearningsConfig      # corrections + rules
    glossary: dict[str, GlossaryTermConfig]
    relationships: dict[str, RelationshipConfig]
    skills: dict[str, SkillConfig]
    llm: LlmConfig
    preferences: Optional[PreferencesConfig]  # user tier only

    # Source attribution — for UI tier badges and promote/remove actions.
    # Keyed by dotted path (e.g., "facts.company_name", "glossary.churn_rate").
    # Only the UI tier management panel reads this. All other consumers ignore it.
    _attribution: dict[str, ConfigSource] = field(default_factory=dict, repr=False)

    def source_of(self, path: str) -> Optional[ConfigSource]:
        """Which tier did this value come from? For UI tier badges only."""
        return self._attribution.get(path)
```

#### TieredConfigLoader

```python
class TieredConfigLoader:
    def __init__(
        self,
        config_dir: Path,
        user_id: str,
        base_dir: Path,
        session_id: Optional[str],
        domain_name: Optional[str] = None,
    ):
        self.tiers = [
            ("system", config_dir),
            ("system_domain", config_dir / "domains" / domain_name) if domain_name else None,
            ("user", base_dir / user_id),
            ("user_domain", base_dir / user_id / "domains" / domain_name) if domain_name else None,
            ("session", base_dir / user_id / "sessions" / session_id) if session_id else None,
        ]
        self.tiers = [(name, path) for name, path in self.tiers if path]

    def resolve(self) -> ResolvedConfig:
        """Load all tiers, merge, return single flat config.

        This is the only public method. Called at:
        - Session start
        - Domain add/remove
        - Source add/remove/enable/disable
        - Glossary/relationship term change
        - Fact/learning/right/role change
        - Tier promotion or removal

        Downstream code holds a reference to the returned ResolvedConfig.
        On any mutation, call resolve() again and replace the reference.
        """
        merged = {}
        attribution = {}
        for tier_name, tier_path in self.tiers:
            raw = self._load_tier(tier_path)       # resolve $ref within tier
            merged = self._deep_merge(merged, raw, tier_name, attribution)
        merged = self._strip_nulls(merged)
        return ResolvedConfig(**merged, _attribution=attribution)
```

#### Downstream Contract

All consumers receive `ResolvedConfig`. No exceptions.

| Consumer | What it reads | Tier-aware? |
|----------|--------------|-------------|
| Session / solve pipeline | sources, glossary, relationships, facts, learnings, rights, roles, llm | No |
| Entity extraction | glossary keys + aliases, relationship keys + aliases, source schemas | No |
| Vector store / search | glossary chunks, relationship chunks | No |
| REPL commands | facts, glossary, learnings, sources | No |
| UI panels (data display) | All sections | No |
| UI tier management panel | All sections + `_attribution` | Yes (only consumer) |

#### Re-resolve Triggers

Any mutation to any tier triggers `resolve()` → new `ResolvedConfig`. Config is small; re-merge is negligible. The session holds the reference and swaps it atomically.

```
tier mutation → loader.resolve() → new ResolvedConfig → session.config = new_config
                                                      → trigger entity rebuild (if scope changed)
                                                      → notify UI (WebSocket)
```

Scope-affecting mutations (sources, glossary, relationships) additionally trigger entity rebuild via the fingerprint mechanism (§10).

### 10. Entity Extraction Strategy

Entity extraction runs NER (spaCy + EntityRuler) across all in-scope chunks to link business terms, schema names, and API names to chunks. This is the most expensive per-session operation — proportional to chunk count, not term count.

#### Cost Profile

| Component | Cost | Notes |
|---|---|---|
| spaCy model load | ~200ms | Process singleton, once per server |
| EntityRuler compilation | ~5ms / 1000 patterns | Cheap even at scale |
| NER per chunk | ~1-3ms | The bottleneck — scales with chunk count |
| 5,000 chunks | ~10s | Acceptable |
| 20,000 chunks | ~40-60s | Unacceptable if blocking |

Term count is not the bottleneck. Chunk count is.

#### Background Rebuild

Entity extraction runs in a **dedicated background thread** (separate from the 4-worker query thread pool). Session start and domain/resource changes are never blocked by extraction.

**Trigger events:**
- Session created (initial extraction)
- Domain added/removed mid-session
- Source added/removed/enabled/disabled (any tier)
- Glossary or relationship term added/removed/modified (any tier)
- Tier promotion of a source, glossary term, or relationship

**During rebuild**, queries still work — vector similarity search is unaffected. Entity-based traversal uses stale data until rebuild completes. No correctness issue, just reduced entity linking until done.

**Cancellation:** If another scope change arrives mid-rebuild, cancel the current rebuild and start a new one with the latest scope. Only the final state matters.

#### WebSocket Progress Events

Rebuild progress pushes to the UI via the existing WebSocket pipeline (`SessionWebSocket` queue bridge):

```
ENTITY_REBUILD_START      → {session_id, trigger, total_chunks}
ENTITY_REBUILD_PROGRESS   → {session_id, processed, total}
ENTITY_REBUILD_COMPLETE   → {session_id, entity_count, duration_ms}
```

The UI shows a non-blocking progress indicator. Entity-dependent panels (glossary matches, related entities) refresh on `ENTITY_REBUILD_COMPLETE`.

#### Fingerprint Caching

Full rebuild is always correct but expensive. Fingerprint caching skips NER entirely when the inputs haven't changed.

```
fingerprint = hash(sorted_chunk_ids + sorted_term_set)
```

The term set includes schema terms + API terms + glossary keys + glossary aliases + relationship keys + relationship aliases. The chunk IDs are the visible chunks for the current scope (base + active domains).

**Flow:**
1. Scope change triggers rebuild
2. Compute fingerprint from current chunk IDs + term set
3. Check fingerprint against stored extractions
4. **Cache hit** → reuse existing entity/chunk_entity rows, skip NER entirely
5. **Cache miss** → full NER rebuild in background, store new fingerprint on completion

**Cross-session reuse:** Two users in the same domain with the same resources produce the same fingerprint. The first session pays the extraction cost; subsequent sessions get instant entity availability.

**Invalidation is natural** — any change to sources, glossary, relationships, or domain scope changes the fingerprint. No explicit cache invalidation logic needed.

#### Full Rebuild vs. Incremental

Full rebuild is preferred over incremental because the EntityRuler is a compiled pattern matcher. When terms change, existing chunks need re-processing with the new ruler — a chunk processed before "churn_rate" was in the glossary won't have that entity linked. Incremental addition of new chunks misses these cross-references.

Fingerprint caching makes full rebuild cheap in the common case (nothing changed). Background execution makes it non-blocking in the expensive case (something did change).

## UI: Tier Management

The existing right-hand panel displays config items (facts, learnings, roles, sources, rights, skills, glossary) with:

- **Tier badge** on each item indicating source (system / domain / user / session)
- **Action buttons** per item:
  - **Promote** — moves item up one tier (session → user, user → system if admin)
  - **Remove** — deletes from current tier (lower tier value resurfaces if one exists)
  - **Create** — add new item at the current tier

**Scope:**
- Promotion is per-item, one tier at a time
- Remove only available for items at your tier or below
- Promote to system requires admin
- Promote to domain requires domain owner
- No demotion — remove from higher tier and re-add at lower tier manually
- No bulk operations for v1

**Domains are managed manually** (file-based). Too structural for UI CRUD in v1.

## Merge Examples

**Facts merge (5 tiers):**
```yaml
# system facts
company_name: "Acme Corp"
fiscal_year_start: "January 1"

# system domain facts (customer)
fiscal_year_start: "April 1"      # overrides system
rating_scale: "1-5"

# user facts
my_department: "Engineering"

# user domain facts (customer)
rating_scale: "1-10"              # overrides system domain

# session facts
focus_area: "Q4 retention"

# Result:
# company_name="Acme Corp" (system), fiscal_year_start="April 1" (sys domain),
# rating_scale="1-10" (user domain), my_department="Engineering" (user),
# focus_area="Q4 retention" (session)
```

**Glossary merge:**
```yaml
# system domain glossary (customer)
churn_rate:
  definition: "Percentage of customers who cancelled in a period"
  is_a: customer_metric

# user domain glossary (customer) — add bindings
churn_rate:
  bindings:
    sales_db.customers:
      columns:
        status:
          values:
            churned: "C"

# Result: definition from system domain, bindings from user domain (deep merge)
```

## Implementation Phases

### Phase 1: Core Config Schema
1. Add `rights`, `facts`, `learnings`, `glossary` maps to config schema
2. Convert learnings from list to map (keyed by id)
3. Ensure all sections use maps (no arrays)
4. Rename "project" to "domain" throughout codebase

### Phase 2: 5-Tier Config Loader → ResolvedConfig
**New file:** `constat/core/tiered_config.py`

1. `TieredConfigLoader` — loads tiers, resolves `$ref` per tier, deep merges, strips `null`
2. `ResolvedConfig` — single flat dataclass, the only config object downstream code sees
3. `_attribution` dict for UI tier badges (dotted path → `ConfigSource`)
4. `resolve()` as the single public method — called at startup and on any tier mutation
5. All existing config consumers migrated to accept `ResolvedConfig`

### Phase 3: Domain Directory Structure
1. `DomainConfig.from_directory(path: Path)` method
2. Support all core config sections + llm + skills/
3. Recursive sub-domain loading and merge
4. Domain discovery scans directories
5. Add `owner` and `definition` fields

### Phase 4: Glossary & Relationships Integration
1. Glossary chunk creation at startup (embedded in vector store)
2. Relationship chunk creation at startup (embedded in vector store)
3. Glossary keys + aliases + relationship keys + aliases feed EntityExtractor business_terms
4. Add `glossary` and `relationships` categories to `search_all` results
5. Bindings included in search results when present

### Phase 5: Entity Extraction Scalability
1. Move entity extraction to dedicated background thread (separate from query pool)
2. Add WebSocket events: `ENTITY_REBUILD_START`, `ENTITY_REBUILD_PROGRESS`, `ENTITY_REBUILD_COMPLETE`
3. Implement fingerprint caching: `hash(chunk_ids + term_set)` → skip NER on cache hit
4. Full rebuild on scope change (domain add/remove, source change, glossary/relationship change)
5. Cancellation: new scope change cancels in-progress rebuild

### Phase 6: Update Existing Loading
- `constat/core/config.py` — `Config.load_domain()` from directory
- `constat/server/routes/databases.py` — domain listing
- `constat/server/session_manager.py` — domain selection with 5-tier merge

### Phase 7: UI Tier Badges and Actions
- Tier badge on config items in right-hand panel
- Promote / Remove / Create actions
- Admin gate for user → system promotion
- Domain owner gate for domain promotion

### Phase 8: Auto-suggest Glossary & Relationships
1. Per-source LLM prompt: source metadata + existing glossary + sample values → candidate glossary entries + relationship candidates
2. FK-derived relationship suggestions (high confidence, no LLM needed)
3. Co-occurrence + SVO relationship suggestions (medium confidence, see `rel_ex.md`)
4. Suggestion UI in right-hand panel: confirm / edit / discard per candidate
5. Confirmed entries persist to glossary/relationships config at chosen tier
6. Auto-suggest bindings for existing glossary terms via semantic search + `get_sample_values`

## Critical Files

| File | Changes |
|------|---------|
| `constat/core/config.py` | Add rights, facts, learnings, glossary, relationships to schema; domain directory loading; rename project → domain |
| `constat/core/tiered_config.py` | **NEW** — `TieredConfigLoader` + `ResolvedConfig`; resolve once, flat config downstream |
| `constat/core/skills.py` | Multi-tier skill discovery |
| `constat/storage/facts.py` | Map-based facts |
| `constat/storage/learnings.py` | Map-based learnings (keyed by id) |
| `constat/discovery/schema_tools.py` | Add glossary + relationships categories to `search_all` |
| `constat/discovery/entity_extractor.py` | Feed glossary + relationship aliases as business_terms |
| `constat/catalog/schema_manager.py` | Build glossary + relationship chunks for vector store |
| `constat/server/session_manager.py` | 5-tier merge on domain selection; background entity rebuild |
| `constat/server/websocket.py` | Add `ENTITY_REBUILD_*` event types |
| `constat/discovery/glossary_suggest.py` | **NEW** — per-source LLM prompt for glossary + relationship auto-suggest |
