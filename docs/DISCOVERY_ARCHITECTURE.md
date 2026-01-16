# Tool-Based Discovery Architecture

## Overview

Replace upfront schema/API loading in the system prompt with on-demand discovery via tool calling. This reduces token usage, scales to large catalogs, and enables dynamic fact resolution.

## Current vs Proposed

### Current Architecture
```
┌─────────────────────────────────────────────────────────┐
│                    System Prompt                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐│
│  │ Full Schema │ │ All APIs    │ │ All Documents       ││
│  │ (all DBs)   │ │ (all ops)   │ │ (full content)      ││
│  └─────────────┘ └─────────────┘ └─────────────────────┘│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       Planner         │
              │   (no tool access)    │
              └───────────────────────┘
```

**Problems:**
- Token-heavy system prompts (10k+ tokens for complex setups)
- Loads everything regardless of relevance
- Doesn't scale to 50+ databases or 100+ APIs
- Documents loaded fully upfront

### Implemented Architecture (Phase 1 Complete)
```
┌─────────────────────────────────────────────────────────┐
│                 Minimal System Prompt                    │
│  "You have access to databases, APIs, and documents.    │
│   Use discovery tools to find what you need."           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │     PromptBuilder     │
              │ (auto mode detection) │
              └───────────┬───────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────────────┐     ┌───────────────────────────────┐
│  Tool-Capable Models  │     │    Legacy Models              │
│  (Claude 3+, GPT-4)   │     │    (Claude 2, GPT-3 Instruct) │
│                       │     │                               │
│  Minimal prompt +     │     │  Full metadata embedded       │
│  discovery tools      │     │  in system prompt             │
└───────────┬───────────┘     └───────────────────────────────┘
            │
  ┌─────────┼─────────┬─────────────┐
  ▼         ▼         ▼             ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐
│Schema│ │ API  │ │ Doc  │ │   Fact   │
│Tools │ │Tools │ │Tools │ │  Tools   │
└──────┘ └──────┘ └──────┘ └──────────┘
```

**Benefits:**
- Minimal initial tokens (~500 vs 10k+)
- Only loads what's relevant to the query
- Scales to unlimited databases/APIs/documents
- Dynamic fact resolution from documents
- **Automatic mode selection** - no configuration needed

### Automatic Model Detection

The `PromptBuilder` class automatically detects whether a model supports tool calling. This is **NOT** a configuration option - it's determined by the model being used.

**Tool-Capable Models (minimal prompt + tools):**
- Claude 3+ (Opus, Sonnet, Haiku)
- GPT-4, GPT-3.5-turbo
- Gemini

**Legacy Models (full metadata in prompt):**
- Claude 2, Claude Instant
- GPT-3.5-turbo-instruct
- Text-davinci, text-curie, etc.

```python
from constat.discovery import DiscoveryTools, PromptBuilder

tools = DiscoveryTools(schema_manager, api_catalog, config)
builder = PromptBuilder(tools)

# Automatic - detects Claude 3 supports tools
prompt, use_tools = builder.build_prompt("claude-sonnet-4-20250514")
# → use_tools=True, minimal prompt (~500 tokens)

# Automatic - detects Claude 2 doesn't support tools
prompt, use_tools = builder.build_prompt("claude-2")
# → use_tools=False, full metadata embedded
```

---

## Discovery Tool Specifications

### 1. Schema Discovery Tools

#### `list_databases`
Returns available databases with descriptions (includes SQL, NoSQL, and file-based sources).

```python
def list_databases() -> list[dict]:
    """
    List all configured databases.

    Returns:
        List of database info:
        [
            {
                "name": "chinook",
                "type": "sql",
                "description": "Digital music store - artists, albums, tracks, sales"
            },
            {
                "name": "mongodb_logs",
                "type": "mongodb",
                "description": "Application event logs"
            },
            {
                "name": "web_metrics",
                "type": "csv",
                "description": "Web analytics data"
            },
            {
                "name": "events",
                "type": "json",
                "description": "Clickstream events"
            }
        ]
    """
```

#### `list_tables`
Returns tables/collections for a specific database.

```python
def list_tables(database: str) -> list[dict]:
    """
    List tables in a database with row counts and descriptions.

    Args:
        database: Database name

    Returns:
        [
            {
                "name": "artists",
                "row_count": 275,
                "description": "Music artists/bands"
            },
            {
                "name": "albums",
                "row_count": 347,
                "description": "Albums linked to artists"
            }
        ]
    """
```

#### `get_table_schema`
Returns full column details for a table.

```python
def get_table_schema(database: str, table: str) -> dict:
    """
    Get detailed schema for a specific table.

    Args:
        database: Database name
        table: Table name

    Returns:
        {
            "database": "chinook",
            "table": "tracks",
            "columns": [
                {"name": "TrackId", "type": "INTEGER", "nullable": False, "primary_key": True},
                {"name": "Name", "type": "VARCHAR(200)", "nullable": False},
                {"name": "AlbumId", "type": "INTEGER", "nullable": True, "foreign_key": "albums.AlbumId"},
                {"name": "UnitPrice", "type": "DECIMAL(10,2)", "nullable": False}
            ],
            "row_count": 3503,
            "sample_values": {
                "Name": ["For Those About To Rock", "Balls to the Wall", "Fast As a Shark"],
                "UnitPrice": [0.99, 0.99, 0.99]
            }
        }
    """
```

#### `search_tables`
Semantic search across all tables.

```python
def search_tables(query: str, limit: int = 5) -> list[dict]:
    """
    Find tables relevant to a natural language query.

    Args:
        query: Natural language description (e.g., "customer purchases")
        limit: Max results

    Returns:
        [
            {
                "database": "chinook",
                "table": "invoices",
                "relevance": 0.92,
                "description": "Customer purchase records with dates and totals"
            },
            {
                "database": "chinook",
                "table": "invoice_lines",
                "relevance": 0.88,
                "description": "Line items for each invoice"
            }
        ]
    """
```

### 2. API Discovery Tools

#### `list_apis`
Returns configured APIs.

```python
def list_apis() -> list[dict]:
    """
    List all configured APIs.

    Returns:
        [
            {
                "name": "countries",
                "type": "graphql",
                "url": "https://countries.trevorblades.com/graphql",
                "description": "Country and continent data"
            },
            {
                "name": "petstore",
                "type": "openapi",
                "description": "Pet store inventory management"
            }
        ]
    """
```

#### `list_operations`
Returns operations for an API.

```python
def list_operations(api: str) -> list[dict]:
    """
    List operations available in an API.

    Args:
        api: API name

    Returns:
        [
            {
                "name": "countries",
                "type": "query",
                "description": "Get all countries with optional filters"
            },
            {
                "name": "country",
                "type": "query",
                "description": "Get a single country by code"
            }
        ]
    """
```

#### `get_operation_details`
Returns full operation schema.

```python
def get_operation_details(api: str, operation: str) -> dict:
    """
    Get detailed schema for an API operation.

    Args:
        api: API name
        operation: Operation name

    Returns:
        {
            "api": "petstore",
            "operation": "getPetById",
            "type": "query",
            "method": "GET",
            "path": "/pet/{petId}",
            "parameters": [
                {"name": "petId", "type": "integer", "required": True, "description": "Pet ID"}
            ],
            "response_schema": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["available", "pending", "sold"]}
                }
            }
        }
    """
```

#### `search_operations`
Semantic search across all API operations.

```python
def search_operations(query: str, limit: int = 5) -> list[dict]:
    """
    Find API operations relevant to a query.

    Args:
        query: Natural language description
        limit: Max results

    Returns:
        [
            {
                "api": "petstore",
                "operation": "findPetsByStatus",
                "relevance": 0.91,
                "description": "Find pets by availability status"
            }
        ]
    """
```

### 3. Document Discovery Tools

#### `list_documents`
Returns configured documents.

```python
def list_documents() -> list[dict]:
    """
    List all configured reference documents.

    Returns:
        [
            {
                "name": "business_rules",
                "type": "file",
                "format": "markdown",
                "description": "Revenue calculation rules and thresholds",
                "tags": ["finance", "rules"]
            },
            {
                "name": "data_dictionary",
                "type": "confluence",
                "description": "Field definitions and business glossary"
            }
        ]
    """
```

#### `get_document`
Returns document content.

```python
def get_document(name: str) -> dict:
    """
    Get the content of a document.

    Args:
        name: Document name

    Returns:
        {
            "name": "business_rules",
            "content": "## Revenue Rules\n\n- VIP threshold: $100k lifetime...",
            "format": "markdown",
            "last_updated": "2024-01-15T10:30:00Z"
        }
    """
```

#### `search_documents`
Semantic search across document content.

```python
def search_documents(query: str, limit: int = 5) -> list[dict]:
    """
    Search across all documents for relevant content.

    Args:
        query: Natural language query
        limit: Max results

    Returns:
        [
            {
                "document": "business_rules",
                "excerpt": "VIP Customer: A customer with lifetime value > $100,000...",
                "relevance": 0.94,
                "section": "Customer Classifications"
            },
            {
                "document": "data_dictionary",
                "excerpt": "churn_date: Date when customer was marked inactive...",
                "relevance": 0.82,
                "section": "Customer Fields"
            }
        ]
    """
```

### 4. Fact Resolution Tool

#### `resolve_fact`
Multi-source fact resolution with confidence scoring.

```python
def resolve_fact(question: str) -> dict:
    """
    Resolve a factual question using all available sources.

    Uses: documents, schema descriptions, configured knowledge, web search (if enabled)

    Args:
        question: Natural language question (e.g., "What defines a VIP customer?")

    Returns:
        {
            "question": "What defines a VIP customer?",
            "answer": "A customer with lifetime value exceeding $100,000",
            "confidence": 0.95,
            "sources": [
                {
                    "type": "document",
                    "name": "business_rules",
                    "excerpt": "VIP Customer: lifetime value > $100,000",
                    "confidence": 0.95
                }
            ],
            "needs_clarification": False
        }
    """
```

### 5. Skill Discovery Tools

Skills are domain-specific knowledge modules that provide specialized context and analysis guidelines. This follows the standard skill/prompt pattern used by Anthropic (Claude Code), OpenAI, and other AI providers for extending chatbot capabilities with domain-specific knowledge.

#### Skill Structure

Skills are stored in directories following the pattern `skills/<skill-name>/SKILL.md`:

```
.constat/skills/
├── financial-analysis/
│   ├── SKILL.md
│   └── references/
│       └── indicators.md
└── healthcare-compliance/
    └── SKILL.md
```

#### SKILL.md Format

Each skill is a Markdown file with YAML frontmatter:

```markdown
---
name: financial-analysis
description: Specialized instructions for financial data analysis
allowed-tools:
  - Read
  - Grep
  - list_tables
  - get_table_schema
---

# Financial Analysis Skill

## Analysis Process
1. Validate data quality
2. Check against each indicator category in [references/indicators.md](references/indicators.md)
3. Calculate key metrics
...
```

**Link Following:** SKILL.md files can contain relative links to other files within the skill directory (e.g., `[references/indicators.md](references/indicators.md)`). Links are always resolved relative to the `<skill-name>/` folder (the directory containing SKILL.md) and loaded as additional context.

#### `list_skills`
Returns available skills from all discovery paths.

```python
def list_skills() -> list[dict]:
    """
    List all available skills.

    Returns:
        [
            {
                "name": "financial-analysis",
                "description": "Specialized instructions for financial data analysis",
                "path": ".constat/skills/financial-analysis/SKILL.md",
                "allowed_tools": ["Read", "Grep", "list_tables", "get_table_schema"]
            },
            {
                "name": "healthcare-compliance",
                "description": "HIPAA and healthcare regulatory compliance checks",
                "path": "~/.constat/skills/healthcare-compliance/SKILL.md",
                "allowed_tools": ["Read", "search_documents"]
            }
        ]
    """
```

#### `get_skill`
Returns the full content of a skill, including linked references.

```python
def get_skill(name: str) -> dict:
    """
    Get a skill's content with linked references resolved.

    Args:
        name: Skill name

    Returns:
        {
            "name": "financial-analysis",
            "description": "Specialized instructions for financial data analysis",
            "content": "# Financial Analysis Skill\n\n## Analysis Process\n...",
            "allowed_tools": ["Read", "Grep", "list_tables"],
            "references": {
                "references/indicators.md": "# Financial Indicators\n\n## Liquidity Ratios..."
            }
        }
    """
```

#### `search_skills`
Semantic search across skill descriptions and content.

```python
def search_skills(query: str, limit: int = 3) -> list[dict]:
    """
    Find skills relevant to a query.

    Args:
        query: Natural language description
        limit: Max results

    Returns:
        [
            {
                "name": "financial-analysis",
                "relevance": 0.92,
                "description": "Specialized instructions for financial data analysis"
            }
        ]
    """
```

#### `list_skill_links`
List all links discovered in a skill's content. Links are parsed when the skill loads but content is NOT fetched until needed.

```python
def list_skill_links(name: str) -> dict:
    """
    List available links for lazy loading.

    Args:
        name: The skill name

    Returns:
        {
            "skill": "financial-analysis",
            "links": [
                {"text": "indicators", "target": "references/indicators.md", "is_url": False, "line_number": 15},
                {"text": "API docs", "target": "https://example.com/api", "is_url": True, "line_number": 23}
            ]
        }
    """
```

#### `resolve_skill_link`
Lazy load a link's content. For relative paths, loads from skill directory. For URLs, fetches via HTTP. Results are cached.

```python
def resolve_skill_link(name: str, target: str) -> dict:
    """
    Fetch linked content on-demand.

    Args:
        name: The skill name
        target: The link target (relative path or URL)

    Returns:
        {
            "skill": "financial-analysis",
            "target": "references/indicators.md",
            "content": "# Financial Indicators\n\n## Liquidity Ratios..."
        }
        # or {"error": "Failed to resolve..."} if not found
    """
```

#### Discovery Paths

Skills are discovered from multiple locations (in order of precedence):

1. **Project skills**: `.constat/skills/` in the project directory
2. **Global skills**: `~/.constat/skills/` in the user's home directory
3. **Config-specified paths**: Additional paths defined in `config.yaml`

---

## Integration with Planner

### Updated Planner System Prompt

```
You are a data analysis planner with access to databases, APIs, and reference documents.

IMPORTANT: You do NOT have full schema information upfront. Use discovery tools to:
1. Find relevant databases and tables for the query
2. Get detailed schemas for tables you need
3. Search documents for business rules and definitions
4. Resolve unclear terms or requirements

## Available Discovery Tools

- list_databases() - See what databases are available
- list_tables(database) - See tables in a database
- get_table_schema(database, table) - Get column details
- search_tables(query) - Find relevant tables by description

- list_apis() - See what APIs are available
- list_operations(api) - See operations in an API
- get_operation_details(api, operation) - Get operation schema
- search_operations(query) - Find relevant operations

- list_documents() - See reference documents
- get_document(name) - Read a document
- search_documents(query) - Search document content

- resolve_fact(question) - Get answers from knowledge sources

## Planning Process

1. DISCOVER: Use tools to find relevant resources
2. CLARIFY: Resolve unclear terms with resolve_fact()
3. PLAN: Create step-by-step execution plan
4. OUTPUT: Return structured plan with discovered context

Only include resources you've verified exist via discovery tools.
```

### Planner Workflow Example

**User Query:** "Show me VIP customer spending trends by quarter"

**Planner Discovery Phase:**
```
1. resolve_fact("What defines a VIP customer?")
   → "Lifetime value > $100,000" (from business_rules doc, confidence: 0.95)

2. search_tables("customer spending purchases")
   → invoices (chinook, relevance: 0.92)
   → customers (chinook, relevance: 0.88)

3. get_table_schema("chinook", "invoices")
   → InvoiceId, CustomerId, InvoiceDate, Total, ...

4. get_table_schema("chinook", "customers")
   → CustomerId, FirstName, LastName, ...
```

**Planner Output:**
```json
{
  "reasoning": "User wants VIP customer spending by quarter. VIP = lifetime value > $100k. Need to calculate customer lifetime values first, filter VIPs, then aggregate by quarter.",
  "discovered_context": {
    "facts": [
      {"term": "VIP customer", "definition": "lifetime value > $100,000", "source": "business_rules"}
    ],
    "tables": ["chinook.invoices", "chinook.customers"],
    "apis": []
  },
  "plan": {
    "steps": [
      {
        "number": 1,
        "goal": "Calculate lifetime value per customer",
        "databases": ["chinook"],
        "tables": ["invoices"]
      },
      {
        "number": 2,
        "goal": "Identify VIP customers (lifetime value > $100k)",
        "inputs": ["step_1_results"],
        "tables": ["customers"]
      },
      {
        "number": 3,
        "goal": "Aggregate VIP spending by quarter",
        "inputs": ["step_2_vip_ids"],
        "tables": ["invoices"]
      }
    ]
  }
}
```

---

## Implementation Status

### Phase 1: Discovery Tool Infrastructure - COMPLETE

**Implemented Files:**
```
constat/discovery/
├── __init__.py           # Module exports
├── tools.py              # DiscoveryTools registry + PromptBuilder
├── schema_tools.py       # Database/table discovery (6 tools)
├── api_tools.py          # API operation discovery (4 tools)
├── doc_tools.py          # Document discovery with loading (4 tools)
├── fact_tools.py         # Fact resolution wrapper (5 tools)
└── skill_tools.py        # Skill discovery with link following (3 tools)

tests/
├── test_discovery_tools.py  # 28 tests covering all tools
└── test_skill_tools.py      # Skill discovery tests
```

**24 Discovery Tools Implemented:**
- Schema: `list_databases`, `list_tables`, `get_table_schema`, `search_tables`, `get_table_relationships`, `get_sample_values`
- API: `list_apis`, `list_api_operations`, `get_operation_details`, `search_operations`
- Documents: `list_documents`, `get_document`, `search_documents`, `get_document_section`
- Facts: `resolve_fact`, `add_fact`, `extract_facts_from_text`, `list_known_facts`, `get_unresolved_facts`
- Skills: `list_skills`, `get_skill`, `search_skills`

**Data Source Types in Schema Discovery:**
- SQL databases (PostgreSQL, MySQL, SQLite, BigQuery, etc.)
- NoSQL databases (MongoDB, DynamoDB, Cassandra, Elasticsearch, etc.)
- File-based sources (CSV, JSON, JSONL, Parquet, Arrow/Feather)

All appear uniformly in `list_databases`, `search_tables`, and `get_table_schema` - the LLM doesn't need to know the underlying type to discover and use them.

**PromptBuilder - Automatic Mode Detection:**
- Detects tool support based on model name
- Tool-capable models: minimal prompt (~500 tokens) + discovery tools
- Legacy models: full metadata embedded in prompt
- NOT configurable - automatic based on model

### Phase 2: Planner Integration - PLANNED

1. **Update Planner to use tools**
   - Add tool definitions to planner API call
   - Handle tool_use responses in planning loop
   - Accumulate discovered context

2. **Create discovery loop**
   ```python
   def plan_with_discovery(self, query: str) -> PlannerResponse:
       messages = [{"role": "user", "content": query}]
       discovered = {"facts": [], "tables": [], "apis": [], "documents": []}

       while True:
           response = self.client.messages.create(
               model=self.model,
               system=DISCOVERY_SYSTEM_PROMPT,
               messages=messages,
               tools=DISCOVERY_TOOLS,
               max_tokens=4096
           )

           if response.stop_reason == "end_turn":
               # Planning complete
               return self._parse_plan(response, discovered)

           elif response.stop_reason == "tool_use":
               # Execute tools, accumulate results
               tool_results = self._execute_tools(response.content)
               discovered = self._accumulate_context(discovered, tool_results)
               messages.append({"role": "assistant", "content": response.content})
               messages.append({"role": "user", "content": tool_results})
   ```

### Phase 3: Document Loader - PARTIAL (inline/file supported)

Document loading for `inline`, `file`, and `http` types is implemented in `doc_tools.py`.

**TODO:**
- Confluence integration
- PDF extraction
- Office document extraction
- Document chunking for vector search

### Phase 4: Caching & Optimization - IMPLEMENTED

1. **Context Preload Cache** (`catalog/preload_cache.py`)
   - Uses seed patterns (from config) to identify relevant tables via vector similarity
   - Caches selected table metadata to `.constat/metadata_preload.json`
   - Preloaded schema is loaded into context at session start
   - Eliminates discovery tool calls for tables matching seed patterns

   ```yaml
   # config.yaml
   context_preload:
     seed_patterns: ["sales", "customer", "revenue"]
     similarity_threshold: 0.3
     max_tables: 50
   ```

2. **Incremental Document Refresh** (`discovery/doc_tools.py`)
   - Tracks file modification times for change detection
   - Only reloads documents that have changed
   - Returns stats: `{added: 1, updated: 2, removed: 0, unchanged: 5}`

3. **`/refresh` Command**
   - Refreshes schema metadata, documents, and preload cache
   - Documents are refreshed incrementally by default
   - Use in REPL to pick up new/changed documents or schema changes

---

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/__init__.py` | New module |
| `constat/discovery/tools.py` | Tool definitions |
| `constat/discovery/schema_tools.py` | Schema discovery implementation |
| `constat/discovery/api_tools.py` | API discovery implementation |
| `constat/discovery/doc_tools.py` | Document discovery with incremental refresh |
| `constat/discovery/fact_tools.py` | Fact resolution implementation |
| `constat/discovery/skill_tools.py` | Skill discovery with lazy link following |
| `constat/catalog/preload_cache.py` | Context preload cache (seed patterns) |
| `constat/documents/loader.py` | Document loading |
| `constat/documents/index.py` | Document vector index |
| `constat/planning/planner.py` | Add tool support, discovery loop |
| `constat/core/config.py` | DocumentConfig, ContextPreloadConfig |

---

## Execution Modes

The system supports three execution modes, automatically selected based on query analysis:

### AUDITABLE Mode
For verification questions requiring formal proof derivation:
- Generates a fact-based derivation with explicit premises
- Each premise is resolved from a specific source (database, LLM knowledge, user input)
- Produces a formal proof with citations and confidence scores
- Code generation with retry logic (up to 3 attempts with error feedback)

**Trigger patterns:** "prove", "verify", "derivation", "true that", "is it true"

### EXPLORATORY Mode
For open-ended data analysis and discovery:
- Step-by-step execution with code generation
- Creates intermediate tables for multi-step analysis
- Supports follow-up questions with session context
- Comparison to previous results available

**Trigger patterns:** "analyze", "explore", "show me", "what is", general questions

### KNOWLEDGE Mode
For questions answerable from general knowledge without data access:
- Direct response using LLM knowledge
- No database or API calls required
- Lower latency responses

**Trigger patterns:** "explain", "how does", "what is a", conceptual questions

### Mode Preservation for Redo Requests

When users request to "redo" an analysis, the system preserves the previous mode:
- Detects redo patterns: "redo", "re-do", "re-run", "rerun", "again", "repeat", "retry"
- If previous session was AUDITABLE, redo stays in AUDITABLE mode
- Extracts any new values from the redo request (e.g., "redo, but change my age to 50")
- Updates facts before re-running the analysis

To explicitly change modes on redo, specify the mode:
- "redo in exploratory mode" - switches to EXPLORATORY for comparison
- "redo and compare to previous" - implies EXPLORATORY

### Mode-Aware Clarification

Personal values (e.g., age, preferences) are handled differently by mode:
- **AUDITABLE mode:** Defers personal values to lazy resolution during premise evaluation
- **EXPLORATORY mode:** Asks for personal values upfront in clarifications (no lazy resolution)

This ensures AUDITABLE proofs never assume personal values - they must be explicitly provided by the user.

---

## Token Savings Estimate

| Scenario | Current | Proposed | Savings |
|----------|---------|----------|---------|
| 5 DBs, 50 tables | ~8,000 tokens | ~500 + ~800 discovered | 84% |
| 10 APIs, 100 ops | ~12,000 tokens | ~500 + ~600 discovered | 91% |
| 20 documents | ~15,000 tokens | ~500 + ~400 discovered | 94% |
| Combined large | ~35,000 tokens | ~500 + ~1,500 discovered | 94% |

The discovered tokens scale with query complexity, not catalog size.
