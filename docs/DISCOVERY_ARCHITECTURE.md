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

### Proposed Architecture
```
┌─────────────────────────────────────────────────────────┐
│                 Minimal System Prompt                    │
│  "You have access to databases, APIs, and documents.    │
│   Use discovery tools to find what you need."           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │       Planner         │
              │   (with tool access)  │
              └───────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│Schema Discovery│ │ API Discovery │ │ Doc Discovery │
│    Tools      │ │    Tools      │ │    Tools      │
└───────────────┘ └───────────────┘ └───────────────┘
```

**Benefits:**
- Minimal initial tokens (~500 vs 10k+)
- Only loads what's relevant to the query
- Scales to unlimited databases/APIs/documents
- Dynamic fact resolution from documents

---

## Discovery Tool Specifications

### 1. Schema Discovery Tools

#### `list_databases`
Returns available databases with descriptions.

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

## Implementation Plan

### Phase 1: Discovery Tool Infrastructure

1. **Create `constat/discovery/` module**
   ```
   constat/discovery/
   ├── __init__.py
   ├── tools.py          # Tool definitions
   ├── schema_tools.py   # Database/table discovery
   ├── api_tools.py      # API discovery
   ├── doc_tools.py      # Document discovery
   └── fact_tools.py     # Fact resolution
   ```

2. **Define tool schemas for LLM**
   ```python
   DISCOVERY_TOOLS = [
       {
           "name": "list_databases",
           "description": "List all available databases with their types and descriptions",
           "input_schema": {"type": "object", "properties": {}, "required": []}
       },
       {
           "name": "search_tables",
           "description": "Find tables relevant to a natural language query",
           "input_schema": {
               "type": "object",
               "properties": {
                   "query": {"type": "string", "description": "What you're looking for"},
                   "limit": {"type": "integer", "default": 5}
               },
               "required": ["query"]
           }
       },
       # ... etc
   ]
   ```

3. **Implement tool handlers**
   - Wire to existing SchemaManager, APICatalog
   - Add DocumentLoader for document tools

### Phase 2: Planner Integration

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

### Phase 3: Document Loader

1. **Implement DocumentLoader class**
   ```python
   class DocumentLoader:
       def load(self, config: DocumentConfig) -> Document
       def load_file(self, path: str) -> str
       def load_http(self, url: str, headers: dict) -> str
       def load_confluence(self, config: DocumentConfig) -> str
       def extract_pdf(self, path: str) -> str
       def extract_office(self, path: str) -> str
   ```

2. **Add document indexing for search**
   - Chunk documents for embedding
   - Store in vector index alongside schema/API embeddings

### Phase 4: Caching & Optimization

1. **Cache discovery results per session**
   - Don't re-discover same tables
   - Persist discovered context

2. **Smart prefetching**
   - After `search_tables`, prefetch likely schemas
   - Batch embedding lookups

---

## File Changes Summary

| File | Change |
|------|--------|
| `constat/discovery/__init__.py` | New module |
| `constat/discovery/tools.py` | Tool definitions |
| `constat/discovery/schema_tools.py` | Schema discovery implementation |
| `constat/discovery/api_tools.py` | API discovery implementation |
| `constat/discovery/doc_tools.py` | Document discovery implementation |
| `constat/discovery/fact_tools.py` | Fact resolution implementation |
| `constat/documents/loader.py` | Document loading (new) |
| `constat/documents/index.py` | Document vector index (new) |
| `constat/planning/planner.py` | Add tool support, discovery loop |
| `constat/core/config.py` | Already has DocumentConfig |

---

## Token Savings Estimate

| Scenario | Current | Proposed | Savings |
|----------|---------|----------|---------|
| 5 DBs, 50 tables | ~8,000 tokens | ~500 + ~800 discovered | 84% |
| 10 APIs, 100 ops | ~12,000 tokens | ~500 + ~600 discovered | 91% |
| 20 documents | ~15,000 tokens | ~500 + ~400 discovered | 94% |
| Combined large | ~35,000 tokens | ~500 + ~1,500 discovered | 94% |

The discovered tokens scale with query complexity, not catalog size.
