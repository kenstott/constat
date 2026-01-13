# Future: Federated Query Optimizer

## Overview

Implement cost-based query optimization for heterogeneous data sources, similar to how traditional database query planners work but across databases, APIs, documents, and LLM knowledge.

## Problem

When a user asks "show 10 random countries with currencies", multiple execution plans are possible:

| Plan | Approach | Estimated Cost |
|------|----------|----------------|
| A | Single `llm_ask()` call | ~30s, 1 LLM call |
| B | API → sample → LLM enrich | API latency + 10 small LLM calls |
| C | Local database query | ~10ms, 0 LLM calls |

Currently the planner picks one approach based on heuristics. A proper optimizer would generate alternatives and select the lowest cost.

## Architecture

### 1. Plan Generation
Generate multiple candidate execution plans for each query:
- Direct LLM knowledge
- API fetch + transform
- Database query
- Hybrid (API + LLM enrichment)
- Document lookup + LLM synthesis

### 2. Cost Model
Factors to consider:
- **Latency**: local DB << cached API << API << LLM
- **Token cost**: LLM input/output tokens
- **API costs**: rate limits, per-call pricing
- **Data freshness**: how stale is acceptable?
- **Authority/Trust**: critical dimension beyond speed/cost

### 3. Authority Hierarchy
Data sources have inherent trustworthiness levels:

| Source | Authority | Notes |
|--------|-----------|-------|
| Internal database | Highest | Ground truth for business data |
| Configured APIs | High | Authoritative for their domain |
| Reference documents | High | Authoritative for policies/rules |
| LLM knowledge | Lower | Convenient but may be outdated, can hallucinate |

**Implications**:
- Even if LLM is faster, prefer database for business facts
- LLM is appropriate for: enrichment, world knowledge, formatting
- LLM is risky for: financial figures, compliance data, current prices
- Authority weight should be configurable per domain (e.g., healthcare = high authority requirement)

### 4. Statistics Collection
**Critical insight**: Cost estimates require historical performance data.

Must capture telemetry over time:
- **API latency by endpoint**: `/countries` may be fast, `/countries?details=full` slow
- **LLM latency by prompt type**: simple Q&A vs complex analysis
- **Database latency by query pattern**: indexed lookups vs full scans
- **Cache hit rates**: for repeated queries

Without this data, cost estimates are guesses.

### 5. Plan Selection (LLM-Based, Not Rule-Based)

**Key insight**: Unlike traditional DB optimizers, we don't build deterministic optimization rules in code. Instead:

1. Surface telemetry stats to the planner as context
2. Let the LLM reason probabilistically about tradeoffs

Example context provided to planner:
```
## Source Performance (last 30 days)
- db_sales: p50=12ms, p95=45ms, authority=high
- api_countries: p50=180ms, p95=800ms, authority=high
- llm_ask: p50=2.1s, p95=8.3s, authority=medium, cost=$0.002/call
```

The planner then reasons: "User wants 10 countries. API is 10x faster than LLM and equally authoritative for geographic data. Use API."

**Advantages over traditional optimizer**:
- No complex optimization rules to maintain
- LLM can reason about novel situations
- Naturally handles fuzzy tradeoffs (speed vs authority vs cost)
- Can explain its reasoning
- Smarter about heterogeneous sources than crude federated planners

**Tradeoff**: LLM plan generation is expensive (~seconds, tokens) vs DB planners (~milliseconds). But for complex heterogeneous queries, the smarter plan may save more than the planning cost.

### 6. Data Movement Optimization

Critical for federated queries: **ship the query to the data, not the data to the query**.

**Anti-pattern**:
```
# BAD: Pull 1B rows to join with 10 locally
big_df = pd.read_sql("SELECT * FROM billion_row_table", db_warehouse)
small_df = pd.read_sql("SELECT * FROM small_table", db_local)
result = big_df.merge(small_df, on='id')  # OOM or very slow
```

**Correct pattern**:
```
# GOOD: Push 10 rows to where the billion live
small_ids = pd.read_sql("SELECT id FROM small_table", db_local)
id_list = ','.join(map(str, small_ids['id']))
result = pd.read_sql(f"SELECT * FROM billion_row_table WHERE id IN ({id_list})", db_warehouse)
```

The planner should reason about:
- Estimated row counts per source
- Which source has the large dataset
- Push filters/joins to the large dataset's location
- Only pull the filtered result

### 7. Probe Queries for Dynamic Cardinality

The planner can run cheap exploratory queries before committing to an execution plan:

```python
# Step 1: Probe to understand data sizes
count_a = pd.read_sql("SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'", db_warehouse)
count_b = pd.read_sql("SELECT COUNT(*) FROM local_customers WHERE region = 'EMEA'", db_local)

# Step 2: Now planner knows - 2M orders, 47 customers
# Decision: Push 47 customer IDs to warehouse, not pull 2M orders
```

**Advantages**:
- Avoids catastrophic mistakes (pulling 1B rows into memory)
- Dynamic - works even when cached statistics are stale
- Can include filters to get actual query cardinality, not just table size

**Caveat: COUNT(*) is not always cheap**

Assumption that probing is cheap depends heavily on the source. See database-specific guidance below.

### Database-Specific Probe Characteristics

| Database | Probe Method | Cost | Notes |
|----------|--------------|------|-------|
| **PostgreSQL** | `SELECT COUNT(*) FROM table` | Cheap | Uses `pg_class.reltuples` estimate or index scan. For exact counts with filters, may need seq scan. |
| **MySQL/MariaDB** | `SELECT COUNT(*) FROM table` | Cheap | InnoDB uses clustered index. With `WHERE`, depends on index coverage. |
| **SQLite** | `SELECT COUNT(*) FROM table` | Moderate | No stored stats. Full table scan unless indexed column. |
| **DuckDB (Parquet)** | `SELECT COUNT(*) FROM table` | **Expensive** | Does NOT use Parquet footer stats. Full scan required. |
| **DuckDB (Iceberg)** | `SELECT COUNT(*) FROM table` | **Expensive** | Same issue - ignores manifest file counts. |
| **DuckDB (native)** | `SELECT COUNT(*) FROM table` | Cheap | Uses internal statistics. |
| **MongoDB** | `collection.estimated_document_count()` | Cheap | Uses metadata. `count_documents()` with filter is slower. |
| **Elasticsearch** | `GET /index/_count` | Cheap | Built-in count endpoint. Filters supported. |
| **BigQuery** | `SELECT COUNT(*) FROM table` | **Expensive** | Scans full table (and bills for it). Use `__TABLES__` metadata instead. |
| **Snowflake** | `SELECT COUNT(*) FROM table` | Moderate | Uses micro-partition pruning. Metadata queries cheaper. |
| **Redshift** | `SELECT COUNT(*) FROM table` | Moderate | `SVV_TABLE_INFO` for estimates without scanning. |

**Recommended probe strategies by database:**

```python
# PostgreSQL - use pg_class for fast estimates
estimate = pd.read_sql("""
    SELECT reltuples::bigint AS estimate
    FROM pg_class WHERE relname = 'orders'
""", db_postgres).iloc[0, 0]

# MySQL - use information_schema
estimate = pd.read_sql("""
    SELECT TABLE_ROWS FROM information_schema.TABLES
    WHERE TABLE_NAME = 'orders'
""", db_mysql).iloc[0, 0]

# BigQuery - use __TABLES__ metadata (free, no scan)
estimate = pd.read_sql("""
    SELECT row_count FROM `project.dataset.__TABLES__`
    WHERE table_id = 'orders'
""", db_bigquery).iloc[0, 0]

# MongoDB - use estimated count (fast)
estimate = db_mongo.orders.estimated_document_count()

# DuckDB/Parquet - DON'T probe, use cached stats or assume large
# Probing is as expensive as the query itself
```

**File format considerations:**

| Format | Has Row Count Metadata | Accessible Without Full Scan |
|--------|------------------------|------------------------------|
| Parquet | Yes (footer) | Yes, but DuckDB ignores it |
| Iceberg | Yes (manifest) | Yes, but DuckDB ignores it |
| CSV | No | No - must count lines |
| JSON/JSONL | No | No - must parse |
| Arrow/Feather | Yes (schema) | Yes |

**What telemetry should track:**
- `db_warehouse.count_latency_p50 = 12ms` (has good stats)
- `db_duckdb_parquet.count_latency_p50 = 45s` (full scan, don't probe!)
- `api_countries.probe_capable = false` (no count operation)

**Fallbacks when probing isn't possible**:
- Use cached row counts from previous full fetches
- Use LLM knowledge for stable domains ("~195 countries")
- Accept uncertainty, use conservative strategy (assume large)
- Paginate with early termination if results exceed threshold

**Hard limits as safety net**:
- Always use `LIMIT` clause with a max (e.g., 100K rows)
- Abort on load if result exceeds memory/row threshold
- Better to fail fast with clear error than OOM or hang

```python
MAX_ROWS = 100000
df = pd.read_sql(f"SELECT * FROM big_table WHERE ... LIMIT {MAX_ROWS + 1}", db)
if len(df) > MAX_ROWS:
    raise ValueError(f"Query exceeded {MAX_ROWS} row limit. Add filters or use aggregation.")
```

Even with smart planning, estimates can be wrong. Hard limits ensure graceful failure.

The planner needs to know which sources are probe-friendly vs probe-expensive vs probe-impossible.

**When to probe**:
- Cross-database joins
- Unknown/untrusted cardinality estimates
- When the cost of getting it wrong is high (OOM, timeout)

### 8. Self-Contained Adaptive Join Scripts

Probing and execution should be in a **single Python script** that decides strategy at runtime:

```python
# Probe both sides
count_left = pd.read_sql("SELECT COUNT(*) FROM warehouse.orders WHERE date > '2024-01-01'", db_warehouse).iloc[0,0]
count_right = pd.read_sql("SELECT COUNT(*) FROM local.customers WHERE region = 'EMEA'", db_local).iloc[0,0]

THRESHOLD = 10000  # Max rows to pull across network

if count_left <= THRESHOLD and count_right <= THRESHOLD:
    # Both small - pull both, join locally
    left = pd.read_sql("SELECT * FROM warehouse.orders WHERE date > '2024-01-01'", db_warehouse)
    right = pd.read_sql("SELECT * FROM local.customers WHERE region = 'EMEA'", db_local)
    result = left.merge(right, on='customer_id')

elif count_right < count_left:
    # Push right (smaller) to left (larger)
    right_ids = pd.read_sql("SELECT customer_id FROM local.customers WHERE region = 'EMEA'", db_local)
    id_list = ','.join(map(str, right_ids['customer_id']))
    result = pd.read_sql(f"""
        SELECT o.*, c.* FROM warehouse.orders o
        JOIN warehouse.customers c ON o.customer_id = c.id
        WHERE o.date > '2024-01-01' AND o.customer_id IN ({id_list})
    """, db_warehouse)

else:
    # Push left (smaller) to right (larger)
    # ... opposite strategy
```

**Why single script**:
- Decision made at runtime with fresh probe data
- No round-trip back to planner between probe and execute
- Strategy adapts to actual data, not stale estimates
- Atomic - no partial execution if strategy changes mid-plan

### 9. GraphQL Schema Introspection

GraphQL lacks standardized filtering, but provides introspection. Use a two-layer approach (same as database schemas):

| Layer | What's Cached | When |
|-------|---------------|------|
| **Overview** | Query/mutation names, descriptions | At startup |
| **Detail** | Arguments, filter types, return schema | On-demand via tool |

**Config:**
```yaml
apis:
  countries:
    type: graphql
    url: https://countries.trevorblades.com/graphql
    introspect: true  # Fetch schema at startup
```

**Startup introspection (cached):**
```graphql
query IntrospectionQuery {
  __schema {
    queryType { fields { name description } }
    mutationType { fields { name description } }
  }
}
```

**On-demand tool for code generator:**
```python
# Tool: get_graphql_query_schema(api_name, query_name)
# Returns detailed schema for a specific query

schema = get_graphql_query_schema('countries', 'countries')
# Returns:
# {
#   "name": "countries",
#   "args": [
#     {"name": "filter", "type": "CountryFilterInput", "fields": [
#       {"name": "code", "type": "StringQueryOperatorInput"},
#       {"name": "continent", "type": "StringQueryOperatorInput"}
#     ]}
#   ],
#   "returns": "Country",
#   "return_fields": ["code", "name", "currency", "continent", ...]
# }
```

**Why this matters:**
- Planner can push filters to API instead of fetching all + filtering client-side
- Reduces data transfer (fetch 10 EU countries vs fetch 250 then filter)
- Enables efficient pagination

**Common filter patterns to recognize:**
| Pattern | Example | Framework |
|---------|---------|-----------|
| Hasura-style | `filter: { continent: { eq: "EU" } }` | Hasura |
| Prisma-style | `where: { continent: "EU" }` | Prisma/Nexus |
| Argument-style | `countries(continent: "EU")` | Custom |
| Relay-style | `first: 10, after: "cursor"` | Relay |

The introspected schema reveals which pattern the API uses.

## Implementation Phases

### Phase 1: Telemetry Collection
- Add timing instrumentation to all data source calls
- Store in local metrics table: `source, endpoint, latency_ms, tokens, timestamp`
- No optimization yet, just collect data

### Phase 2: Cost Model
- Define cost formula: `cost = w1*latency + w2*tokens + w3*api_calls`
- Build statistical summaries from telemetry (p50, p95 latency by source/endpoint)
- Allow user-configurable weights (optimize for speed vs cost)

### Phase 3: Plan Generation
- Planner generates 2-3 candidate plans instead of one
- Each plan tagged with estimated cost

### Phase 4: Adaptive Selection
- Select plan based on cost estimates
- Track actual vs estimated cost
- Adjust model based on prediction errors

## Open Questions

1. How to handle cold start (no historical data)?
   - Use conservative defaults
   - Prefer authoritative sources over LLM initially

2. How granular should endpoint tracking be?
   - `GET /countries` vs `GET /countries?region=europe`
   - Query params can dramatically affect latency

3. Should users see plan alternatives?
   - "I chose Plan B (API) because it's 3x faster than LLM for this query"

4. How to handle cost vs accuracy tradeoffs?
   - LLM might be faster but less accurate than authoritative API
   - Some queries need accuracy, others just need ballpark

## Related Work

- PostgreSQL query planner: statistics, cost estimation, plan selection
- Apache Spark Catalyst: rule-based + cost-based optimization
- Trino/Presto: federated query optimization across heterogeneous sources
