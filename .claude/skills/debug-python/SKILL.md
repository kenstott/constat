---
name: debug-python
description: Python debugging conventions and diagnostic tools for this project. Auto-triggers when investigating errors or failures.
---

# Debugging Conventions

## Logging
- Module-scoped: `logger = logging.getLogger(__name__)`
- Context tags in debug output: `[PARALLEL]`, `[COMPLEXITY]`, `[DYNAMIC_CONTEXT]`
- No `print()` — always `logger.debug()` / `logger.info()`

## Standard Tools
- `pdb` / `breakpoint()` for interactive debugging
- IPython `embed()` for exploration
- `git bisect` for regression hunting

## DuckDB Diagnostics
- Memory: `duckdb_memory()`
- Query plans: `EXPLAIN ANALYZE`
- Parquet inspection: `parquet_schema()`, `parquet_metadata()`
- Connection issues: check thread-local connections in `duckdb_pool.py`

## DataFrame Diagnostics (Polars)
- Schema: `df.schema`
- Nulls: `df.null_count()`
- Preview: `df.head()`, `df.describe()`
- Find bad rows: filter on null/NaN

## Common Issues
- DuckDB `DROP VIEW IF EXISTS` raises CatalogException if name is a TABLE — use try/except
- DuckDB SIGABRT in tests — suppressed via `addopts = "-p no:faulthandler"`
- ON CONFLICT DO UPDATE for upserts (no savepoints needed)
- Arrow zero-copy: verify schema alignment between DataFrame and DuckDB table

## Investigation Process
1. Reproduce reliably
2. Isolate minimal failing case
3. Observe (stack traces, logs, state)
4. Hypothesize (rank: recent changes > config > data > env > library bug)
5. Verify hypothesis before fixing