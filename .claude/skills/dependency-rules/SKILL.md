---
name: dependency-rules
description: Package governance and dependency management rules. Auto-triggers when adding or modifying dependencies.
---

# Dependency Rules

## Approved Core Dependencies
- `pydantic>=2` — data validation
- `anthropic` — LLM provider
- `duckdb` — analytics engine
- `sqlalchemy>=2` — relational ORM
- `polars` — DataFrame operations
- `numpy` — numerical computing
- `fastapi` — HTTP server
- `uvicorn` — ASGI server

## Banned
- `flask` — use FastAPI
- `django` — use FastAPI
- `pymongo` — use motor (async)

## Optional Extras (in `pyproject.toml`)
- `mysql` — MySQL connector
- `bigquery` — BigQuery support
- `snowflake` — Snowflake support
- `mongodb` — MongoDB via motor
- `elasticsearch` — ES support
- `s3` — S3/MinIO support

## Package Manager
- `pip` with `pyproject.toml` (hatchling build backend)
- Install dev: `pip install -e ".[dev]"`
- No poetry, no conda, no pipenv

## Rules
- No new dependencies without explicit user approval
- Prefer stdlib over third-party when reasonable
- Pin major versions in pyproject.toml
- Check license compatibility (BSL 1.1 project)