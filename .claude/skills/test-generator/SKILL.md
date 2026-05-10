---
description: "Systematic test generation with gap analysis, fix loops, and checkpoint tracking for constat Python modules"
model: sonnet
effort: high
---

# Test Generator

Systematically generate tests for a constat module: $ARGUMENTS

## Argument Parsing (Step 0)

`$ARGUMENTS` can be either **structured** or **natural language**.

### Structured Format

`<module-name> [flags...]`

- First token = module name (from lookup table below)
- Remaining tokens = flags:
  - `--coverage` — run pytest-cov after generation
  - `--unit-only` — skip integration and e2e test generation
  - `--integration-only` — generate only integration tests
  - `--e2e-only` — generate only Playwright e2e tests
  - `--resume` — resume from existing checkpoint file
  - `--dry-run` — perform gap analysis only, do not generate tests

### Natural Language Format

If `$ARGUMENTS` does not start with a known module name or flag, treat the entire string as a natural language prompt. Extract:

1. **Module name** — look for any known module name mentioned anywhere in the text (e.g., "the storage module" → `storage`, "session code" → `session`)
2. **Intent** — map the user's intent to flags:
   - "verify", "check", "analyze", "audit", "assess" → `--dry-run` (gap analysis only, unless the text also asks to create/generate)
   - "create", "generate", "add", "write", "build" tests → full generation (no `--dry-run`)
   - "verify ... and if needed, create" or "check ... and fix gaps" → full generation
   - "resume", "continue", "pick up where" → `--resume`
   - "unit tests only", "no integration" → `--unit-only`
   - "integration tests", "integration only" → `--integration-only`
   - "e2e tests", "browser tests", "playwright" → `--e2e-only`
   - "coverage" → `--coverage`
3. **Scope narrowing** — if the user mentions specific classes, files, or patterns (e.g., "focus on the vector store"), filter the gap list to match after Step 3

If no module name can be identified, report an error and list the known modules.

### Examples

| Input | Parsed As |
|---|---|
| `storage` | module=storage |
| `session --dry-run --coverage` | module=session, flags=--dry-run --coverage |
| `verify that the storage module has full test coverage and create tests for gaps` | module=storage, full generation |
| `check test coverage for the execution module` | module=execution, --dry-run |
| `generate unit tests for discovery, skip integration` | module=discovery, --unit-only |
| `resume test generation for server with coverage` | module=server, --resume --coverage |
| `create playwright tests for the glossary UI` | module=server, --e2e-only, scope filter=glossary |
| `focus on the vector store and relational store` | module=storage, full generation, scope filter=*vector*, *relational* |

## Module Resolution (Step 1)

Map the module name to its source root, test root, and sub-packages.

### Known Module Lookup Table

| Module | Source Root | Unit Tests | Integration/E2E |
|---|---|---|---|
| `api` | `constat/api/` | `tests/unit/test_api_*.py` | — |
| `catalog` | `constat/catalog/` | `tests/unit/test_*glossary*.py`, `tests/unit/test_schema*.py` | — |
| `commands` | `constat/commands/` | `tests/unit/test_commands.py` | — |
| `core` | `constat/core/` | `tests/unit/test_models.py`, `tests/unit/test_config.py` | — |
| `discovery` | `constat/discovery/` | `tests/unit/test_discovery*.py`, `tests/unit/test_doc_tools.py` | — |
| `execution` | `constat/execution/` | `tests/unit/test_dag.py`, `tests/unit/test_executor.py` | — |
| `learning` | `constat/learning/` | `tests/unit/test_exemplar*.py` | — |
| `llm` | `constat/llm/` | `tests/unit/test_llm*.py` | — |
| `providers` | `constat/providers/` | `tests/unit/test_providers*.py` | — |
| `server` | `constat/server/` | `tests/unit/test_server*.py` | `tests/integration/`, `tests/e2e/` |
| `session` | `constat/session/` | `tests/unit/test_session*.py` | — |
| `storage` | `constat/storage/` | `tests/unit/test_duckdb*.py`, `tests/unit/test_datastore*.py` | `tests/integration/` |
| `repl` | `constat/repl/` | `tests/unit/test_repl*.py` | — |
| `visualization` | `constat/visualization/` | `tests/unit/test_viz*.py` | — |
| `testing` | `constat/testing/` | — | — |
| `prompts` | `constat/prompts/` | `tests/unit/test_prompt*.py` | — |

> **Legacy:** Flat tests in `tests/test_*.py` exist from before the tier structure. Tolerated but new tests go in the correct subdirectory.

### Fallback Convention (Unknown Modules)

If the module name is not in the table:
- Source root: `constat/<module-name>/`
- Test root: `tests/`
- Test pattern: `tests/test_<module-name>*.py`

Verify the source directory exists before proceeding. If it does not exist, report an error.

## Checkpoint Management (Step 2)

### Checkpoint File Location

`tests/TEST_PROGRESS_<MODULE>.md`

Example: `tests/TEST_PROGRESS_storage.md`

### If `--resume` Flag is Set

1. Read the existing `TEST_PROGRESS_<MODULE>.md`
2. Parse the tables to identify:
   - **Completed** classes (skip these)
   - **Failed** classes (retry these)
   - **Remaining** classes (process these)
3. Resume from the first unprocessed class

### If Starting Fresh (No `--resume` or No Checkpoint)

Create initial checkpoint:

```markdown
# Test Generation Progress: <Module Name>

**Started**: <date>
**Module**: <module-name>
**Source Root**: <source-root>
**Flags**: <flags>

## Summary

| Metric | Count |
|---|---|
| Source Files | <N> |
| Already Covered | <N> |
| Gaps Found | <N> |
| Tests Generated | 0 |
| Tests Passing | 0 |
| Tests Failed | 0 |

## Completed

| Source File | Test File | Status | Notes |
|---|---|---|---|

## Failed

| Source File | Test File | Attempts | Error |
|---|---|---|---|

## Remaining

| Source File | Priority | Reason |
|---|---|---|
```

## Gap Analysis (Step 3)

### Pass 1: Collect Source Files

Use `Glob` to find all `.py` files under the source root. Exclude:
- `__init__.py` (unless it contains substantial logic — >50 lines of non-import code)
- `__pycache__/`
- Files containing only type aliases or re-exports

Record each file with its module path (e.g., `constat.storage.duckdb_backend`).

### Pass 2: Collect Existing Test Files

Use `Glob` to find all test files matching the module's test pattern.

For each test file, determine which source files it covers by:
1. Reading imports — `from constat.<module>.<file> import ...` directly maps to source
2. Reading test class/function names — `TestDuckDBBackend` covers `duckdb_backend.py`
3. Reading docstrings — `"""Tests for vector_store module."""` covers `vector_store.py`

Build a set of covered source files.

### Pass 3: Identify Gaps and Prioritize

Gap = source files NOT in the covered set.

Rank gaps by priority (highest first):

| Priority | Pattern | Rationale |
|---|---|---|
| 1 | `*_store.py`, `*_backend.py` | Data persistence — must work |
| 2 | `*_manager.py`, `*_builder.py` | Core orchestration |
| 3 | `models.py`, `config.py`, `types.py` | Shared data structures |
| 4 | `*_engine.py`, `*_planner.py` | Query/execution logic |
| 5 | `routes/*.py`, `app.py` | API surface |
| 6 | `*_provider.py` | External integrations |
| 7 | `*_loader.py`, `*_parser.py` | Data pipeline |
| 8 | Everything else | Remaining |

### Report to User

**Before generating any tests**, print the gap analysis:

```
=== Gap Analysis: <module> ===

Source files:      <N>
Already covered:   <N>
Gaps found:        <N>

Priority 1 (Stores/Backends):
  - duckdb_backend.py
  - split_store.py

Priority 2 (Managers/Builders):
  - schema_manager.py

...

Proceed with test generation? (Y/n)
```

If `--dry-run` is set, stop here after printing the gap analysis.

## Detect Module Test Pattern (Step 4)

Read 2-3 existing test files for this module to determine conventions. Check for:

### Pattern Detection Rules

| Feature | Check | Example |
|---|---|---|
| **Mock style** | `unittest.mock` vs `pytest-mock` (`mocker` fixture) | Most constat tests use `unittest.mock` |
| **Async tests** | `@pytest.mark.asyncio` + `async def test_*` | 82 tests use asyncio |
| **Class grouping** | `class Test*:` groups vs standalone functions | Both used, prefer classes for related tests |
| **Fixture usage** | `tmp_path`, `tmp_path_factory`, custom fixtures | Module-specific fixtures in conftest |
| **Parametrize** | `@pytest.mark.parametrize` for data-driven tests | Used across modules |
| **DuckDB patterns** | `duckdb.connect()` with yield teardown | Storage module tests |
| **Docker markers** | `@pytest.mark.requires_*` for external services | Provider/integration tests |
| **API key tests** | Real-API integration tests require key in env; use `pytest.fail()` if absent, never `pytest.skip()` | Integration tests only — unit tests must mock the client |
| **xfail** | `@pytest.mark.xfail(reason="LLM non-deterministic")` | Session/LLM tests |

Store detected patterns for use in generation.

### Module-Specific Pattern Defaults

| Module | Pattern |
|---|---|
| `storage` | DuckDB fixtures, `tmp_path` for DB files, yield teardown, no mocks for DB layer |
| `session` | Heavy mocking of providers/storage, `@pytest.mark.asyncio`, class-based grouping |
| `execution` | Pure logic tests, parametrize for DAG scenarios, minimal fixtures |
| `server` | `httpx.AsyncClient` or `TestClient`, `@pytest.mark.asyncio`, mock session deps |
| `providers` | Mock HTTP responses, `@pytest.mark.skipif` for API keys, async tests |
| `discovery` | Vector store fixtures (session-scoped), embedding mocks, `tmp_path` |
| `core` | Pure dataclass/config tests, parametrize, no fixtures needed |
| `catalog` | Mock schema sources, parametrize for SQL dialects |

## Test Type Vocabulary (Step 4b)

### Unit Tests

- **Location**: `tests/unit/test_<name>.py`
- **Marker**: None (default) or `@pytest.mark.slow` for LLM-touching tests
- **Scope**: Single class or function in isolation. Zero I/O — no network, no file system (except `tmp_path` when testing I/O logic directly), no DB connections
- **Dependencies**: Mocked at boundaries (unittest.mock or pytest-mock)
- **Database**: In-memory DuckDB (`duckdb.connect()`) only — not a real file DB
- **Run**: `python -m pytest tests/unit/ -x -q`

### Integration Tests

- **Location**: `tests/integration/test_<name>.py`
- **Marker**: `pytestmark = pytest.mark.integration` (module-level)
- **Scope**: Multiple components working together with real services (DB, queues, external APIs). Does NOT require a running HTTP app server — that's e2e.
- **Dependencies**: Docker-started containers (MongoDB, Elasticsearch, Cassandra, PostgreSQL). If Docker is unavailable the fixture calls `pytest.fail()`, never `pytest.skip()`.
- **Fixtures**: From `tests/integration/conftest.py` — `server_url`, `server_port`, `integration_data_dir`
- **Run**: `python -m pytest tests/integration/ -x -q`

Integration fixture rule — **never skip, always fail**:
```python
@pytest.fixture(scope="session")
def elasticsearch_container():
    if not is_docker_available():
        pytest.fail("Docker is required. Run: docker info")
    if not start_container("constat_test_es", "elasticsearch:8.13.0", "9200:9200"):
        pytest.fail("Elasticsearch container failed to start")
    yield {"host": "localhost", "port": 9200}
    stop_container("constat_test_es")
```

Integration test structure:
```python
import pytest
import requests

pytestmark = pytest.mark.integration

class TestGlossaryAPI:
    def test_graphql_introspection(self, server_url):
        query = {"query": "{ __schema { queryType { name } } }"}
        resp = requests.post(f"{server_url}/api/graphql", json=query)
        assert resp.status_code == 200
```

### E2E Tests (Playwright)

- **Location**: `tests/e2e/test_<panel_or_feature>.py` — organized by UI panel or user workflow
- **Marker**: `pytestmark = pytest.mark.e2e` (module-level)
- **Scope**: Full browser-driven user workflows through the running app
- **Dependencies**: Running backend server + Vite dev server + Chromium browser
- **Fixtures**: From `tests/e2e/conftest.py` (or `tests/integration/conftest.py` until migrated) — `page`, `browser_context`, `ui_url`
- **Run**: `python -m pytest tests/e2e/ -x -q`

> **Note:** Existing Playwright tests live in `tests/integration/test_*_ui.py` — a legacy location. New e2e tests go in `tests/e2e/`. Migrate existing ones when touched.

E2E test structure:
```python
import pytest

pytestmark = pytest.mark.e2e

class TestGlossaryUI:
    def test_glossary_panel_loads(self, page, ui_url):
        page.goto(f"{ui_url}")
        page.wait_for_selector("#section-glossary")
        assert page.locator("#section-glossary").is_visible()
```

## Generate Test Files (Step 5)

For each gap file (in priority order):

### 5a. Read Source File

Read the source `.py` file to identify:
- Classes and their public methods
- Module-level functions
- `__init__` signatures and dependencies
- Exception paths (raise statements)
- Async methods (`async def`)
- Dataclass/Pydantic model fields
- Context managers (`__enter__`/`__exit__` or `@contextmanager`)

### 5b. Determine Test Type

For each source file, determine which test types to generate:

| Source Characteristic | Unit Test | Integration Test | E2E Test |
|---|---|---|---|
| Pure logic, dataclasses, utilities | Yes | No | No |
| Database reads/writes (DuckDB, SQLite) | Yes (in-memory mock) | **Yes — required** | No |
| HTTP/API endpoints (FastAPI routes) | Yes (TestClient mock) | **Yes — required** (real server) | No |
| WebSocket handlers | Yes (mock) | **Yes — required** (real server) | No |
| UI-facing features (glossary, artifacts) | No | No | **Yes — required** |
| LLM provider calls | Yes (mocked client) | **Yes — required** if real API path exists | No |
| External service calls (Elasticsearch, MongoDB, etc.) | Yes (mocked) | **Yes — required** (Docker-started service) | No |

**Rule:** A mocked unit test does NOT replace the integration test. Both rows must be filled for any component that touches a real service. The unit test gives fast CI feedback; the integration test proves the real boundary works.

### 5c. Generate Unit Test File

Create `tests/test_<source_file_name>.py`.

#### Template Structure

```python
# Copyright (c) Kenneth Stott. All rights reserved.
# Licensed under the Business Source License 1.1.
from __future__ import annotations

<stdlib-imports>

<third-party-imports>

<local-imports>


class Test<ClassName>:
    """Tests for <ClassName>."""

    <test-methods>
```

#### File Header

Always use the standard constat header:
```python
# Copyright (c) Kenneth Stott. All rights reserved.
# Licensed under the Business Source License 1.1.
from __future__ import annotations
```

#### Import Rules

- `from __future__ import annotations` — always first
- stdlib: `import os`, `from pathlib import Path`, `from unittest.mock import Mock, patch, AsyncMock`
- third-party: `import pytest`, `import duckdb` (only if needed)
- local: `from constat.<module>.<file> import <Class>` (absolute imports only)

#### Python 3.11+ Style Checklist

All generated tests MUST use modern Python:
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `X | None` not `Optional[X]`
- `match/case` where appropriate
- No `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union`

#### Test Method Guidelines

- One `def test_*` method per public method on the source class (at minimum)
- Additional methods for edge cases, None inputs, exception paths
- Method names: `test_<method_name>_<scenario>` (e.g., `test_get_schema_returns_none_for_missing`)
- Each test follows Arrange-Act-Assert pattern
- Use plain `assert` statements (not `self.assertEqual`)
- Use `pytest.raises` for expected exceptions
- Use `@pytest.mark.asyncio` for async tests with `async def test_*`
- Use `@pytest.mark.parametrize` for data-driven tests

#### If Generating Integration Tests

Create `tests/integration/test_<feature_name>.py`:

- Set `pytestmark = pytest.mark.integration` at module level
- Use `server_url` fixture for HTTP/GraphQL calls
- Use `requests` or `httpx` for API calls
- If a test needs a service (Elasticsearch, MongoDB, etc.), start it via Docker in the fixture — call `pytest.fail()` if Docker is unavailable, **never** `pytest.skip()`
- Test real interactions (not mocked)

#### If Generating E2E Tests

Create `tests/e2e/test_<panel_or_feature>.py`:

- Set `pytestmark = pytest.mark.e2e` at module level
- Use `page` and `ui_url` fixtures
- Use Playwright selectors (prefer `#section-<id>`, `button[title='...']`, `text=` over fragile CSS)
- Pre-inject localStorage state before `page.reload()` to set session context and expand accordion sections
- Wait for elements before asserting (`wait_for_selector`)
- Test user-visible workflows end-to-end
- Keep tests independent (fresh `page` per test via fixture)
- LLM-driven assertions must include a retry loop (up to 3 attempts)

### 5d. Write the File

Use the `Write` tool to create the test file at the correct path.

## Run and Fix Loop (Step 6)

After writing each test file:

### 6a. Run the Test

```bash
python -m pytest tests/test_<name>.py -x -q 2>&1
```

For integration tests:
```bash
python -m pytest tests/integration/test_<name>.py -v 2>&1
```

### 6b. If Test Fails

Parse the failure output to identify:
- `ImportError` — wrong import path, missing dependency
- `AttributeError` — wrong method name or signature
- `TypeError` — wrong argument count or types
- `AssertionError` — wrong expected values
- `ModuleNotFoundError` — missing package
- `DuckDB CatalogException` — wrong table/column name

Fix the test file using `Edit` tool. Common fixes:
- Fix import paths
- Update method signatures to match actual API
- Replace private member access with public interface
- Fix mock setup (return values, side effects)
- Add missing fixtures

### 6c. Retry Limit

Up to **3 fix iterations** per test file (run + fix = 1 iteration).

If still failing after 3 attempts:
1. Mark as **FAILED** in checkpoint
2. Log the error summary
3. Move to the next gap file

## Update Checkpoint (Step 7)

After processing each file, update `TEST_PROGRESS_<MODULE>.md`:

### On Success

Move from Remaining to Completed:
```markdown
| source_file.py | test_source_file.py | PASS | 5 tests, 0 failures |
```

### On Failure

Move from Remaining to Failed:
```markdown
| source_file.py | test_source_file.py | 3 | ImportError: cannot import ... |
```

### Update Summary Counts

Recalculate all counters in the Summary table.

## Post-Generation Summary (Step 8)

Print a final summary:

```
=== Test Generation Complete: <module> ===

Files processed:   <N>
Tests passing:     <N>
Tests failed:      <N>
New test files:    <N>

New files created:
  - tests/test_new1.py
  - tests/test_new2.py

Run all generated tests:
  python -m pytest tests/test_<module>*.py -x -q

Run integration tests:
  python -m pytest tests/integration/ -v

Checkpoint saved: tests/TEST_PROGRESS_<MODULE>.md
```

## Optional Coverage Report (Step 9)

If `--coverage` flag was provided:

```bash
python -m pytest tests/test_<module>*.py --cov=constat/<module> --cov-report=term-missing -q 2>&1
```

Report a summary of coverage percentages.

## Example Invocations

```
/test-generator storage
/test-generator session --unit-only
/test-generator server --dry-run
/test-generator execution --resume
/test-generator storage --coverage
/test-generator server --e2e-only
/test-generator check what tests are missing for the discovery module
/test-generator create playwright tests for glossary
```

## Critical Rules

1. **Python 3.11+ strict** — use modern type hints, `match/case` OK, no legacy `typing` imports
2. **Copyright header on every file** — `# Copyright (c) Kenneth Stott. All rights reserved.`
3. **`from __future__ import annotations`** — always first import
4. **Report gaps before generating** — always show the user the gap list first
5. **Checkpoint after every file** — progress must survive interruption
6. **3-attempt limit** — do not loop forever on a failing test
7. **Do not remove existing tests** — only add new test files
8. **Use module conventions** — match the style of existing tests for that module
9. **No fallback values or silent error handling** — follows project CLAUDE.md
10. **Absolute imports only** — `from constat.<module>.<file> import ...`
11. **No Parquet references** — all storage is native DuckDB tables
12. **Mock LLM calls** — never make real API calls in unit tests
13. **`tmp_path` for temp files** — never write to the project directory in tests
14. **Test tier placement is mandatory** — unit → `tests/unit/`, integration → `tests/integration/`, e2e → `tests/e2e/`
15. **Never `pytest.skip()` for infra** — if a service is required, start it via Docker or call `pytest.fail()`. Skipped infra tests are silent coverage lies.
