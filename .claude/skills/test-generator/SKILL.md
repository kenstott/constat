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

| Module | Source Root | Test Pattern | Notes |
|---|---|---|---|
| `api` | `constat/api/` | `tests/test_api_*.py` | API factory, protocol, implementation |
| `catalog` | `constat/catalog/` | `tests/test_*glossary*.py`, `tests/test_schema*.py` | Schema discovery, API catalog, glossary |
| `commands` | `constat/commands/` | `tests/test_commands.py` | Shared REPL/UI command handlers |
| `core` | `constat/core/` | `tests/test_models.py`, `tests/test_config.py` | Config, models, types, domain tiers |
| `discovery` | `constat/discovery/` | `tests/test_discovery*.py`, `tests/test_doc_tools.py`, `tests/test_hybrid_search.py` | Schema/API/doc/fact discovery, vector store |
| `execution` | `constat/execution/` | `tests/test_dag.py`, `tests/test_executor.py`, `tests/test_fact_resolver.py` | Planning, code execution, DAG |
| `learning` | `constat/learning/` | `tests/test_exemplar*.py` | Rule compaction, fine-tune |
| `llm` | `constat/llm/` | `tests/test_llm*.py` | LLM primitives (enrich/score/summarize) |
| `providers` | `constat/providers/` | `tests/test_providers*.py` | LLM provider integrations |
| `server` | `constat/server/` | `tests/test_server*.py`, `tests/integration/` | FastAPI server, routes, websocket, GraphQL |
| `session` | `constat/session/` | `tests/test_session*.py`, `tests/test_auditable*.py` | Session orchestration (12+ mixins) |
| `storage` | `constat/storage/` | `tests/test_duckdb*.py`, `tests/test_datastore*.py`, `tests/test_split_store*.py` | DuckDB session store, vector backend, history |
| `repl` | `constat/repl/` | `tests/test_repl*.py` | REPL/CLI interface |
| `textual_repl` | `constat/textual_repl/` | `tests/test_textual*.py` | Textual TUI interface |
| `visualization` | `constat/visualization/` | `tests/test_viz*.py`, `tests/test_output*.py` | Chart/output rendering |
| `testing` | `constat/testing/` | — | Test infrastructure (not tested itself) |
| `prompts` | `constat/prompts/` | `tests/test_prompt*.py` | Prompt/template loading |

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
| **Skipif guards** | `@pytest.mark.skipif(not os.environ.get(...))` | LLM-dependent tests |
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

- **Location**: `tests/test_<name>.py`
- **Marker**: None (default) or `@pytest.mark.slow` for LLM-touching tests
- **Scope**: Single class or function in isolation
- **Dependencies**: Mocked (unittest.mock or pytest-mock)
- **Database**: In-memory DuckDB (`duckdb.connect()`) or `tmp_path` file
- **Run**: `python -m pytest tests/test_<name>.py -x -q`

### Integration Tests

- **Location**: `tests/integration/test_<name>.py`
- **Marker**: `pytestmark = pytest.mark.integration` (module-level)
- **Scope**: Multiple components working together with real dependencies
- **Dependencies**: Real server process, real database, real HTTP calls
- **Fixtures**: From `tests/integration/conftest.py` — `server_url`, `server_port`, `integration_data_dir`
- **Guards**: `@pytest.mark.requires_docker`, `@pytest.mark.requires_postgresql`, etc.
- **Run**: `python -m pytest tests/integration/ -v`

Integration test fixtures provide:
```python
@pytest.fixture(scope="session")
def server_process(server_port, integration_data_dir):
    """Spawns real constat server on dynamic port."""
    # python -m constat.cli serve -c demo/config.yaml --port {port}
    # Waits for HTTP readiness + warmup completion
    ...

@pytest.fixture(scope="session")
def server_url(server_port):
    return f"http://localhost:{server_port}"
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

    def test_glossary_crud(self, server_url):
        # Create → Read → Update → Delete lifecycle
        ...
```

### E2E Tests (Playwright)

- **Location**: `tests/integration/test_*_ui.py`
- **Marker**: `pytestmark = pytest.mark.integration` (same marker, UI suffix distinguishes)
- **Scope**: Full browser-driven user workflows
- **Dependencies**: Real server + Vite dev server + Chromium browser
- **Fixtures**: From `tests/integration/conftest.py` — `page`, `browser_context`, `ui_url`
- **Run**: `python -m pytest tests/integration/test_*_ui.py -v`

E2E fixtures provide:
```python
@pytest.fixture(scope="session")
def browser_context():
    """Headless Chromium via Playwright."""
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        browser.close()

@pytest.fixture
def page(browser_context):
    """Fresh browser page per test."""
    page = browser_context.new_page()
    yield page
    page.close()

@pytest.fixture(scope="session")
def ui_url(ui_port):
    return f"http://localhost:{ui_port}"
```

E2E test structure:
```python
import pytest

pytestmark = pytest.mark.integration

class TestGlossaryUI:
    def test_glossary_panel_loads(self, page, ui_url):
        page.goto(f"{ui_url}")
        page.wait_for_selector("[data-testid='glossary-panel']")
        assert page.locator("[data-testid='glossary-panel']").is_visible()

    def test_add_glossary_term(self, page, ui_url):
        page.goto(f"{ui_url}")
        page.click("[data-testid='add-term-button']")
        page.fill("[data-testid='term-name-input']", "Revenue")
        page.fill("[data-testid='term-definition-input']", "Total sales amount")
        page.click("[data-testid='save-term-button']")
        page.wait_for_selector("text=Revenue")
        assert page.locator("text=Revenue").is_visible()
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
| Database reads/writes (DuckDB, SQLite) | Yes (in-memory) | Yes (if complex queries) | No |
| HTTP/API endpoints (FastAPI routes) | Yes (TestClient) | Yes (real server) | No |
| WebSocket handlers | Yes (mock) | Yes (real server) | No |
| UI-facing features (glossary, artifacts) | No | Maybe | Yes |
| LLM provider calls | Yes (mocked) | Skip (needs API key) | No |
| External service calls (Docker DBs) | Yes (mocked) | Yes (with marker) | No |

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
- Use `server_url` fixture for HTTP calls
- Use `requests` or `httpx` for API calls
- Add appropriate `@pytest.mark.requires_*` markers for external services
- Test real interactions (not mocked)

#### If Generating E2E Tests

Create `tests/integration/test_<feature_name>_ui.py`:

- Set `pytestmark = pytest.mark.integration` at module level
- Use `page` and `ui_url` fixtures from integration conftest
- Use Playwright selectors (`data-testid` preferred, then `text=`, then CSS)
- Wait for elements before asserting (`wait_for_selector`)
- Test user-visible workflows end-to-end
- Keep tests independent (fresh `page` per test via fixture)

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
