---
name: pytest-patterns
description: Test conventions, tier structure, and pytest patterns for this project. Auto-triggers when writing or reviewing tests.
---

# Test Tier Structure

## Tiers and placement

| Tier | Directory | What belongs here | Allowed I/O |
|------|-----------|-------------------|-------------|
| **unit** | `tests/unit/` | Pure logic, algorithms, data transformations, parsers | None — in-memory only |
| **integration** | `tests/integration/` | Code that talks to a real DB, queue, or external service | Docker-started services only |
| **e2e** | `tests/e2e/` | Full HTTP round-trips through the app via Playwright | Running backend + Vite dev server |

**Currently** most tests live flat in `tests/` — a legacy arrangement. New tests must be placed in the correct tier. Existing flat tests are tolerated but should be migrated when touched.

## The no-skip rule

`pytest.skip()` is **forbidden in all forms**. This includes:
- Infrastructure unavailable (`"Docker not available"`)
- Missing environment variable (`"ANTHROPIC_API_KEY not set"`)
- Missing API key or credentials
- Service not running

A skipped test is a silent lie about coverage — the suite appears green while entire feature areas go untested.

**Wrong (all of these):**
```python
pytest.skip("Docker not available")
pytest.skip("ANTHROPIC_API_KEY not set")
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs key")
```

**Right — infrastructure:** The fixture starts the service via Docker. If Docker is unavailable, call `pytest.fail()`:
```python
if not docker_available:
    pytest.fail("Docker is required. Install Docker and ensure it is running.")
```

**Right — API keys:** Unit tests must mock the API client. Real-API tests are integration tests that require the key to be set in the environment — if it's missing, `pytest.fail()`:
```python
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    pytest.fail("ANTHROPIC_API_KEY must be set for this integration test")
```

A failing test is honest. A skipped test is a lie.

## Unit test rules

- Zero network calls. Zero file I/O (use `tmp_path` only when testing I/O logic directly).
- Zero service dependencies. Any class that hits a DB or network must be mocked at the boundary.
- Must run in < 100ms per test. If it's slow, it belongs in integration/.
- Use `pytest.MonkeyPatch` or `unittest.mock.patch` to isolate boundaries.
- **Mocks are valid for unit tests** — they enable fast CI/CD feedback without services. But a mocked unit test does **not** replace an integration test. Both must exist for any component that talks to a real service.

```python
# Good unit test: pure logic
def test_parse_uri_extracts_host():
    host, port = parse_uri("postgresql://localhost:5432/mydb")
    assert host == "localhost"
    assert port == 5432
```

## Integration test rules

- Services are started via Docker in session-scoped fixtures. The fixture must **fail**, not skip, if Docker is unavailable.
- Each test must clean up its own data (use unique names, `yield`-based cleanup).
- Do not share mutable state across tests. Order must not matter.
- Start containers once per session (session-scoped fixtures), not per test.

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

## E2E test rules

- Playwright only. Each test navigates the real app through a browser.
- Pre-inject localStorage state before `page.reload()` to set session context and accordion expansion — `expandedSectionsVar` is initialized from localStorage at module load.
- Use `scope="class"` session fixtures to share a session across a test class; use isolated sessions for submission tests that mutate state.
- LLM-driven assertions must include a retry loop (up to 3 attempts).

```python
@pytest.fixture(scope="class")
def session_id(server_url):
    sid = str(uuid.uuid4())
    resp = requests.post(f"{server_url}/api/sessions", json={"session_id": sid})
    assert resp.status_code == 200
    yield resp.json()["session_id"]
    requests.delete(f"{server_url}/api/sessions/{sid}")
```

## Pytest fixtures

- Session-scoped for expensive setup (DB connections, container starts, model loading).
- Function-scoped for state that needs per-test cleanup.
- Generator fixtures with `yield` for guaranteed teardown.

```python
@pytest.fixture
def db_conn():
    conn = duckdb.connect()
    yield conn
    conn.close()
```

## Markers

- `@pytest.mark.slow` — long-running tests (>5s)
- `@pytest.mark.integration` — requires Docker / services
- `addopts = "-p no:faulthandler"` in pytest config (suppress DuckDB SIGABRT)

## General patterns

- One assertion per test where practical — when it fails, you know exactly what broke.
- Test behavior, not implementation — tests survive refactoring.
- Use `tmp_path` for temp files in tests that exercise file I/O.
- Parametrize for boundary/equivalence testing:

```python
@pytest.mark.parametrize("input,expected", [(0, True), (-1, False)])
def test_validate(input, expected):
    assert validate(input) == expected
```

## Test count

665+ tests passing. Never remove tests to make the suite pass.
