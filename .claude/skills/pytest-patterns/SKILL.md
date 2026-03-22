---
name: pytest-patterns
description: Test conventions and pytest patterns for this project. Auto-triggers when writing or reviewing tests.
---

# Pytest Conventions

## Structure
- Tests in `tests/`, files: `test_*.py`, functions: `test_*`
- Fixtures in `conftest.py`
- Run: `python -m pytest tests/ -x -q` (~7 min full suite)

## Fixtures
- Session-scoped for expensive setup (DB connections, model loading)
- Function-scoped for state that needs cleanup
- Generator fixtures with `yield` for teardown:
```python
@pytest.fixture
def db_conn():
    conn = duckdb.connect()
    yield conn
    conn.close()
```

## Markers
- `@pytest.mark.slow` — long-running tests
- `@pytest.mark.skipif` — conditional skip
- `addopts = "-p no:faulthandler"` in pytest config (suppress DuckDB SIGABRT)

## Known Failures
3 pre-existing: `test_simple_sum` (LLM-dependent), `test_elasticsearch` (infra), `test_load_pdf_via_http_content_type` (SSL)

## Test Count
665+ tests passing. Never remove tests to make the suite pass.

## Patterns
- One assertion per test where practical
- Test behavior, not implementation
- Mock external dependencies (LLM calls, network)
- Use `tmp_path` fixture for temp files
- Parametrize for boundary/equivalence testing:
```python
@pytest.mark.parametrize("input,expected", [(0, True), (-1, False)])
def test_validate(input, expected):
    assert validate(input) == expected
```