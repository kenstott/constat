---
name: venv-setup
description: Python virtual environment and tooling setup. Auto-triggers when setting up development environment.
---

# Environment Setup

## Python Version
Python 3.12 via `.venv/`

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Build Backend
- `pyproject.toml` with `hatchling`
- No `setup.py`, no `setup.cfg`

## Formatting / Linting
- **ruff** — linter (rules: E, F, I, B, UP, ANN, S, A, C4, T20, PT, PTH, SIM, ARG)
- **black** — formatter
- Line length: 100
- Target: Python 3.11+

## Verification
```bash
python -m pytest tests/unit/ -x -q           # unit tests (fast, no services)
python -m pytest tests/integration/ -x -q    # integration tests (requires Docker)
python -m pytest tests/e2e/ -x -q            # e2e tests (requires backend + Vite)
python -m pytest tests/ -x -q               # all backend tests
cd constat-ui && npm run build               # frontend build
cd constat-ui && npm run lint                # frontend lint
cd constat-ui && npx tsc --noEmit            # type check
```

## Server
```bash
python -m constat.server -c demo/config.yaml
```
Never run server or CLI without `-c demo/config.yaml`.