---
name: python-style
description: Python coding conventions for this project. Auto-triggers when writing or reviewing Python code.
---

# Python Style Conventions

## File Header
Every `.py` file starts with:
```python
# Copyright (c) Kenneth Stott. All rights reserved.
# Licensed under the Business Source License 1.1.
from __future__ import annotations
```

## Import Order
1. `from __future__ import annotations`
2. stdlib (blank line)
3. third-party (blank line)
4. local (blank line)

## Type Hints
- Modern style: `list[str]`, `dict[str, int]`, `X | None` (not `List`, `Optional`)
- Return types required on all functions
- Avoid `Any` — use specific types or generics

## Naming
- `PascalCase` — classes
- `snake_case` — functions, variables, modules
- `_leading_underscore` — private
- `UPPER_SNAKE` — constants

## Formatting
- Line length: 100 (ruff + black)
- Ruff rules: E, F, I, B, UP, ANN, S, A, C4, T20, PT, PTH, SIM, ARG
- Target: Python 3.11+

## Docstrings
Google format:
```python
def func(x: int, y: str) -> bool:
    """Short summary.

    Args:
        x: Description.
        y: Description.

    Returns:
        Description.

    Raises:
        ValueError: When x < 0.
    """
```

## Logging
- No `print()` in production code
- Use `logger = logging.getLogger(__name__)` at module scope
- Debug context tags: `[PARALLEL]`, `[COMPLEXITY]`, `[DYNAMIC_CONTEXT]`

## General
- Dataclasses/Pydantic for data structures (not plain dicts)
- Context managers for resource cleanup
- `pathlib.Path` over `os.path`
- No mutable default arguments
- No bare `except:` — always specify exception type