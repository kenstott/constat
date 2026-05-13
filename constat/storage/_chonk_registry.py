# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Process-wide registry for the chonk global Store.

Warmup sets the store path. The read-only Store is opened lazily and reused
across searches — chonk's ThreadLocalDuckDB gives each thread its own connection.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chonk.storage import Store

_lock = threading.Lock()
_global_store_path: str | None = None
_readonly_store: "Store | None" = None


def set_global_store_path(path: str | Path) -> None:
    global _global_store_path, _readonly_store
    with _lock:
        _global_store_path = str(path)
        _readonly_store = None  # reset so next call re-opens


def get_global_store_path() -> str | None:
    with _lock:
        return _global_store_path


def get_global_store_readonly() -> "Store | None":
    """Return (or lazily open) the singleton read-only global chonk Store."""
    global _readonly_store
    with _lock:
        if _readonly_store is not None:
            return _readonly_store
        if not _global_store_path or not Path(_global_store_path).exists():
            return None
        from chonk.storage import Store
        _readonly_store = Store(_global_store_path, read_only=True)
        return _readonly_store
