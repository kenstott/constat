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
    from chonk.cluster import ClusterMap
    from chonk.community import CommunityIndex
    from chonk.storage import Store

_lock = threading.Lock()
_global_store_path: str | None = None
_readonly_store: Store | None = None
_community_index: CommunityIndex | None = None
_cluster_map: ClusterMap | None = None
_domain_dag: dict[str, list[str]] = {}  # path -> [child_paths]
_domain_descriptions: dict[str, str] = {}  # path -> description


def set_global_store_path(path: str | Path) -> None:
    global _global_store_path, _readonly_store
    with _lock:
        _global_store_path = str(path)
        _readonly_store = None  # reset so next call re-opens


def get_global_store_path() -> str | None:
    with _lock:
        return _global_store_path


def get_global_store_readonly() -> Store | None:
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


def set_community_index(idx: CommunityIndex) -> None:
    global _community_index
    with _lock:
        _community_index = idx


def get_community_index() -> CommunityIndex | None:
    with _lock:
        return _community_index


def set_cluster_map(cm: ClusterMap) -> None:
    global _cluster_map
    with _lock:
        _cluster_map = cm


def get_cluster_map() -> ClusterMap | None:
    with _lock:
        return _cluster_map


def set_domain_descriptions(descriptions: dict[str, str]) -> None:
    global _domain_descriptions
    with _lock:
        _domain_descriptions = descriptions


def get_domain_descriptions() -> dict[str, str]:
    with _lock:
        return dict(_domain_descriptions)


def set_domain_dag(dag: dict[str, list[str]]) -> None:
    global _domain_dag
    with _lock:
        _domain_dag = dag


def get_domain_dag() -> dict[str, list[str]]:
    with _lock:
        return dict(_domain_dag)
