# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Config hashing utilities for warmup invalidation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _compute_config_hash(data: dict) -> str:
    config_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def compute_db_config_hash(databases: dict) -> str:
    db_data = {}
    if databases:
        for db_name, db_config in sorted(databases.items()):
            db_data[db_name] = {
                "type": db_config.type or "",
                "uri": db_config.uri or "",
                "database": db_config.database or "",
                "path": db_config.path or "",
                "jdbc_url": getattr(db_config, "jdbc_url", None) or "",
                "jdbc_driver": getattr(db_config, "jdbc_driver", None) or "",
            }
    return _compute_config_hash(db_data)


def compute_api_config_hash(apis: dict) -> str:
    api_data = {}
    if apis:
        for api_name, api_config in sorted(apis.items()):
            api_data[api_name] = {
                "type": api_config.type or "",
                "url": api_config.url or "",
                "spec_url": api_config.spec_url or "",
                "spec_path": api_config.spec_path or "",
            }
    return _compute_config_hash(api_data)


def compute_doc_config_hash(documents: dict) -> str:
    doc_data = {}
    if documents:
        for doc_name, doc_config in sorted(documents.items()):
            doc_data[doc_name] = {
                "path": doc_config.path or "",
                "description": doc_config.description or "",
                "type": doc_config.type or "",
            }
    return _compute_config_hash(doc_data)


def compute_db_resource_hash(db_name: str, db_config) -> str:
    data = {
        "name": db_name,
        "type": db_config.type or "",
        "uri": db_config.uri or "",
        "database": db_config.database or "",
        "path": db_config.path or "",
    }
    return _compute_config_hash(data)


def compute_api_resource_hash(api_name: str, api_config) -> str:
    data = {
        "name": api_name,
        "type": api_config.type or "",
        "url": api_config.url or "",
        "spec_url": api_config.spec_url or "",
        "spec_path": api_config.spec_path or "",
    }
    return _compute_config_hash(data)


def compute_er_config_hash(entity_resolution: list, apis: dict | None = None) -> str:
    data = {}
    for i, cfg in enumerate(entity_resolution):
        data[f"er_{i}"] = {
            "entity_type": cfg.entity_type, "source": cfg.source,
            "table": cfg.table, "query": cfg.query, "endpoint": cfg.endpoint,
            "items_path": cfg.items_path, "name_field": cfg.name_field,
            "values": sorted(cfg.values) if cfg.values else None,
            "max_values": cfg.max_values,
        }
    if apis:
        for name, api_cfg in sorted(apis.items()):
            data[f"api_{name}"] = {"type": getattr(api_cfg, 'type', ''), "url": getattr(api_cfg, 'url', '')}
    return _compute_config_hash(data)


def compute_doc_resource_hash(doc_name: str, doc_config, config_dir: str | None) -> str:
    data = {
        "name": doc_name,
        "path": doc_config.path or "",
        "url": doc_config.url or "",
        "description": doc_config.description or "",
        "type": doc_config.type or "",
    }
    if doc_config.path:
        doc_path = Path(doc_config.path)
        if not doc_path.is_absolute() and config_dir:
            doc_path = (Path(config_dir) / doc_config.path).resolve()
        if doc_path.exists():
            try:
                stat = os.stat(doc_path)
                data["mtime"] = stat.st_mtime
                data["size"] = stat.st_size
            except OSError:
                pass
    return _compute_config_hash(data)
