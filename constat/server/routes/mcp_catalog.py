# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""MCP catalog REST endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from constat.mcp.catalog import McpCatalog

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/catalog")
async def list_catalog(
    category: str | None = None,
    q: str | None = None,
) -> list[dict[str, Any]]:
    """List MCP servers from the catalog with optional filters."""
    catalog = McpCatalog()
    return catalog.list_servers(category=category, query=q)


@router.get("/catalog/{slug}")
async def get_catalog_server(slug: str) -> dict[str, Any]:
    """Get a specific MCP server by slug."""
    catalog = McpCatalog()
    server = catalog.get_server(slug)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server not found: {slug}")
    return server
