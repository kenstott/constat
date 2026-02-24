# Copyright (c) 2025 Kenneth Stott
#
# Data source commands - databases, apis, documents, files.

"""Data source commands."""

from __future__ import annotations

from constat.commands.base import (
    CommandContext,
    TableResult,
    ListResult,
    ErrorResult,
)


def databases_command(ctx: CommandContext) -> TableResult:
    """List configured databases."""
    config = ctx.session.config

    if not config or not config.databases:
        return TableResult(
            success=True,
            title="Databases",
            columns=["Name", "Type", "Description"],
            rows=[],
            footer="No databases configured.",
        )

    rows = []
    for name, db_config in config.databases.items():
        # Determine type from URI or config
        db_type = "unknown"
        if hasattr(db_config, "type") and db_config.type:
            db_type = db_config.type
        elif hasattr(db_config, "uri") and db_config.uri:
            uri = db_config.uri
            if "sqlite" in uri:
                db_type = "sqlite"
            elif "postgres" in uri:
                db_type = "postgres"
            elif "mysql" in uri:
                db_type = "mysql"

        desc = getattr(db_config, "description", "")[:50] if hasattr(db_config, "description") else ""
        rows.append([name, db_type, desc])

    return TableResult(
        success=True,
        title="Databases",
        columns=["Name", "Type", "Description"],
        rows=rows,
        footer=f"{len(rows)} database(s) configured",
    )


def apis_command(ctx: CommandContext) -> TableResult:
    """List configured APIs."""
    config = ctx.session.config

    if not config or not config.apis:
        return TableResult(
            success=True,
            title="APIs",
            columns=["Name", "Type", "URL"],
            rows=[],
            footer="No APIs configured.",
        )

    rows = []
    for name, api_config in config.apis.items():
        api_type = getattr(api_config, "type", "rest")
        url = getattr(api_config, "url", "") or getattr(api_config, "base_url", "")
        rows.append([name, api_type, url[:50]])

    return TableResult(
        success=True,
        title="APIs",
        columns=["Name", "Type", "URL"],
        rows=rows,
        footer=f"{len(rows)} API(s) configured",
    )


def documents_command(ctx: CommandContext) -> ListResult:
    """List all documents."""
    config = ctx.session.config

    if not config or not config.documents:
        return ListResult(
            success=True,
            title="Documents",
            items=[],
            empty_message="No documents configured.",
        )

    items = []
    for name, doc_config in config.documents.items():
        items.append({
            "name": name,
            "type": getattr(doc_config, "type", "file"),
            "path": getattr(doc_config, "path", ""),
            "description": getattr(doc_config, "description", ""),
        })

    return ListResult(
        success=True,
        title="Documents",
        items=items,
    )


def files_command(ctx: CommandContext) -> ListResult:
    """List all data files (CSV, JSON, Parquet, etc.)."""
    config = ctx.session.config

    items = []

    # Check databases for file-based sources
    if config and config.databases:
        for name, db_config in config.databases.items():
            db_type = getattr(db_config, "type", None)
            if db_type in ("csv", "json", "parquet", "arrow"):
                items.append({
                    "name": name,
                    "type": db_type,
                    "path": getattr(db_config, "path", ""),
                    "description": getattr(db_config, "description", ""),
                })

    if not items:
        return ListResult(
            success=True,
            title="Data Files",
            items=[],
            empty_message="No data files configured.",
        )

    return ListResult(
        success=True,
        title="Data Files",
        items=items,
    )


def discover_command(ctx: CommandContext) -> "JsonResult | ErrorResult":
    """Search all data sources and return structured JSON results.

    Usage:
        /discover <query>  - Search all sources (tables, APIs, documents, glossary)
    """
    from constat.commands.base import JsonResult
    from constat.discovery.schema_tools import SchemaDiscoveryTools

    args = ctx.args.strip()
    if not args:
        return ErrorResult(error="Usage: /discover <query>\nExample: /discover employee compensation")

    session = ctx.session
    if not hasattr(session, 'schema_manager') or not session.schema_manager:
        return ErrorResult(error="No schema manager available.")

    session_id = getattr(session, 'session_id', None)
    user_id = getattr(session, 'user_id', None)

    tools = SchemaDiscoveryTools(
        schema_manager=session.schema_manager,
        doc_tools=getattr(session, 'doc_tools', None),
        api_tools=None,
        session_id=session_id,
        user_id=user_id,
        api_schema_manager=getattr(session, 'api_schema_manager', None),
    )

    result = tools.search_all(args, limit=15)

    return JsonResult(
        title=f"Discovery: {args}",
        data=result,
        footer=f"tables={len(result.get('tables', []))} apis={len(result.get('apis', []))} documents={len(result.get('documents', []))} glossary={len(result.get('glossary', []))}",
    )
