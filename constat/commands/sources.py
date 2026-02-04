# Copyright (c) 2025 Kenneth Stott
#
# Data source commands - databases, apis, documents, files.

"""Data source commands."""

from __future__ import annotations

from typing import Any

from constat.commands.base import (
    CommandContext,
    CommandResult,
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


def discover_command(ctx: CommandContext) -> TableResult:
    """Unified semantic search across all data sources (databases, APIs, documents).

    Usage:
        /discover <query>           - Search all sources
        /discover database <query>  - Search database tables/columns only
        /discover api <query>       - Search API endpoints only
        /discover document <query>  - Search documents only
    """
    import numpy as np
    from constat.embedding_loader import EmbeddingModelLoader

    args = ctx.args.strip()
    if not args:
        return TableResult(
            success=True,
            title="Usage",
            columns=["Command", "Description"],
            rows=[
                ["/discover <query>", "Search all sources"],
                ["/discover database <query>", "Search database tables/columns"],
                ["/discover api <query>", "Search API endpoints"],
                ["/discover document <query>", "Search documents"],
            ],
            footer="Example: /discover performance review",
        )

    # Parse scope filter
    parts = args.split()
    source_filter = None
    scope_map = {
        "database": "schema", "db": "schema", "databases": "schema",
        "table": "schema", "tables": "schema",
        "api": "api", "apis": "api",
        "document": "document", "documents": "document", "doc": "document", "docs": "document",
    }

    if parts and parts[0].lower() in scope_map:
        source_filter = scope_map[parts[0].lower()]
        query = " ".join(parts[1:])
    else:
        query = args

    if not query:
        return ErrorResult(error="Please provide a search query.")

    # Get vector store
    vector_store = None
    session = ctx.session
    if hasattr(session, 'schema_manager') and session.schema_manager:
        vector_store = getattr(session.schema_manager, '_vector_store', None)
    if not vector_store and hasattr(session, 'doc_tools') and session.doc_tools:
        vector_store = getattr(session.doc_tools, '_vector_store', None)

    if not vector_store:
        return ErrorResult(error="No vector store available.")

    # Embed query and search
    model = EmbeddingModelLoader.get_instance().get_model()
    query_embedding = model.encode(query, convert_to_numpy=True)
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)

    # Get active project IDs for filtering
    project_ids = None
    if hasattr(session, 'doc_tools') and session.doc_tools:
        project_ids = getattr(session.doc_tools, '_active_project_ids', None)

    # Get session_id for entity lookup
    session_id = getattr(ctx, 'session_id', None) or getattr(session, 'session_id', None)

    # Search chunks - base + active projects, with entities
    enriched_results = vector_store.search_enriched(
        query_embedding=query_embedding,
        limit=15,
        project_ids=project_ids,
        session_id=session_id,
    )

    # Filter by source type
    if source_filter:
        enriched_results = [
            r for r in enriched_results
            if r.chunk.source == source_filter
        ]

    # Filter by minimum score
    enriched_results = [r for r in enriched_results if r.score >= 0.3]

    if not enriched_results:
        scope_str = f" in {source_filter}" if source_filter else ""
        return TableResult(
            success=True,
            title="Discovery Results",
            columns=["Score", "Source", "Name", "Entities", "Content"],
            rows=[],
            footer=f"No results found for '{query}'{scope_str}.",
        )

    # Build result rows
    rows = []
    for r in enriched_results:
        score = f"{r.score:.2f}"
        source_type = r.chunk.source

        # Format source type for display
        source_display = {
            "schema": "DATABASE",
            "api": "API",
            "document": "DOCUMENT",
        }.get(source_type, "UNKNOWN")

        # Use document_name as the name
        name = r.chunk.document_name

        # Get entity names (display names)
        entity_names = [e.display_name or e.name for e in r.entities[:5]]  # Limit to 5
        entities_str = ", ".join(entity_names) if entity_names else "-"
        if len(r.entities) > 5:
            entities_str += f" (+{len(r.entities) - 5})"

        # Truncate content
        content = r.chunk.content[:50].replace("\n", " ")
        if len(r.chunk.content) > 50:
            content += "..."

        rows.append([score, source_display, name, entities_str, content])

    scope_str = f" ({source_filter})" if source_filter else ""
    return TableResult(
        success=True,
        title=f"Discovery Results{scope_str}",
        columns=["Score", "Source", "Name", "Entities", "Content"],
        rows=rows,
        footer=f"Found {len(rows)} matches for '{query}'",
    )
