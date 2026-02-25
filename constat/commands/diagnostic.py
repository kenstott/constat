# Copyright (c) 2025 Kenneth Stott
#
# Diagnostic commands - expose LLM tool outputs for inspection.

"""Diagnostic commands for inspecting LLM-facing tool outputs."""

from __future__ import annotations

from constat.commands.base import (
    CommandContext,
    CommandResult,
    ErrorResult,
    JsonResult,
)


def _build_schema_tools(ctx: CommandContext):
    """Build SchemaDiscoveryTools from session context."""
    from constat.discovery.schema_tools import SchemaDiscoveryTools

    session = ctx.session
    if not hasattr(session, 'schema_manager') or not session.schema_manager:
        return None

    return SchemaDiscoveryTools(
        schema_manager=session.schema_manager,
        doc_tools=getattr(session, 'doc_tools', None),
        api_tools=None,
        session_id=getattr(session, 'session_id', None),
        user_id=getattr(session, 'user_id', None),
        api_schema_manager=getattr(session, 'api_schema_manager', None),
    )


def schema_command(ctx: CommandContext) -> CommandResult:
    """Show detailed table schema as the LLM sees it."""
    args = ctx.args.strip()
    if not args or '.' not in args:
        return ErrorResult(error="Usage: /schema <db.table>\nExample: /schema hr.employees")

    db, table = args.split('.', 1)
    tools = _build_schema_tools(ctx)
    if not tools:
        return ErrorResult(error="No schema manager available.")

    result = tools.get_table_schema(db, table)
    return JsonResult(title=f"Schema: {args}", data=result)


def search_tables_command(ctx: CommandContext) -> CommandResult:
    """Semantic search for relevant tables."""
    query = ctx.args.strip()
    if not query:
        return ErrorResult(error="Usage: /search-tables <query>\nExample: /search-tables employee compensation")

    tools = _build_schema_tools(ctx)
    if not tools:
        return ErrorResult(error="No schema manager available.")

    result = tools.search_tables(query)
    return JsonResult(
        title=f"Table search: {query}",
        data=result,
        footer=f"{len(result)} table(s) found",
    )


def search_apis_command(ctx: CommandContext) -> CommandResult:
    """Semantic search for relevant APIs."""
    query = ctx.args.strip()
    if not query:
        return ErrorResult(error="Usage: /search-apis <query>\nExample: /search-apis cat")

    api_schema_manager = getattr(ctx.session, 'api_schema_manager', None)
    if not api_schema_manager:
        return ErrorResult(error="No API schema manager available.")

    result = api_schema_manager.find_relevant_apis(query)
    return JsonResult(
        title=f"API search: {query}",
        data=result,
        footer=f"{len(result)} API(s) found",
    )


def search_docs_command(ctx: CommandContext) -> CommandResult:
    """Semantic search for relevant documents."""
    query = ctx.args.strip()
    if not query:
        return ErrorResult(error="Usage: /search-docs <query>\nExample: /search-docs business rules")

    doc_tools = getattr(ctx.session, 'doc_tools', None)
    if not doc_tools:
        return ErrorResult(error="No document tools available.")

    session_id = getattr(ctx.session, 'session_id', None)
    result = doc_tools.search_documents(query, session_id=session_id)
    return JsonResult(
        title=f"Document search: {query}",
        data=result,
        footer=f"{len(result)} document(s) found",
    )


def lookup_command(ctx: CommandContext) -> CommandResult:
    """Look up a glossary term with full details."""
    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /lookup <name>\nExample: /lookup bronze")

    tools = _build_schema_tools(ctx)
    if not tools:
        return ErrorResult(error="No schema manager available.")

    result = tools.lookup_glossary_term(name)
    return JsonResult(title=f"Glossary: {name}", data=result)


def entity_command(ctx: CommandContext) -> CommandResult:
    """Find entity across schema and documents."""
    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /entity <name>\nExample: /entity customer")

    tools = _build_schema_tools(ctx)
    if not tools:
        return ErrorResult(error="No schema manager available.")

    result = tools.find_entity(name)
    return JsonResult(title=f"Entity: {name}", data=result)


def known_facts_command(ctx: CommandContext) -> CommandResult:
    """List all known/cached facts as the LLM sees them."""
    from constat.discovery.fact_tools import FactResolutionTools

    fact_resolver = getattr(ctx.session, 'fact_resolver', None)
    if not fact_resolver:
        return ErrorResult(error="No fact resolver available.")

    doc_tools = getattr(ctx.session, 'doc_tools', None)
    tools = FactResolutionTools(fact_resolver=fact_resolver, doc_tools=doc_tools)
    result = tools.list_known_facts()
    return JsonResult(
        title="Known Facts",
        data=result,
        footer=f"{result.get('count', 0)} fact(s)",
    )


def sources_command(ctx: CommandContext) -> CommandResult:
    """Find relevant sources for a query."""
    query = ctx.args.strip()
    if not query:
        return ErrorResult(error="Usage: /sources <query>\nExample: /sources employee salary")

    session = ctx.session
    if not hasattr(session, 'find_relevant_sources'):
        return ErrorResult(error="Session does not support source search.")

    result = session.find_relevant_sources(query)
    total = sum(len(v) for v in result.values() if isinstance(v, list))
    return JsonResult(
        title=f"Sources: {query}",
        data=result,
        footer=f"{total} source(s) found",
    )