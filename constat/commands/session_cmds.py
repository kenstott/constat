# Copyright (c) 2025 Kenneth Stott
#
# Session management commands - state, reset, facts, context, preferences.

"""Session management commands."""

from __future__ import annotations

from typing import Any

from constat.commands.base import (
    CommandContext,
    CommandResult,
    TableResult,
    ListResult,
    TextResult,
    KeyValueResult,
    ErrorResult,
)


def state_command(ctx: CommandContext) -> KeyValueResult:
    """Show current session state."""
    session = ctx.session

    # Basic session info
    pairs: dict[str, Any] = {
        "Session ID": session.session_id or "Not set",
        "User": session.user_id,
    }

    # Conversation state if available
    if hasattr(session, "_conversation_state"):
        state = session._conversation_state
        pairs["Mode"] = state.mode.value.upper() if hasattr(state, "mode") else "Unknown"
        pairs["Phase"] = state.phase.value if hasattr(state, "phase") else "Unknown"

    # Plan info
    if session.plan:
        pairs["Plan"] = f"{len(session.plan.steps)} steps"
        pairs["Plan Status"] = "Active"
    else:
        pairs["Plan"] = "None"

    # Datastore info
    if session.datastore:
        tables = session.datastore.list_tables()
        pairs["Tables"] = len(tables)
        artifacts = session.datastore.list_artifacts()
        pairs["Artifacts"] = len(artifacts)
    else:
        pairs["Datastore"] = "Not initialized"

    # Facts
    if hasattr(session, "fact_resolver"):
        facts = session.fact_resolver.get_all_facts()
        pairs["Facts"] = len(facts)

    return KeyValueResult(
        success=True,
        title="Session State",
        pairs=pairs,
    )


def reset_command(ctx: CommandContext) -> TextResult:
    """Reset session state."""
    session = ctx.session

    # Clear plan
    session.plan = None

    # Clear datastore
    if session.datastore:
        session.datastore.clear()

    # Clear facts
    if hasattr(session, "fact_resolver"):
        session.fact_resolver.clear()

    # Clear scratchpad
    if hasattr(session, "scratchpad"):
        session.scratchpad.clear()

    return TextResult(
        success=True,
        content="Session reset. All tables, artifacts, and facts cleared.",
    )


def facts_command(ctx: CommandContext) -> CommandResult:
    """Show cached facts from this session."""
    session = ctx.session

    if not hasattr(session, "fact_resolver"):
        return ListResult(
            success=True,
            title="Facts",
            items=[],
            empty_message="Fact resolver not available.",
        )

    facts = session.fact_resolver.get_all_facts()

    if not facts:
        return ListResult(
            success=True,
            title="Facts",
            items=[],
            empty_message="No facts cached yet.",
        )

    items = []
    for name, fact in facts.items():
        source = fact.source.value if hasattr(fact.source, "value") else str(fact.source)
        items.append({
            "name": name,
            "value": fact.value,
            "source": source,
            "reasoning": getattr(fact, "reasoning", None),
            "confidence": getattr(fact, "confidence", None),
        })

    return ListResult(
        success=True,
        title="Cached Facts",
        items=items,
    )


def context_command(ctx: CommandContext) -> KeyValueResult:
    """Show context size and token usage."""
    session = ctx.session

    pairs: dict[str, Any] = {}

    # Scratchpad stats
    if hasattr(session, "scratchpad") and session.scratchpad:
        sp = session.scratchpad
        pairs["Scratchpad Entries"] = len(sp._entries) if hasattr(sp, "_entries") else "N/A"

    # Context estimator
    if hasattr(session, "context_estimator"):
        try:
            stats = session.context_estimator.estimate()
            pairs["Estimated Tokens"] = stats.total_tokens
            pairs["Schema Tokens"] = stats.schema_tokens
            pairs["History Tokens"] = stats.history_tokens
            pairs["Scratchpad Tokens"] = stats.scratchpad_tokens
        except Exception:
            pairs["Context Estimation"] = "Error"

    # Plan context
    if session.plan:
        pairs["Plan Steps"] = len(session.plan.steps)

    if not pairs:
        pairs["Status"] = "Context tracking not available"

    return KeyValueResult(
        success=True,
        title="Context Usage",
        pairs=pairs,
    )


def preferences_command(ctx: CommandContext) -> KeyValueResult:
    """Show current preferences."""
    session = ctx.session
    config = session.session_config if hasattr(session, "session_config") else None

    pairs: dict[str, Any] = {}

    if config:
        pairs["Verbose"] = "on" if getattr(config, "verbose", False) else "off"
        pairs["Raw Output"] = "on" if getattr(config, "show_raw_output", True) else "off"
        pairs["Insights"] = "on" if getattr(config, "enable_insights", True) else "off"
        pairs["Auto-Approve"] = "on" if getattr(config, "auto_approve_plans", False) else "off"
    else:
        pairs["Status"] = "Session config not available"

    return KeyValueResult(
        success=True,
        title="Preferences",
        pairs=pairs,
    )
