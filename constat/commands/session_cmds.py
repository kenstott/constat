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

    # Active role
    if hasattr(session, "role_manager"):
        role_name = session.role_manager.active_role_name
        pairs["Role"] = role_name or "None"

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


# ============================================================================
# Learnings & Rules Commands
# ============================================================================


def learnings_command(ctx: CommandContext) -> CommandResult:
    """Show learnings and rules."""
    from constat.storage.learnings import LearningStore

    try:
        store = LearningStore(user_id=ctx.session.user_id)
        learnings = store.list_raw_learnings(limit=20)
        rules = store.list_rules(limit=20)
        stats = store.get_stats()

        sections = []

        # Rules section
        if rules:
            rule_pairs = {}
            for r in rules:
                tags_str = f" [{', '.join(r.get('tags', []))}]" if r.get('tags') else ""
                conf_str = f" ({r.get('confidence', 0):.0%})"
                rule_pairs[r['id']] = f"{r['summary']}{conf_str}{tags_str}"
            sections.append(("Rules", rule_pairs))

        # Learnings section
        if learnings:
            learn_pairs = {}
            for l in learnings:
                cat_str = f" [{l.get('category', '')}]"
                learn_pairs[l['id']] = f"{l.get('correction', '')}{cat_str}"
            sections.append(("Pending Learnings", learn_pairs))

        # Stats
        stat_pairs = {
            "Total Rules": stats.get("total_rules", 0),
            "Pending Learnings": stats.get("unpromoted", 0),
            "Archived": stats.get("total_archived", 0),
        }
        sections.append(("Stats", stat_pairs))

        if not rules and not learnings:
            return TextResult(
                success=True,
                content="No learnings or rules yet. Use `/correct <text>` to add corrections or `/rule <text>` to add rules directly.",
            )

        return KeyValueResult(
            success=True,
            title="Learnings & Rules",
            sections=sections,
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to load learnings: {e}")


def rule_command(ctx: CommandContext) -> CommandResult:
    """Add a new rule directly."""
    if not ctx.args.strip():
        return ErrorResult(error="Usage: /rule <rule text>")

    from constat.storage.learnings import LearningStore, LearningCategory

    try:
        store = LearningStore(user_id=ctx.session.user_id)
        rule_id = store.save_rule(
            summary=ctx.args.strip(),
            category=LearningCategory.USER_CORRECTION,
            confidence=0.9,
            source_learnings=[],
            tags=[],
        )
        return TextResult(
            success=True,
            content=f"Rule added: `{rule_id}`\n\n> {ctx.args.strip()}",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to add rule: {e}")


def rule_edit_command(ctx: CommandContext) -> CommandResult:
    """Edit an existing rule."""
    parts = ctx.args.strip().split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /rule-edit <rule_id> <new text>")

    rule_id, new_text = parts[0], parts[1]

    from constat.storage.learnings import LearningStore

    try:
        store = LearningStore(user_id=ctx.session.user_id)
        success = store.update_rule(rule_id=rule_id, summary=new_text)
        if success:
            return TextResult(
                success=True,
                content=f"Rule `{rule_id}` updated.\n\n> {new_text}",
            )
        return ErrorResult(error=f"Rule not found: {rule_id}")
    except Exception as e:
        return ErrorResult(error=f"Failed to update rule: {e}")


def rule_delete_command(ctx: CommandContext) -> CommandResult:
    """Delete a rule."""
    rule_id = ctx.args.strip()
    if not rule_id:
        return ErrorResult(error="Usage: /rule-delete <rule_id>")

    from constat.storage.learnings import LearningStore

    try:
        store = LearningStore(user_id=ctx.session.user_id)
        success = store.delete_rule(rule_id)
        if success:
            return TextResult(
                success=True,
                content=f"Rule `{rule_id}` deleted.",
            )
        return ErrorResult(error=f"Rule not found: {rule_id}")
    except Exception as e:
        return ErrorResult(error=f"Failed to delete rule: {e}")


def correct_command(ctx: CommandContext) -> CommandResult:
    """Record a correction for future reference."""
    if not ctx.args.strip():
        return ErrorResult(error="Usage: /correct <correction text>")

    from constat.storage.learnings import LearningStore, LearningCategory, LearningSource

    try:
        store = LearningStore(user_id=ctx.session.user_id)
        learning_id = store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={},
            correction=ctx.args.strip(),
            source=LearningSource.EXPLICIT_COMMAND,
        )
        return TextResult(
            success=True,
            content=f"Correction recorded: `{learning_id}`\n\n> {ctx.args.strip()}\n\nUse `/compact-learnings` to promote similar corrections to rules.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to record correction: {e}")


def role_command(ctx: CommandContext) -> CommandResult:
    """Set or show the current role."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")

    role_manager = session.role_manager
    args = ctx.args.strip()

    if not args:
        # Show current role
        if role_manager.active_role_name:
            role = role_manager.active_role
            return TextResult(
                success=True,
                content=f"**Current role:** {role_manager.active_role_name}\n\n> {role.prompt[:200]}{'...' if len(role.prompt) > 200 else ''}",
            )
        else:
            return TextResult(
                success=True,
                content="No role active. Use `/role <name>` to set one, or `/roles` to list available roles.",
            )

    # Set role
    if role_manager.set_active_role(args):
        if args.lower() == "none":
            return TextResult(success=True, content="Role cleared.")
        role = role_manager.active_role
        return TextResult(
            success=True,
            content=f"Role set to **{args}**\n\n> {role.prompt[:200]}{'...' if len(role.prompt) > 200 else ''}",
        )
    else:
        available = ", ".join(role_manager.list_roles()) or "none"
        return ErrorResult(
            error=f"Role not found: {args}",
            details=f"Available roles: {available}",
        )


def roles_command(ctx: CommandContext) -> CommandResult:
    """List available roles."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")

    role_manager = session.role_manager

    if not role_manager.has_roles:
        return TextResult(
            success=True,
            content=f"No roles defined.\n\nCreate roles in: `{role_manager.roles_file_path}`\n\nExample:\n```yaml\ncfo:\n  prompt: |\n    Focus on financial impact and executive summaries.\n```",
        )

    items = []
    for name in role_manager.list_roles():
        role = role_manager.get_role(name)
        is_active = name == role_manager.active_role_name
        items.append({
            "name": f"{'→ ' if is_active else ''}{name}",
            "prompt": role.prompt[:60] + "..." if len(role.prompt) > 60 else role.prompt,
        })

    return ListResult(
        success=True,
        title="Available Roles",
        items=items,
        empty_message="No roles defined.",
    )
