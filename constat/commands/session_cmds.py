# Copyright (c) 2025 Kenneth Stott
#
# Session management commands - state, reset, facts, context, preferences.

"""Session management commands."""

from __future__ import annotations

from typing import Any

from constat.commands.base import (
    CommandContext,
    CommandResult,
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
            content=f"No roles defined.\n\nCreate roles with `/role-create <name>` or in: `{role_manager.roles_file_path}`",
        )

    items = []
    for name in role_manager.list_roles():
        role = role_manager.get_role(name)
        is_active = name == role_manager.active_role_name
        items.append({
            "name": f"{'→ ' if is_active else ''}{name}",
            "description": role.description or role.prompt[:60] + "...",
        })

    return ListResult(
        success=True,
        title="Available Roles",
        items=items,
        empty_message="No roles defined.",
    )


def role_create_command(ctx: CommandContext) -> CommandResult:
    """Create a new role."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /role-create <name> [description]",
            details="Creates a role. You'll be prompted to enter the prompt content.",
        )

    # Parse name and optional description
    parts = args.split(maxsplit=1)
    name = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    role_manager = session.role_manager

    # Check if exists
    if role_manager.get_role(name):
        return ErrorResult(error=f"Role '{name}' already exists. Use /role-edit to modify it.")

    # Create with placeholder prompt (user should edit)
    try:
        role_manager.create_role(
            name=name,
            prompt="[Enter your role prompt here - describe the persona, communication style, and priorities]",
            description=description,
        )
        return TextResult(
            success=True,
            content=f"Role **{name}** created.\n\nEdit the prompt with: `/role-edit {name} <prompt>`\n\nOr use `/role-draft {name} <description>` to generate a prompt with AI.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to create role: {e}")


def role_edit_command(ctx: CommandContext) -> CommandResult:
    """Edit an existing role."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(error="Usage: /role-edit <name> <prompt>")

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /role-edit <name> <prompt>")

    name, prompt = parts[0], parts[1]
    role_manager = session.role_manager

    # Check if exists
    if not role_manager.get_role(name):
        return ErrorResult(error=f"Role '{name}' not found. Use /roles to list available roles.")

    try:
        role_manager.update_role(name=name, prompt=prompt)
        return TextResult(
            success=True,
            content=f"Role **{name}** updated.\n\n> {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to update role: {e}")


def role_delete_command(ctx: CommandContext) -> CommandResult:
    """Delete a role."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")

    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /role-delete <name>")

    role_manager = session.role_manager

    if role_manager.delete_role(name):
        return TextResult(success=True, content=f"Role **{name}** deleted.")
    else:
        return ErrorResult(error=f"Role '{name}' not found.")


def role_draft_command(ctx: CommandContext) -> CommandResult:
    """Draft a role using AI based on description."""
    session = ctx.session

    if not hasattr(session, "role_manager"):
        return ErrorResult(error="Role manager not available.")
    if not hasattr(session, "llm"):
        return ErrorResult(error="LLM not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /role-draft <name> <description>",
            details="Example: /role-draft cfo Focus on financial impact and executive summaries",
        )

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /role-draft <name> <description>")

    name, description = parts[0], parts[1]
    role_manager = session.role_manager

    # Check if exists
    if role_manager.get_role(name):
        return ErrorResult(error=f"Role '{name}' already exists. Delete it first with /role-delete.")

    try:
        role = role_manager.draft_role(name, description, session.llm)
        # Save the drafted role
        role_manager.create_role(name=role.name, prompt=role.prompt, description=role.description)
        return TextResult(
            success=True,
            content=f"Role **{name}** drafted and saved.\n\n**Description:** {role.description}\n\n**Prompt:**\n> {role.prompt[:300]}{'...' if len(role.prompt) > 300 else ''}\n\nUse `/role {name}` to activate.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to draft role: {e}")


# ============================================================================
# Skills Commands
# ============================================================================


def skill_command(ctx: CommandContext) -> CommandResult:
    """Show or activate a skill."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    skill_manager = session.skill_manager
    args = ctx.args.strip()

    if not args:
        # Show active skills
        if skill_manager.active_skills:
            items = []
            for name in skill_manager.active_skills:
                skill = skill_manager.get_skill(name)
                if skill:
                    items.append({
                        "name": name,
                        "description": skill.description or skill.prompt[:60] + "...",
                    })
            return ListResult(
                success=True,
                title="Active Skills",
                items=items,
            )
        else:
            return TextResult(
                success=True,
                content="No skills active. Use `/skill <name>` to activate, or `/skills` to list available.",
            )

    # Activate skill
    if skill_manager.activate_skill(args):
        skill = skill_manager.get_skill(args)
        return TextResult(
            success=True,
            content=f"Skill **{args}** activated.\n\n> {skill.description or skill.prompt[:100]}{'...' if len(skill.prompt) > 100 else ''}",
        )
    else:
        available = ", ".join(skill_manager.list_skills()) or "none"
        return ErrorResult(
            error=f"Skill not found: {args}",
            details=f"Available skills: {available}",
        )


def skills_command(ctx: CommandContext) -> CommandResult:
    """List available skills."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    skill_manager = session.skill_manager

    if not skill_manager.has_skills:
        return TextResult(
            success=True,
            content=f"No skills defined.\n\nCreate skills with `/skill-create <name>` or in: `{skill_manager.skills_dir}`",
        )

    items = []
    for skill in skill_manager.get_all_skills():
        is_active = skill.name in skill_manager.active_skills
        items.append({
            "name": f"{'→ ' if is_active else ''}{skill.name}",
            "description": skill.description or skill.prompt[:60] + "...",
        })

    return ListResult(
        success=True,
        title="Available Skills",
        items=items,
        empty_message="No skills defined.",
    )


def skill_create_command(ctx: CommandContext) -> CommandResult:
    """Create a new skill."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /skill-create <name> [description]",
            details="Creates a skill. Use /skill-draft for AI-assisted creation.",
        )

    parts = args.split(maxsplit=1)
    name = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    skill_manager = session.skill_manager

    # Check if exists
    if skill_manager.get_skill(name):
        return ErrorResult(error=f"Skill '{name}' already exists. Use /skill-edit to modify it.")

    try:
        skill_manager.create_skill(
            name=name,
            prompt="[Enter domain-specific patterns, SQL queries, and reference material here]",
            description=description,
        )
        return TextResult(
            success=True,
            content=f"Skill **{name}** created.\n\nEdit with: `/skill-edit {name} <content>`\n\nOr use `/skill-draft {name} <description>` to generate with AI.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to create skill: {e}")


def skill_edit_command(ctx: CommandContext) -> CommandResult:
    """Edit an existing skill."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(error="Usage: /skill-edit <name> <content>")

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /skill-edit <name> <content>")

    name, content = parts[0], parts[1]
    skill_manager = session.skill_manager

    if not skill_manager.get_skill(name):
        return ErrorResult(error=f"Skill '{name}' not found. Use /skills to list available skills.")

    try:
        skill_manager.update_skill(name=name, prompt=content)
        return TextResult(
            success=True,
            content=f"Skill **{name}** updated.\n\n> {content[:100]}{'...' if len(content) > 100 else ''}",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to update skill: {e}")


def skill_delete_command(ctx: CommandContext) -> CommandResult:
    """Delete a skill."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /skill-delete <name>")

    skill_manager = session.skill_manager

    if skill_manager.delete_skill(name):
        return TextResult(success=True, content=f"Skill **{name}** deleted.")
    else:
        return ErrorResult(error=f"Skill '{name}' not found.")


def skill_deactivate_command(ctx: CommandContext) -> CommandResult:
    """Deactivate a skill."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /skill-deactivate <name>")

    skill_manager = session.skill_manager

    if skill_manager.deactivate_skill(name):
        return TextResult(success=True, content=f"Skill **{name}** deactivated.")
    else:
        return ErrorResult(error=f"Skill '{name}' was not active.")


def skill_draft_command(ctx: CommandContext) -> CommandResult:
    """Draft a skill using AI based on description."""
    session = ctx.session

    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")
    if not hasattr(session, "llm"):
        return ErrorResult(error="LLM not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /skill-draft <name> <description>",
            details="Example: /skill-draft inventory-analysis SQL patterns for inventory metrics and stock analysis",
        )

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /skill-draft <name> <description>")

    name, description = parts[0], parts[1]
    skill_manager = session.skill_manager

    # Check if exists
    if skill_manager.get_skill(name):
        return ErrorResult(error=f"Skill '{name}' already exists. Delete it first with /skill-delete.")

    try:
        content, skill_description = skill_manager.draft_skill(name, description, session.llm)
        # Save the drafted skill by creating with the generated content
        # Use update_skill_content on a newly created skill
        skill_manager.create_skill(name=name, prompt="placeholder", description=skill_description)
        skill_manager.update_skill_content(name, content)
        return TextResult(
            success=True,
            content=f"Skill **{name}** drafted and saved.\n\n**Description:** {skill_description}\n\n**Preview:**\n```\n{content[:500]}{'...' if len(content) > 500 else ''}\n```\n\nUse `/skill {name}` to activate.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to draft skill: {e}")


def prove_command(ctx: CommandContext) -> CommandResult:
    """Run proof verification on the current conversation.

    Re-runs the session's claims through the auditable solver to generate
    a verifiable proof chain with full provenance.
    """
    session = ctx.session

    # Check if we have something to prove
    if not session.datastore:
        return ErrorResult(error="No active session to prove")

    original_problem = session.datastore.get_session_meta("problem")
    if not original_problem:
        return TextResult(
            success=True,
            content="No conversation to prove. Submit a query first, then use /prove to verify it.",
        )

    # Run the proof (events will be emitted to UI)
    guidance = ctx.args.strip() if ctx.args else None
    result = session.prove_conversation(guidance=guidance)

    if result.get("error"):
        return ErrorResult(error=result["error"])

    if result.get("no_claims"):
        return TextResult(
            success=True,
            content="No claims to prove in the current conversation.",
        )

    # The actual proof facts are streamed via events
    return TextResult(
        success=True,
        content="Proof verification complete. See the Proof panel for the full audit trail.",
    )
