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

    # Active agent
    if hasattr(session, "agent_manager"):
        agent_name = session.agent_manager.active_agent_name
        pairs["Agent"] = agent_name or "None"

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


def agent_command(ctx: CommandContext) -> CommandResult:
    """Set or show the current agent."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")

    agent_manager = session.agent_manager
    args = ctx.args.strip()

    if not args:
        # Show current agent
        if agent_manager.active_agent_name:
            agent = agent_manager.active_agent
            return TextResult(
                success=True,
                content=f"**Current agent:** {agent_manager.active_agent_name}\n\n> {agent.prompt[:200]}{'...' if len(agent.prompt) > 200 else ''}",
            )
        else:
            return TextResult(
                success=True,
                content="No agent active. Use `/agent <name>` to set one, or `/agents` to list available agents.",
            )

    # Set agent
    if agent_manager.set_active_agent(args):
        if args.lower() == "none":
            return TextResult(success=True, content="Agent cleared.")
        agent = agent_manager.active_agent
        return TextResult(
            success=True,
            content=f"Agent set to **{args}**\n\n> {agent.prompt[:200]}{'...' if len(agent.prompt) > 200 else ''}",
        )
    else:
        available = ", ".join(agent_manager.list_agents()) or "none"
        return ErrorResult(
            error=f"Agent not found: {args}",
            details=f"Available agents: {available}",
        )


def agents_command(ctx: CommandContext) -> CommandResult:
    """List available agents."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")

    agent_manager = session.agent_manager

    if not agent_manager.has_agents:
        return TextResult(
            success=True,
            content=f"No agents defined.\n\nCreate agents with `/agent-create <name>` or in: `{agent_manager.agents_file_path}`",
        )

    items = []
    for name in agent_manager.list_agents():
        agent = agent_manager.get_agent(name)
        is_active = name == agent_manager.active_agent_name
        items.append({
            "name": f"{'→ ' if is_active else ''}{name}",
            "description": agent.description or agent.prompt[:60] + "...",
        })

    return ListResult(
        success=True,
        title="Available Agents",
        items=items,
        empty_message="No agents defined.",
    )


def agent_create_command(ctx: CommandContext) -> CommandResult:
    """Create a new agent."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /agent-create <name> [description]",
            details="Creates an agent. You'll be prompted to enter the prompt content.",
        )

    # Parse name and optional description
    parts = args.split(maxsplit=1)
    name = parts[0]
    description = parts[1] if len(parts) > 1 else ""

    agent_manager = session.agent_manager

    # Check if exists
    if agent_manager.get_agent(name):
        return ErrorResult(error=f"Agent '{name}' already exists. Use /agent-edit to modify it.")

    # Create with placeholder prompt (user should edit)
    try:
        agent_manager.create_agent(
            name=name,
            prompt="[Enter your agent prompt here - describe the persona, communication style, and priorities]",
            description=description,
        )
        return TextResult(
            success=True,
            content=f"Agent **{name}** created.\n\nEdit the prompt with: `/agent-edit {name} <prompt>`\n\nOr use `/agent-draft {name} <description>` to generate a prompt with AI.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to create agent: {e}")


def agent_edit_command(ctx: CommandContext) -> CommandResult:
    """Edit an existing agent."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(error="Usage: /agent-edit <name> <prompt>")

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /agent-edit <name> <prompt>")

    name, prompt = parts[0], parts[1]
    agent_manager = session.agent_manager

    # Check if exists
    if not agent_manager.get_agent(name):
        return ErrorResult(error=f"Agent '{name}' not found. Use /agents to list available agents.")

    try:
        agent_manager.update_agent(name=name, prompt=prompt)
        return TextResult(
            success=True,
            content=f"Agent **{name}** updated.\n\n> {prompt[:100]}{'...' if len(prompt) > 100 else ''}",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to update agent: {e}")


def agent_delete_command(ctx: CommandContext) -> CommandResult:
    """Delete an agent."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")

    name = ctx.args.strip()
    if not name:
        return ErrorResult(error="Usage: /agent-delete <name>")

    agent_manager = session.agent_manager

    if agent_manager.delete_agent(name):
        return TextResult(success=True, content=f"Agent **{name}** deleted.")
    else:
        return ErrorResult(error=f"Agent '{name}' not found.")


def agent_draft_command(ctx: CommandContext) -> CommandResult:
    """Draft an agent using AI based on description."""
    session = ctx.session

    if not hasattr(session, "agent_manager"):
        return ErrorResult(error="Agent manager not available.")
    if not hasattr(session, "llm"):
        return ErrorResult(error="LLM not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(
            error="Usage: /agent-draft <name> <description>",
            details="Example: /agent-draft cfo Focus on financial impact and executive summaries",
        )

    parts = args.split(maxsplit=1)
    if len(parts) < 2:
        return ErrorResult(error="Usage: /agent-draft <name> <description>")

    name, description = parts[0], parts[1]
    agent_manager = session.agent_manager

    # Check if exists
    if agent_manager.get_agent(name):
        return ErrorResult(error=f"Agent '{name}' already exists. Delete it first with /agent-delete.")

    try:
        agent = agent_manager.draft_agent(name, description, session.llm)
        # Save the drafted agent
        agent_manager.create_agent(name=agent.name, prompt=agent.prompt, description=agent.description)
        return TextResult(
            success=True,
            content=f"Agent **{name}** drafted and saved.\n\n**Description:** {agent.description}\n\n**Prompt:**\n> {agent.prompt[:300]}{'...' if len(agent.prompt) > 300 else ''}\n\nUse `/agent {name}` to activate.",
        )
    except Exception as e:
        return ErrorResult(error=f"Failed to draft agent: {e}")


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


def skill_download_command(ctx: CommandContext) -> CommandResult:
    """Download a skill as a Claude Desktop compatible zip.

    Usage: /skill-download <name> [filename]
    """
    from constat.core.skill_packager import package_skill

    session = ctx.session
    if not hasattr(session, "skill_manager"):
        return ErrorResult(error="Skill manager not available.")

    args = ctx.args.strip()
    if not args:
        return ErrorResult(error="Usage: /skill-download <name> [filename]")

    parts = args.split(maxsplit=1)
    name = parts[0]
    filename = parts[1] if len(parts) > 1 else f"{name}.zip"
    if not filename.endswith(".zip"):
        filename += ".zip"

    try:
        zip_bytes = package_skill(name, session.skill_manager)
    except ValueError as e:
        return ErrorResult(error=str(e))

    from pathlib import Path
    out_path = Path(filename).resolve()
    out_path.write_bytes(zip_bytes)

    size_kb = len(zip_bytes) / 1024
    return TextResult(
        success=True,
        content=f"Saved **{out_path}** ({size_kb:.1f} KB)",
    )


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
