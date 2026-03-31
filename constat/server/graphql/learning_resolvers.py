# Copyright (c) 2025 Kenneth Stott
# Canary: e44a2ed6-427f-4040-bc9e-2545c5537dc6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for learnings, rules, skills, and agents (Phase 8)."""

from __future__ import annotations

import logging
from typing import Optional

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    AgentInfoType,
    CompactionResultType,
    CreateRuleInput,
    DeleteResultType,
    LearningInfoType,
    LearningListType,
    RuleInfoType,
    SetActiveSkillsResultType,
    SetAgentResultType,
    SkillContentType,
    SkillInfoType,
    SkillsListType,
    UpdateRuleInput,
)
from constat.server.routes.learnings import _domain_data_dirs
from constat.server.routes.skills import get_skill_manager

logger = logging.getLogger(__name__)


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


@strawberry.type
class Query:
    @strawberry.field
    async def learnings(
        self, info: Info, category: Optional[str] = None
    ) -> LearningListType:
        user_id = _require_auth(info)
        config = info.context.config

        from constat.storage.learnings import LearningCategory, LearningStore

        store = LearningStore(user_id=user_id)
        cat_enum = LearningCategory(category) if category else None
        learnings_data = store.list_raw_learnings(category=cat_enum, limit=100)
        rules_data = store.list_rules(category=cat_enum, limit=50)

        import yaml as _yaml

        seen_rule_ids = {r["id"] for r in rules_data}
        for _key, domain_cfg in (config.domains or {}).items():
            if domain_cfg.source_path:
                for d in _domain_data_dirs(domain_cfg, user_id):
                    lf = d / "learnings.yaml"
                    if lf.is_file():
                        try:
                            ldata = _yaml.safe_load(lf.read_text()) or {}
                            for rid, rdata in (ldata.get("rules") or {}).items():
                                if rid not in seen_rule_ids:
                                    seen_rule_ids.add(rid)
                                    rules_data.append({"id": rid, **rdata})
                        except Exception:
                            pass

        return LearningListType(
            learnings=[
                LearningInfoType(
                    id=l.get("id", ""),
                    content=l.get("correction", ""),
                    category=l.get("category", "user_correction"),
                    source=l.get("source"),
                    context=l.get("context"),
                    applied_count=l.get("applied_count", 0),
                    created_at=l["created"] if l.get("created") else None,
                    scope=l.get("scope"),
                )
                for l in learnings_data
            ],
            rules=[
                RuleInfoType(
                    id=r.get("id", ""),
                    summary=r.get("summary", ""),
                    category=r.get("category", "user_correction"),
                    confidence=r.get("confidence", 0.0),
                    source_count=len(r.get("source_learnings", [])),
                    tags=r.get("tags", []),
                    domain=r.get("domain"),
                    source=r.get("source"),
                    scope=r.get("scope"),
                )
                for r in rules_data
            ],
        )

    @strawberry.field
    async def skills(self, info: Info) -> SkillsListType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        manager = get_skill_manager(user_id, server_config.data_dir)

        skills = []
        for skill in manager.get_all_skills():
            skills.append(
                SkillInfoType(
                    name=skill.name,
                    description=skill.description,
                    prompt=(
                        skill.prompt[:200] + "..."
                        if len(skill.prompt) > 200
                        else skill.prompt
                    ),
                    filename=skill.filename,
                    is_active=skill.name in manager.active_skills,
                    domain=skill.domain,
                    source=skill.source,
                )
            )

        return SkillsListType(
            skills=skills,
            active_skills=list(manager.active_skills),
            skills_dir=str(manager.skills_dir),
        )

    @strawberry.field
    async def skill(self, info: Info, name: str) -> SkillContentType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        manager = get_skill_manager(user_id, server_config.data_dir)

        result = manager.get_skill_content(name)
        if not result:
            raise ValueError(f"Skill not found: {name}")

        content, path = result
        return SkillContentType(name=name, content=content, path=path)

    @strawberry.field
    async def agents(self, info: Info, session_id: str) -> list[AgentInfoType]:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session_or_none(session_id)
        if not managed or managed.user_id != user_id:
            raise ValueError("Session not found")

        session = managed.session
        if not hasattr(session, "agent_manager"):
            raise ValueError("Agent manager not available")

        agent_manager = session.agent_manager
        current_agent = agent_manager.active_agent_name

        agents = []
        for name in agent_manager.list_agents():
            agent = agent_manager.get_agent(name)
            if agent:
                agents.append(
                    AgentInfoType(
                        name=name,
                        description=agent.description,
                        domain=agent.domain,
                        source=agent.source,
                        is_active=(name == current_agent),
                    )
                )
        return agents


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def compact_learnings(self, info: Info) -> CompactionResultType:
        user_id = _require_auth(info)
        config = info.context.config

        from constat.storage.learnings import LearningStore
        from constat.learning.compactor import LearningCompactor
        from constat.providers import TaskRouter

        store = LearningStore(user_id=user_id)
        stats = store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        if unpromoted < 2:
            return CompactionResultType(
                status="skipped",
                message=f"Not enough learnings to compact ({unpromoted} pending, need at least 2)",
            )

        llm = TaskRouter(config.llm)
        compactor = LearningCompactor(store, llm)
        result = compactor.compact(dry_run=False)

        return CompactionResultType(
            status="success",
            rules_created=result.rules_created,
            rules_strengthened=result.rules_strengthened,
            rules_merged=result.rules_merged,
            learnings_archived=result.learnings_archived,
            groups_found=result.groups_found,
            skipped_low_confidence=result.skipped_low_confidence,
            errors=result.errors or None,
        )

    @strawberry.mutation
    async def delete_learning(
        self, info: Info, learning_id: str
    ) -> DeleteResultType:
        user_id = _require_auth(info)

        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)
        if not store.delete_learning(learning_id):
            raise ValueError(f"Learning not found: {learning_id}")
        return DeleteResultType(status="deleted", name=learning_id)

    @strawberry.mutation
    async def create_rule(
        self, info: Info, input: CreateRuleInput
    ) -> RuleInfoType:
        user_id = _require_auth(info)

        from constat.storage.learnings import LearningCategory, LearningStore

        store = LearningStore(user_id=user_id)
        try:
            category = LearningCategory(input.category)
        except ValueError:
            category = LearningCategory.USER_CORRECTION

        rule_id = store.save_rule(
            summary=input.summary,
            category=category,
            confidence=input.confidence,
            source_learnings=[],
            tags=input.tags,
        )
        return RuleInfoType(
            id=rule_id,
            summary=input.summary,
            category=input.category,
            confidence=input.confidence,
            source_count=0,
            tags=input.tags,
        )

    @strawberry.mutation
    async def update_rule(
        self, info: Info, rule_id: str, input: UpdateRuleInput
    ) -> RuleInfoType:
        user_id = _require_auth(info)

        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)
        rules = store.list_rules()
        existing = next((r for r in rules if r["id"] == rule_id), None)
        if not existing:
            raise ValueError(f"Rule not found: {rule_id}")

        success = store.update_rule(
            rule_id=rule_id,
            summary=input.summary,
            tags=input.tags,
            confidence=input.confidence,
        )
        if not success:
            raise ValueError(f"Rule not found: {rule_id}")

        rules = store.list_rules()
        updated = next((r for r in rules if r["id"] == rule_id), None)

        return RuleInfoType(
            id=rule_id,
            summary=updated["summary"] if updated else input.summary or existing["summary"],
            category=updated["category"] if updated else existing["category"],
            confidence=updated["confidence"] if updated else input.confidence or existing["confidence"],
            source_count=len(updated.get("source_learnings", [])) if updated else 0,
            tags=updated["tags"] if updated else input.tags or existing.get("tags", []),
        )

    @strawberry.mutation
    async def delete_rule(
        self,
        info: Info,
        rule_id: str,
        session_id: Optional[str] = None,
    ) -> DeleteResultType:
        user_id = _require_auth(info)
        config = info.context.config

        import yaml as _yaml
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)
        rule_summary: str | None = None
        rule_domain: str | None = None
        for r in store.list_rules():
            if r["id"] == rule_id:
                rule_summary = r.get("summary")
                rule_domain = r.get("domain")
                break

        if store.delete_rule(rule_id):
            if session_id and rule_domain == "__system__" and rule_summary:
                session_manager = info.context.session_manager
                managed = session_manager.get_session_or_none(session_id)
                if managed and managed.resolved_config:
                    cfg_rules = managed.resolved_config.learnings.get("rules", {})
                    for cfg_key, cfg_val in cfg_rules.items():
                        cfg_summary = (
                            cfg_val
                            if isinstance(cfg_val, str)
                            else (
                                cfg_val.get("summary", "")
                                if isinstance(cfg_val, dict)
                                else ""
                            )
                        )
                        if cfg_summary == rule_summary:
                            session_manager.write_config_tombstone(
                                session_id, "learnings", f"rules.{cfg_key}"
                            )
                            break
            return DeleteResultType(status="deleted", name=rule_id)

        for _key, domain_cfg in (config.domains or {}).items():
            if domain_cfg.source_path:
                for d in _domain_data_dirs(domain_cfg, user_id):
                    lf = d / "learnings.yaml"
                    if lf.is_file():
                        try:
                            ldata = _yaml.safe_load(lf.read_text()) or {}
                            if rule_id in (ldata.get("rules") or {}):
                                del ldata["rules"][rule_id]
                                lf.write_text(
                                    _yaml.dump(ldata, default_flow_style=False, sort_keys=False)
                                )
                                return DeleteResultType(status="deleted", name=rule_id)
                        except Exception:
                            pass

        raise ValueError(f"Rule not found: {rule_id}")

    @strawberry.mutation
    async def activate_agent(
        self,
        info: Info,
        session_id: str,
        agent_name: Optional[str] = None,
    ) -> SetAgentResultType:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session_or_none(session_id)
        if not managed or managed.user_id != user_id:
            raise ValueError("Session not found")

        session = managed.session
        if not hasattr(session, "agent_manager"):
            raise ValueError("Agent manager not available")

        agent_manager = session.agent_manager

        if not agent_name or agent_name.lower() == "none":
            agent_manager.set_active_agent(None)
            return SetAgentResultType(success=True, current_agent=None, message="Agent cleared")

        if agent_manager.set_active_agent(agent_name):
            return SetAgentResultType(
                success=True,
                current_agent=agent_name,
                message=f"Agent set to '{agent_name}'",
            )

        available = agent_manager.list_agents()
        raise ValueError(
            f"Agent not found: {agent_name}. Available: {', '.join(available) or 'none'}"
        )

    @strawberry.mutation
    async def set_active_skills(
        self, info: Info, skill_names: list[str]
    ) -> SetActiveSkillsResultType:
        user_id = _require_auth(info)
        server_config = info.context.server_config
        manager = get_skill_manager(user_id, server_config.data_dir)
        activated = manager.set_active_skills(skill_names)
        return SetActiveSkillsResultType(status="updated", active_skills=activated)
