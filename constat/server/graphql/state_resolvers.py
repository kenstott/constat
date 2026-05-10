# Copyright (c) 2025 Kenneth Stott
# Canary: d0108a32-021f-419f-9e7b-a25fe53c9c7b
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for read-only session state."""

from __future__ import annotations

import json
import logging
from typing import Optional

import strawberry
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    ActiveAgentType,
    ActiveSkillType,
    ApiEndpointType,
    ApiFieldType,
    ApiSchemaType,
    DatabaseSchemaType,
    DatabaseTableInfoType,
    ExecutionOutputType,
    InferenceCodeListType,
    InferenceCodeType,
    MessagesType,
    ObjectivesEntryType,
    ProofFactsType,
    ProofTreeNodeType,
    ProofTreeType,
    PromptContextType,
    SaveResultType,
    ScratchpadEntryType,
    ScratchpadType,
    StepCodeListType,
    StepCodeType,
    StoredMessageType,
    StoredProofFactType,
    UpdateSystemPromptResultType,
)

logger = logging.getLogger(__name__)


def _get_managed(info: Info, session_id: str):
    sm = info.context.session_manager
    managed = sm.get_session_or_none(session_id)
    if not managed:
        raise ValueError(f"Session {session_id} not found")
    return managed


@strawberry.type
class Query:
    @strawberry.field
    async def steps(self, info: Info, session_id: str) -> StepCodeListType:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)

        if managed:
            history = managed.session.history
            history_session_id = managed.session.session_id
        else:
            from constat.storage.history import SessionHistory
            history = SessionHistory(user_id=user_id)
            history_session_id = history.find_session_by_server_id(session_id)

        raw = history.list_step_codes(history_session_id) if history_session_id else []
        steps = [
            StepCodeType(
                step_number=s["step_number"],
                goal=s.get("goal", ""),
                code=s.get("code", ""),
                prompt=s.get("prompt"),
                model=s.get("model"),
            )
            for s in raw
        ]
        return StepCodeListType(steps=steps, total=len(steps))

    @strawberry.field
    async def inference_codes(self, info: Info, session_id: str) -> InferenceCodeListType:
        sm = info.context.session_manager
        user_id = info.context.user_id
        managed = sm.get_session_or_none(session_id)

        if managed:
            history = managed.session.history
            history_session_id = managed.session.session_id
        else:
            from constat.storage.history import SessionHistory
            history = SessionHistory(user_id=user_id)
            history_session_id = history.find_session_by_server_id(session_id)

        raw = history.list_inference_codes(history_session_id) if history_session_id else []
        inferences = [
            InferenceCodeType(
                inference_id=ic["inference_id"],
                name=ic.get("name", ""),
                operation=ic.get("operation", ""),
                code=ic.get("code", ""),
                attempt=ic.get("attempt", 0),
                prompt=ic.get("prompt"),
                model=ic.get("model"),
            )
            for ic in raw
        ]
        return InferenceCodeListType(inferences=inferences, total=len(inferences))

    @strawberry.field
    async def scratchpad(self, info: Info, session_id: str) -> ScratchpadType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            return ScratchpadType(entries=[], total=0)
        raw = managed.session.datastore.get_scratchpad()
        entries = [
            ScratchpadEntryType(
                step_number=e.get("step_number", 0),
                goal=e.get("goal", ""),
                narrative=e.get("narrative", ""),
                tables_created=e.get("tables_created", []),
                code=e.get("code", ""),
                user_query=e.get("user_query", ""),
                objective_index=e.get("objective_index"),
            )
            for e in raw
        ]
        return ScratchpadType(entries=entries, total=len(entries))

    @strawberry.field
    async def session_ddl(self, info: Info, session_id: str) -> str:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            return ""
        return managed.session.datastore.get_ddl()

    @strawberry.field
    async def execution_output(self, info: Info, session_id: str) -> ExecutionOutputType:
        managed = _get_managed(info, session_id)
        output = ""
        if managed.session.scratchpad:
            recent = managed.session.scratchpad.get_recent_context(max_steps=1)
            if recent:
                output = recent
        return ExecutionOutputType(
            output=output,
            suggestions=[],
            current_query=managed.current_query,
        )

    @strawberry.field
    async def session_routing(self, info: Info, session_id: str) -> JSON:
        managed = _get_managed(info, session_id)
        session = managed.session
        if not hasattr(session, "router") or not session.router:
            return {}

        from constat.server.models import ModelRouteInfo

        layers = session.router.get_routing_layers(active_domains=managed.active_domains)
        default_provider = session.config.llm.provider

        result = {}
        for layer_name, routes in layers.items():
            result[layer_name] = {}
            for task_type, specs in routes.items():
                result[layer_name][task_type] = [
                    ModelRouteInfo(
                        provider=spec.provider or default_provider,
                        model=spec.model,
                    ).model_dump()
                    for spec in specs
                ]
        return result

    @strawberry.field
    async def proof_tree(self, info: Info, session_id: str) -> ProofTreeType:
        managed = _get_managed(info, session_id)
        all_facts = managed.session.fact_resolver.get_all_facts()
        nodes = [
            ProofTreeNodeType(
                name=name,
                value=fact.value,
                source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                reasoning=fact.reasoning,
                dependencies=getattr(fact, "dependencies", []),
            )
            for name, fact in all_facts.items()
        ]
        return ProofTreeType(facts=nodes)

    @strawberry.field
    async def proof_facts(self, info: Info, session_id: str) -> ProofFactsType:
        managed = _get_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        facts_raw, summary = history.load_proof_facts_by_server_id(session_id)
        facts = [
            StoredProofFactType(
                id=f.get("id", ""),
                name=f.get("name", ""),
                description=f.get("description"),
                status=f.get("status", ""),
                value=f.get("value"),
                source=f.get("source"),
                confidence=f.get("confidence"),
                tier=f.get("tier"),
                strategy=f.get("strategy"),
                formula=f.get("formula"),
                reason=f.get("reason"),
                dependencies=f.get("dependencies", []),
                elapsed_ms=f.get("elapsed_ms"),
            )
            for f in facts_raw
        ]
        return ProofFactsType(facts=facts, summary=summary)

    @strawberry.field
    async def messages(self, info: Info, session_id: str) -> MessagesType:
        managed = _get_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        raw = history.load_messages_by_server_id(session_id)
        msgs = [
            StoredMessageType(
                id=m.get("id", ""),
                type=m.get("type", ""),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", ""),
                step_number=m.get("stepNumber") or m.get("step_number"),
                is_final_insight=m.get("isFinalInsight") or m.get("is_final_insight"),
                step_duration_ms=m.get("stepDurationMs") or m.get("step_duration_ms"),
                role=m.get("role"),
                skills=m.get("skills"),
            )
            for m in raw
        ]
        return MessagesType(messages=msgs)

    @strawberry.field
    async def objectives(self, info: Info, session_id: str) -> list[ObjectivesEntryType]:
        managed = _get_managed(info, session_id)
        log_json = None
        if managed.session.datastore:
            log_json = managed.session.datastore.get_session_meta("objectives_log")
        raw = json.loads(log_json) if log_json else []
        return [
            ObjectivesEntryType(
                type=o.get("type", ""),
                text=o.get("text"),
                question=o.get("question"),
                answer=o.get("answer"),
                mode=o.get("mode"),
                guidance=o.get("guidance"),
                ts=o.get("ts"),
            )
            for o in raw
        ]

    @strawberry.field
    async def prompt_context(self, info: Info, session_id: str) -> PromptContextType:
        managed = _get_managed(info, session_id)
        user_id = info.context.user_id
        if managed.user_id != user_id:
            raise ValueError("Not authorized")

        session = managed.session
        system_prompt = session._get_system_prompt() if hasattr(session, '_get_system_prompt') else (session.config.system_prompt or "")

        active_agent = None
        if hasattr(session, "agent_manager"):
            agent_name = session.agent_manager.active_agent_name
            if agent_name:
                agent = session.agent_manager.get_agent(agent_name)
                active_agent = ActiveAgentType(
                    name=agent_name,
                    prompt=agent.prompt if agent else "",
                )

        active_skills = []
        if hasattr(session, "skill_manager"):
            for name in session.skill_manager.active_skills:
                skill = session.skill_manager.get_skill(name)
                if skill:
                    active_skills.append(ActiveSkillType(
                        name=skill.name,
                        prompt=skill.prompt,
                        description=skill.description,
                    ))

        return PromptContextType(
            system_prompt=system_prompt,
            active_agent=active_agent,
            active_skills=active_skills,
        )

    @strawberry.field
    async def database_schema(self, info: Info, session_id: str, db_name: str) -> DatabaseSchemaType:
        managed = _get_managed(info, session_id)
        if not managed.has_database(db_name):
            raise ValueError(f"Database not found: {db_name}")
        tables = managed.session.schema_manager.get_tables_for_db(db_name)
        return DatabaseSchemaType(
            database=db_name,
            tables=[
                DatabaseTableInfoType(
                    name=t.name,
                    row_count=t.row_count,
                    column_count=len(t.columns),
                )
                for t in tables
            ],
        )

    @strawberry.field
    async def api_schema(self, info: Info, session_id: str, api_name: str) -> ApiSchemaType:
        managed = _get_managed(info, session_id)
        api_config = managed.session.config.apis.get(api_name)
        if not api_config:
            for domain_filename in managed.active_domains:
                domain = managed.session.config.load_domain(domain_filename)
                if domain and api_name in domain.apis:
                    api_config = domain.apis[api_name]
                    break
        if not api_config:
            raise ValueError(f"API not found: {api_name}")

        endpoints_meta = managed.session.api_schema_manager.get_api_schema(api_name)
        return ApiSchemaType(
            name=api_name,
            type=api_config.type,
            description=api_config.description,
            endpoints=[
                ApiEndpointType(
                    name=ep.endpoint_name,
                    kind=ep.api_type,
                    return_type=ep.return_type,
                    description=ep.description,
                    http_method=ep.http_method,
                    http_path=ep.http_path,
                    fields=[
                        ApiFieldType(
                            name=f.name,
                            type=f.type,
                            description=f.description,
                            is_required=f.is_required,
                        )
                        for f in ep.fields
                    ],
                )
                for ep in endpoints_meta
            ],
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def save_proof_facts(
        self, info: Info, session_id: str, facts: JSON, summary: Optional[str] = None,
    ) -> SaveResultType:
        managed = _get_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        history.save_proof_facts_by_server_id(session_id, facts, summary)
        return SaveResultType(status="saved", count=len(facts))

    @strawberry.mutation
    async def save_messages(
        self, info: Info, session_id: str, messages: JSON,
    ) -> SaveResultType:
        managed = _get_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        history.save_messages_by_server_id(session_id, messages)
        return SaveResultType(status="saved", count=len(messages))

    @strawberry.mutation
    async def update_system_prompt(
        self, info: Info, session_id: str, system_prompt: str,
    ) -> UpdateSystemPromptResultType:
        managed = _get_managed(info, session_id)
        user_id = info.context.user_id
        if managed.user_id != user_id:
            raise ValueError("Not authorized")
        managed.session.config.system_prompt = system_prompt
        managed.session_prompt = system_prompt
        return UpdateSystemPromptResultType(status="updated", system_prompt=system_prompt)
