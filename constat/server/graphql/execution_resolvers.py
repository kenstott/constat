# Copyright (c) 2025 Kenneth Stott
# Canary: 87c84475-cf47-4d31-961d-d27b3875b0c4
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for query execution (Phase 7).

Ports the REST execution endpoints from routes/queries.py into
GraphQL mutations and a query, enabling the frontend to migrate away from
GraphQL subscriptions.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import strawberry
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    ApprovePlanInput,
    AutocompleteItemType,
    AutocompleteResultType,
    ExecutionActionResultType,
    ExecutionPlanType,
    PlanStepType,
    QuerySubmissionType,
    SubmitQueryInput,
)
from constat.server.models import EventType, SessionStatus

logger = logging.getLogger(__name__)


def _plan_to_gql(plan) -> ExecutionPlanType:
    steps = []
    for s in plan.steps:
        result = None
        if s.result:
            result = {
                "success": s.result.success,
                "stdout": s.result.stdout,
                "error": s.result.error,
                "attempts": s.result.attempts,
                "duration_ms": s.result.duration_ms,
                "tables_created": s.result.tables_created,
                "tables_modified": s.result.tables_modified,
            }
        step_code = s.code
        if step_code is not None and not isinstance(step_code, str):
            logger.warning(
                "step.code is not a string (step=%s, type=%s, value=%r) — coercing to None",
                getattr(s, "number", "?"), type(step_code).__name__, step_code,
            )
            step_code = None
        steps.append(PlanStepType(
            number=s.number,
            goal=s.goal,
            status=s.status.value,
            expected_inputs=s.expected_inputs,
            expected_outputs=s.expected_outputs,
            depends_on=s.depends_on,
            code=step_code,
            domain=s.domain,
            result=result,
        ))
    return ExecutionPlanType(
        problem=plan.problem,
        steps=steps,
        current_step=plan.current_step,
        completed_steps=plan.completed_steps,
        failed_steps=plan.failed_steps,
        is_complete=plan.is_complete,
    )


@strawberry.type
class Query:
    @strawberry.field
    async def execution_plan(self, info: Info, session_id: str) -> ExecutionPlanType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")
        if not managed.session.plan:
            raise ValueError("No plan exists for this session")
        return _plan_to_gql(managed.session.plan)


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def submit_query(self, info: Info, session_id: str, input: SubmitQueryInput) -> QuerySubmissionType:
        from constat.server.routes.queries import (
            _active_tasks,
            _create_event_handler,
            _execute_query_async,
        )

        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        # Cancel busy session
        if managed.status in (SessionStatus.PLANNING, SessionStatus.EXECUTING, SessionStatus.AWAITING_APPROVAL):
            logger.info(f"Session {session_id} busy ({managed.status.value}), cancelling for new query")
            if hasattr(managed.session, "_cancelled"):
                managed.session._cancelled = True
            if hasattr(managed.session, "_execution_context"):
                managed.session._execution_context.cancel()
            if managed.approval_event:
                managed.approval_response = {"approved": False, "feedback": "Cancelled by user"}
                managed.approval_event.set()
            if managed.clarification_event:
                managed.clarification_response = {"skip": True}
                managed.clarification_event.set()
            managed.status = SessionStatus.IDLE
            managed.current_query = None
            managed.execution_id = None

        execution_id = str(uuid.uuid4())

        # Slash command fast path
        stripped = input.problem.strip()
        _lower = stripped.lower()
        _async_commands = ("/redo", "/reason", "/replay")
        if stripped.startswith("/") and not any(_lower.startswith(c) for c in _async_commands):
            try:
                from constat.commands.registry import is_command
                if is_command(stripped):
                    if managed.session.doc_tools and hasattr(managed, "active_domains"):
                        managed.session.doc_tools._active_domain_ids = managed.active_domains or []
                    result = managed.session._handle_slash_command(stripped)
                    status = "completed" if result.get("success", True) else "error"
                    return QuerySubmissionType(
                        execution_id=execution_id,
                        status=status,
                        message=result.get("output") or (result.get("error", "Command failed") if status == "error" else "Command executed."),
                    )
            except Exception as e:
                logger.warning(f"Slash command fast path failed, falling back to async: {e}")

        # Reset cancellation flag
        if hasattr(managed.session, "_cancelled"):
            managed.session._cancelled = False
        managed.current_query = input.problem
        managed.execution_id = execution_id
        managed.status = SessionStatus.PLANNING

        require_approval = input.require_approval
        if require_approval is not strawberry.UNSET and require_approval is not None:
            managed.session.session_config.require_approval = require_approval
            managed.session.session_config.auto_approve = not require_approval
            managed.session.session_config.force_approval = require_approval

        handler = _create_event_handler(managed, sm)
        managed.session._event_handlers = []
        managed.session.on_event(handler)

        task = asyncio.create_task(
            _execute_query_async(
                managed,
                input.problem,
                execution_id,
                input.is_followup,
                replay=input.replay,
                objective_index=input.objective_index if input.objective_index is not strawberry.UNSET else None,
                session_manager=sm,
            )
        )
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)

        return QuerySubmissionType(
            execution_id=execution_id,
            status="started",
            message="Query execution started. Subscribe to queryExecution for progress events.",
        )

    @strawberry.mutation
    async def cancel_execution(self, info: Info, session_id: str) -> ExecutionActionResultType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        if hasattr(managed.session, "_cancelled"):
            managed.session._cancelled = True
        if hasattr(managed.session, "_execution_context"):
            managed.session._execution_context.cancel()
        managed.status = SessionStatus.CANCELLED

        return ExecutionActionResultType(status="cancelling", message="Cancellation requested")

    @strawberry.mutation
    async def approve_plan(self, info: Info, session_id: str, input: ApprovePlanInput) -> ExecutionActionResultType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        if managed.approval_event is None:
            raise ValueError("Session is not awaiting plan approval")

        deleted_steps = input.deleted_steps if input.deleted_steps is not strawberry.UNSET else None
        edited_steps_raw = input.edited_steps if input.edited_steps is not strawberry.UNSET else None
        edited_steps = None
        if edited_steps_raw:
            edited_steps = [{"number": s.number, "goal": s.goal} for s in edited_steps_raw]

        feedback = input.feedback if input.feedback is not strawberry.UNSET else None

        managed.approval_response = {
            "approved": input.approved,
            "feedback": feedback,
            "deleted_steps": deleted_steps,
            "edited_steps": edited_steps,
        }
        managed.approval_event.set()

        if input.approved:
            managed.status = SessionStatus.EXECUTING
            deleted_count = len(deleted_steps) if deleted_steps else 0
            msg = "Plan approved, execution continuing"
            if deleted_count > 0:
                msg += f" ({deleted_count} step(s) removed)"
            return ExecutionActionResultType(status="approved", message=msg)
        else:
            managed.status = SessionStatus.IDLE
            return ExecutionActionResultType(status="rejected", message=f"Plan rejected: {feedback}")

    @strawberry.mutation
    async def answer_clarification(
        self,
        info: Info,
        session_id: str,
        answers: JSON,
        structured_answers: JSON,
    ) -> ExecutionActionResultType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        managed.clarification_response = {
            "answers": answers,
            "structured_answers": structured_answers,
            "skip": False,
        }
        if managed.clarification_event:
            managed.clarification_event.set()
        return ExecutionActionResultType(status="ok")

    @strawberry.mutation
    async def skip_clarification(self, info: Info, session_id: str) -> ExecutionActionResultType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        managed.clarification_response = {"skip": True}
        if managed.clarification_event:
            managed.clarification_event.set()
        return ExecutionActionResultType(status="ok")

    @strawberry.mutation
    async def replan_from(
        self,
        info: Info,
        session_id: str,
        step_number: int,
        mode: str,
        edited_goal: Optional[str] = None,
    ) -> ExecutionActionResultType:
        from constat.server.routes.queries import (
            _active_tasks,
            _create_event_handler,
            _executor,
        )

        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        replan_start = {
            "event_type": "replan_start",
            "session_id": session_id,
            "step_number": step_number,
            "data": {"mode": mode, "from_step": step_number},
        }
        sm.publish_execution_event(session_id, replan_start)

        managed.status = SessionStatus.EXECUTING
        execution_id = str(uuid.uuid4())
        managed.execution_id = execution_id

        handler = _create_event_handler(managed, sm)
        managed.session._event_handlers = []
        managed.session.on_event(handler)

        async def _run():
            _loop = asyncio.get_event_loop()
            try:
                result = await _loop.run_in_executor(
                    _executor,
                    lambda: managed.session.replan_from_step(step_number, mode, edited_goal),
                )
                _payload = {
                    "event_type": EventType.QUERY_COMPLETE.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "final_answer": result.get("final_answer", ""),
                        "suggestions": result.get("suggestions", []),
                    },
                }
                sm.publish_execution_event(session_id, _payload)
            except Exception as err:
                logger.error(f"Replan error: {err}")
                _payload = {
                    "event_type": EventType.QUERY_ERROR.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {"error": str(err)},
                }
                sm.publish_execution_event(session_id, _payload)
            finally:
                managed.status = SessionStatus.IDLE
                managed.execution_id = None

        task = asyncio.create_task(_run())
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)

        return ExecutionActionResultType(status="started")

    @strawberry.mutation
    async def edit_objective(
        self,
        info: Info,
        session_id: str,
        objective_index: int,
        new_text: str,
    ) -> ExecutionActionResultType:
        from constat.server.routes.queries import (
            _active_tasks,
            _create_event_handler,
            _executor,
        )

        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        managed.status = SessionStatus.EXECUTING
        execution_id = str(uuid.uuid4())
        managed.execution_id = execution_id

        handler = _create_event_handler(managed, sm)
        managed.session._event_handlers = []
        managed.session.on_event(handler)

        async def _run():
            _loop = asyncio.get_event_loop()
            try:
                result = await _loop.run_in_executor(
                    _executor,
                    lambda: managed.session.edit_objective(objective_index, new_text),
                )
                _payload = {
                    "event_type": EventType.QUERY_COMPLETE.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "final_answer": result.get("final_answer", ""),
                        "suggestions": result.get("suggestions", []),
                    },
                }
                sm.publish_execution_event(session_id, _payload)
            except Exception as err:
                logger.error(f"Edit objective error: {err}")
                _payload = {
                    "event_type": EventType.QUERY_ERROR.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {"error": str(err)},
                }
                sm.publish_execution_event(session_id, _payload)
            finally:
                managed.status = SessionStatus.IDLE
                managed.execution_id = None

        task = asyncio.create_task(_run())
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)

        return ExecutionActionResultType(status="started")

    @strawberry.mutation
    async def delete_objective(
        self,
        info: Info,
        session_id: str,
        objective_index: int,
    ) -> ExecutionActionResultType:
        from constat.server.routes.queries import (
            _active_tasks,
            _create_event_handler,
            _executor,
        )

        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        managed.status = SessionStatus.EXECUTING
        execution_id = str(uuid.uuid4())
        managed.execution_id = execution_id

        handler = _create_event_handler(managed, sm)
        managed.session._event_handlers = []
        managed.session.on_event(handler)

        async def _run():
            _loop = asyncio.get_event_loop()
            try:
                result = await _loop.run_in_executor(
                    _executor,
                    lambda: managed.session.delete_objective(objective_index),
                )
                _payload = {
                    "event_type": EventType.QUERY_COMPLETE.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {
                        "success": result.get("success", False),
                        "output": result.get("output", ""),
                        "final_answer": result.get("final_answer", ""),
                        "suggestions": result.get("suggestions", []),
                    },
                }
                sm.publish_execution_event(session_id, _payload)
            except Exception as err:
                logger.error(f"Delete objective error: {err}")
                _payload = {
                    "event_type": EventType.QUERY_ERROR.value,
                    "session_id": session_id,
                    "step_number": 0,
                    "data": {"error": str(err)},
                }
                sm.publish_execution_event(session_id, _payload)
            finally:
                managed.status = SessionStatus.IDLE
                managed.execution_id = None

        task = asyncio.create_task(_run())
        _active_tasks.add(task)
        task.add_done_callback(_active_tasks.discard)

        return ExecutionActionResultType(status="started")

    @strawberry.mutation
    async def request_autocomplete(
        self,
        info: Info,
        session_id: str,
        context: str,
        prefix: str,
        parent: Optional[str] = None,
    ) -> AutocompleteResultType:
        sm = info.context.session_manager
        managed = sm.get_session_or_none(session_id)
        if not managed:
            raise ValueError(f"Session not found: {session_id}")

        request_id = str(uuid.uuid4())
        _prefix = prefix.lower()
        items: list[AutocompleteItemType] = []

        try:
            if context == "table":
                tables = managed.session.datastore.list_tables()
                for t in tables:
                    name = t.get("name", "")
                    if name.lower().startswith(_prefix):
                        items.append(AutocompleteItemType(
                            label=name,
                            value=name,
                            description=t.get("description") or f"{t.get('row_count', 0)} rows",
                        ))
            elif context == "column" and parent:
                schema = managed.session.datastore.get_table_schema(parent)
                if schema:
                    for col in schema:
                        name = col.get("name", "")
                        if name.lower().startswith(_prefix):
                            items.append(AutocompleteItemType(
                                label=name,
                                value=name,
                                description=col.get("type", ""),
                            ))
            elif context == "entity":
                if hasattr(managed.session, "schema_manager") and managed.session.schema_manager:
                    entities = managed.session.schema_manager.get_schema_entities(include_columns=False)
                    for entity in entities:
                        if entity.lower().startswith(_prefix):
                            items.append(AutocompleteItemType(
                                label=entity,
                                value=entity,
                                description="Table",
                            ))
        except Exception as ac_err:
            logger.warning(f"Autocomplete error: {ac_err}")

        return AutocompleteResultType(request_id=request_id, items=items[:20])

    @strawberry.mutation
    async def heartbeat(self, info: Info, session_id: str, since: Optional[str] = None) -> ExecutionActionResultType:
        sm = info.context.session_manager
        server_time = sm.process_heartbeat(session_id, since)
        return ExecutionActionResultType(status="ok", message=server_time)
