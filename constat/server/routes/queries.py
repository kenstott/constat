# Copyright (c) 2025 Kenneth Stott
# Canary: 52acd853-d514-4d89-9235-66bf5de001fc
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Query execution REST endpoints."""

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.api.types import SolveResult, FollowUpResult
from constat.messages import WelcomeMessage
from constat.server.models import (
    ApprovalRequest,
    ApprovalResponse,
    EventType,
    PlanResponse,
    QueryRequest,
    QueryResponse,
    SessionStatus,
    StepEventWS,
    StepResponse,
)
from constat.server.session_manager import SessionManager, ManagedSession
from constat.session import (
    StepEvent,
    PlanApprovalRequest,
    PlanApprovalResponse,
    ClarificationRequest,
    ClarificationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Thread pool for running synchronous Session.solve() calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-worker")

# Track active query tasks for graceful shutdown
_active_tasks: set[asyncio.Task] = set()

async def shutdown_executor_async() -> None:
    """Shutdown the thread pool executor and cancel active tasks. Called during app shutdown."""
    # Cancel all active query tasks
    tasks_to_cancel = list(_active_tasks)
    for task in tasks_to_cancel:
        task.cancel()

    # Wait briefly for tasks to handle cancellation
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    _active_tasks.clear()

    # Then shutdown the executor
    _executor.shutdown(wait=False, cancel_futures=True)


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


def _step_to_response(step) -> StepResponse:
    """Convert a Step to StepResponse."""
    result_dict = None
    if step.result:
        result_dict = {
            "success": step.result.success,
            "stdout": step.result.stdout,
            "error": step.result.error,
            "attempts": step.result.attempts,
            "duration_ms": step.result.duration_ms,
            "tables_created": step.result.tables_created,
            "tables_modified": step.result.tables_modified,
        }

    return StepResponse(
        number=step.number,
        goal=step.goal,
        status=step.status.value,
        expected_inputs=step.expected_inputs,
        expected_outputs=step.expected_outputs,
        depends_on=step.depends_on,
        code=step.code,
        domain=step.domain,
        result=result_dict,
    )


def _plan_to_response(plan) -> PlanResponse:
    """Convert a Plan to PlanResponse."""
    return PlanResponse(
        problem=plan.problem,
        steps=[_step_to_response(s) for s in plan.steps],
        current_step=plan.current_step,
        completed_steps=plan.completed_steps,
        failed_steps=plan.failed_steps,
        is_complete=plan.is_complete,
    )


def _api_result_to_dict(result: SolveResult | FollowUpResult) -> dict[str, Any]:
    """Convert API result dataclass to dict for existing handlers.

    The server's existing code expects dict results from session.solve().
    This bridges the API's clean types back to the expected format.
    """
    return {
        "success": result.success,
        "summary": result.answer,
        "answer": result.answer,
        "final_answer": result.answer,
        "output": result.answer,
        "error": result.error,
        "raw_output": result.raw_output,
        "suggestions": list(result.suggestions),
        "tables_created": list(result.tables_created),
        "artifacts": [
            {"id": a.id, "name": a.name, "type": a.artifact_type, "step_number": a.step_number}
            for a in result.artifacts
        ],
        "plan": {"goal": result.plan_goal} if hasattr(result, 'plan_goal') and result.plan_goal else None,
        "step_results": [
            {"description": s.description, "status": s.status, "code": s.code}
            for s in result.steps
        ],
    }


def _api_result_to_dict_raw(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a raw dict result (from session.replay()) to the standard output format."""
    output = result.get("final_answer") or result.get("output", "")
    return {
        "success": result.get("success", False),
        "output": output,
        "final_answer": output,
        "error": result.get("error"),
        "suggestions": result.get("suggestions", []),
    }


def _create_approval_callback(managed: ManagedSession, loop: asyncio.AbstractEventLoop, session_manager: "SessionManager | None" = None):
    """Create an approval callback that bridges sync session to async GraphQL subscription.

    This callback is called from the synchronous session.solve() thread when
    plan approval is required. It:
    1. Updates status to awaiting_approval
    2. Emits plan_ready event via GraphQL subscription
    3. Blocks waiting for approval_event to be set
    4. Returns the approval response
    """
    def approval_callback(request: PlanApprovalRequest) -> PlanApprovalResponse:
        logger.debug(f"Approval callback called for session {managed.session_id}")

        # Update status
        managed.status = SessionStatus.AWAITING_APPROVAL

        # Create approval event for waiting
        managed.approval_event = asyncio.Event()
        managed.approval_response = None

        plan_data = {
            "problem": request.problem,
            "steps": request.steps,
            "reasoning": request.reasoning,
        }
        payload = {
            "event_type": EventType.PLAN_READY.value,
            "session_id": managed.session_id,
            "step_number": 0,
            "data": {"plan": plan_data},
        }
        if session_manager is not None:
            try:
                loop.call_soon_threadsafe(
                    session_manager.publish_execution_event, managed.session_id, payload
                )
            except RuntimeError:
                session_manager.publish_execution_event(managed.session_id, payload)

        # Wait for approval (blocking in thread, but event is set from async context)
        # Use run_coroutine_threadsafe to wait on the async event from sync code
        async def wait_for_approval():
            await managed.approval_event.wait()
            return managed.approval_response

        future = asyncio.run_coroutine_threadsafe(wait_for_approval(), loop)
        try:
            response = future.result()
        except Exception as e:
            logger.error(f"Error waiting for approval: {type(e).__name__}: {e}")
            return PlanApprovalResponse.reject("Approval interrupted")

        if response is None:
            return PlanApprovalResponse.approve()

        if response.get("approved"):
            managed.status = SessionStatus.EXECUTING
            deleted_steps = response.get("deleted_steps")
            edited_steps = response.get("edited_steps")
            return PlanApprovalResponse.approve(deleted_steps=deleted_steps, edited_steps=edited_steps)
        else:
            feedback = response.get("feedback", "")
            edited_steps = response.get("edited_steps")
            # If user provided feedback, return SUGGEST to trigger replanning
            # within the solve() loop (preserves original problem context)
            if feedback and feedback != "Rejected by user" and feedback != "Cancelled by user":
                managed.status = SessionStatus.PLANNING
                return PlanApprovalResponse.suggest(feedback, edited_steps=edited_steps)
            else:
                # No feedback = pure rejection, end the flow
                managed.status = SessionStatus.IDLE
                return PlanApprovalResponse.reject(feedback or "Rejected by user")

    return approval_callback


def _create_clarification_callback(managed: ManagedSession, loop: asyncio.AbstractEventLoop, session_manager: "SessionManager | None" = None):
    """Create a clarification callback that bridges sync session to async GraphQL subscription.

    This callback is called from the synchronous session.solve() thread when
    clarification is needed. It sends all questions at once for stepper UI.
    """
    def clarification_callback(request: ClarificationRequest) -> ClarificationResponse:
        logger.debug(f"Clarification callback called for session {managed.session_id}")

        # Update status
        managed.status = SessionStatus.AWAITING_APPROVAL

        # Create event for waiting
        managed.clarification_event = asyncio.Event()
        managed.clarification_response = None

        clarification_data = {
            "original_question": request.original_question,
            "ambiguity_reason": request.ambiguity_reason,
            "questions": [
                {
                    "text": q.text,
                    "suggestions": q.suggestions,
                    **({"widget": {"type": q.widget.type.value, "config": q.widget.config}} if q.widget else {}),
                }
                for q in request.questions
            ],
        }
        payload = {
            "event_type": EventType.CLARIFICATION_NEEDED.value,
            "session_id": managed.session_id,
            "step_number": 0,
            "data": clarification_data,
        }
        if session_manager is not None:
            try:
                loop.call_soon_threadsafe(
                    session_manager.publish_execution_event, managed.session_id, payload
                )
            except RuntimeError:
                session_manager.publish_execution_event(managed.session_id, payload)

        # Wait for all answers
        async def wait_for_clarification():
            await managed.clarification_event.wait()
            return managed.clarification_response

        future = asyncio.run_coroutine_threadsafe(wait_for_clarification(), loop)
        try:
            response = future.result()
        except Exception as e:
            logger.error(f"Error waiting for clarification: {e}")
            return ClarificationResponse(answers={}, skip=True)

        if response is None or response.get("skip"):
            return ClarificationResponse(answers={}, skip=True)

        managed.status = SessionStatus.PLANNING
        return ClarificationResponse(
            answers=response.get("answers", {}),
            skip=False,
            structured_answers=response.get("structured_answers", {}),
        )

    return clarification_callback


def _create_event_handler(managed: ManagedSession, session_manager: "SessionManager | None" = None):
    """Create an event handler that publishes events to GraphQL subscribers.

    The handler is called from a thread-pool thread (run_in_executor), so
    we use loop.call_soon_threadsafe to schedule publish on the event loop.
    """
    loop = asyncio.get_event_loop()

    # Map session event types to API event types
    event_type_map = {
        "step_start": EventType.STEP_START,
        "generating": EventType.STEP_GENERATING,
        "executing": EventType.STEP_EXECUTING,
        "step_complete": EventType.STEP_COMPLETE,
        "step_error": EventType.STEP_ERROR,
        "step_failed": EventType.STEP_FAILED,
        "model_escalation": EventType.MODEL_ESCALATION,
        "facts_extracted": EventType.FACTS_EXTRACTED,
        "progress": EventType.PROGRESS,
        "dynamic_context": EventType.DYNAMIC_CONTEXT,
        "planning_start": EventType.PLANNING_START,
        "plan_ready": EventType.PLAN_READY,
        "plan_updated": EventType.PLAN_UPDATED,
        "proof_start": EventType.PROOF_START,
        "replanning": EventType.REPLANNING,
        "steps_truncated": EventType.STEPS_TRUNCATED,
        "synthesizing": EventType.SYNTHESIZING,
        "generating_insights": EventType.GENERATING_INSIGHTS,
        # Fact resolution events (for proof DAG)
        "fact_start": EventType.FACT_START,
        "fact_planning": EventType.FACT_PLANNING,
        "fact_executing": EventType.FACT_EXECUTING,
        "fact_resolved": EventType.FACT_RESOLVED,
        "fact_failed": EventType.FACT_FAILED,
        "dag_execution_start": EventType.DAG_EXECUTION_START,
        "inference_code": EventType.INFERENCE_CODE,
        "premise_resolving": EventType.PREMISE_RESOLVING,
        "premise_resolved": EventType.PREMISE_RESOLVED,
        "inference_executing": EventType.INFERENCE_EXECUTING,
        "inference_complete": EventType.INFERENCE_COMPLETE,
        "proof_complete": EventType.PROOF_COMPLETE,
        "proof_summary_ready": EventType.PROOF_SUMMARY_READY,
    }

    def handle_event(event: StepEvent) -> None:
        """Handle session events by publishing to GraphQL subscribers."""
        try:
            if event.event_type == "dynamic_context":
                logger.info(f"[EVENT_HANDLER] Received dynamic_context event: {event.data}")

            api_event_type = event_type_map.get(event.event_type, EventType.PROGRESS)

            # Sanitize event data — replace non-serializable types (recursive)
            safe_data = event.data
            if safe_data:
                import pandas as pd

                def _sanitize(obj):
                    if isinstance(obj, pd.DataFrame):
                        return f"DataFrame({len(obj)} rows, {len(obj.columns)} cols)"
                    if isinstance(obj, dict):
                        return {k: _sanitize(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_sanitize(v) for v in obj]
                    return obj

                safe_data = _sanitize(safe_data)

            ws_event = StepEventWS(
                event_type=api_event_type,
                session_id=managed.session_id,
                step_number=event.step_number,
                data=safe_data,
            )

            payload = ws_event.model_dump(mode="json")

            if session_manager is not None:
                try:
                    loop.call_soon_threadsafe(
                        session_manager.publish_execution_event, managed.session_id, payload
                    )
                except RuntimeError:
                    session_manager.publish_execution_event(managed.session_id, payload)

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    return handle_event


def _run_query(managed: ManagedSession, problem: str, loop: asyncio.AbstractEventLoop, is_followup: bool = False, replay: bool = False, objective_index: int | None = None, session_manager: "SessionManager | None" = None) -> dict[str, Any]:
    """Run a query synchronously (called from thread pool).

    Uses the ConstatAPI for solve/follow_up operations, ensuring
    consistent behavior with the REPL.

    Args:
        managed: The managed session
        problem: Query problem text
        loop: Event loop for async bridge
        is_followup: Whether this is a follow-up to a previous query
        replay: Whether to replay stored code instead of LLM codegen
        objective_index: If replaying, only replay entries with this objective_index

    Returns:
        Query result dict
    """
    try:
        # Register approval callback if plan approval is required
        if managed.session.session_config.require_approval:
            approval_callback = _create_approval_callback(managed, loop, session_manager)
            managed.api.set_approval_callback(approval_callback)
            logger.debug(f"Registered approval callback for session {managed.session_id}")

        # Register clarification callback if clarifications are enabled
        if managed.session.session_config.ask_clarifications:
            clarification_callback = _create_clarification_callback(managed, loop, session_manager)
            managed.session.set_clarification_callback(clarification_callback)
            logger.debug(f"Registered clarification callback for session {managed.session_id}")

        # Sync active domains to doc_tools before query execution
        if managed.session.doc_tools and hasattr(managed, 'active_domains'):
            managed.session.doc_tools._active_domain_ids = managed.active_domains or []

        # Slash commands that need the async pipeline (for real-time event delivery)
        # /redo and /reason are handled by follow_up()/solve() directly, not the
        # command registry — skip them here so they fall through.
        stripped = problem.strip()
        _lower = stripped.lower()
        _passthrough_commands = ("/redo", "/reason", "/replay-reason", "/replay")
        if stripped.startswith("/") and not any(_lower.startswith(c) for c in _passthrough_commands):
            from constat.commands.registry import is_command
            if is_command(stripped):
                result = managed.session._handle_slash_command(stripped)
                output = result.get("output", "")
                success = result.get("success", True)
                return {
                    "success": success,
                    "output": output,
                    "final_answer": output,
                    "error": result.get("error") if not success else None,
                }

        # Replay-reason path: re-execute stored inference codes without LLM codegen
        if _lower.startswith("/replay-reason"):
            logger.info(f"Replaying proof code for session {managed.session_id}")
            result = managed.session.replay_proof()
            return _api_result_to_dict_raw(result)

        # Replay path: re-execute stored scratchpad code without LLM codegen
        # Triggered by replay=true in body OR /replay slash command
        if replay or _lower.startswith("/replay"):
            logger.info(f"Replaying stored code for session {managed.session_id}")
            replay_problem = problem
            if _lower.startswith("/replay"):
                # Use stored problem from session
                replay_problem = managed.session.datastore.get_session_meta("problem") if managed.session.datastore else None
                if not replay_problem:
                    return {"success": False, "error": "No previous problem found to replay."}
            result = managed.session.replay(replay_problem, objective_index=objective_index)
            return _api_result_to_dict_raw(result)

        # Auto-detect follow-up: if session already has datastore tables, treat as follow-up
        if not is_followup and managed.session.datastore and managed.session.datastore.list_tables():
            logger.info(f"Auto-detected follow-up for session {managed.session_id} (existing tables found)")
            is_followup = True

        # Downgrade follow-up to solve if no datastore exists (e.g. prior query was
        # conversational/knowledge and never created a datastore)
        if is_followup and not managed.session.datastore:
            logger.info(f"Downgrading follow-up to solve for session {managed.session_id} (no datastore)")
            is_followup = False

        # Run the query via API - use follow_up if explicitly marked or auto-detected
        if is_followup:
            logger.debug(f"Running follow-up query for session {managed.session_id}")
            api_result = managed.api.follow_up(problem)
        else:
            logger.debug(f"Running new query for session {managed.session_id}")
            api_result = managed.api.solve(problem)

        # Convert API result to dict for existing handlers
        return _api_result_to_dict(api_result)
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        return {
            "success": False,
            "error": str(e),
        }


async def _execute_query_async(
    managed: ManagedSession,
    problem: str,
    execution_id: str,
    is_followup: bool = False,
    replay: bool = False,
    objective_index: int | None = None,
    session_manager: "SessionManager | None" = None,
) -> None:
    """Execute query in background and update session state.

    Args:
        managed: The managed session
        problem: Query problem text
        execution_id: Execution tracking ID
        is_followup: Whether this is a follow-up to a previous query
        replay: Whether to replay stored code instead of LLM codegen
        objective_index: If replaying, only replay entries with this objective_index
        session_manager: Session manager for GraphQL subscription pub/sub
    """
    loop = asyncio.get_event_loop()

    def _publish_terminal(payload: dict) -> None:
        if session_manager is not None:
            session_manager.publish_execution_event(managed.session_id, payload)

    try:
        # Run the synchronous solve() in thread pool
        # noinspection PyTypeChecker
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_query(managed, problem, loop, is_followup, replay=replay, objective_index=objective_index, session_manager=session_manager)
        )

        # Queue completion event
        if result.get("success"):
            # Use synthesized final_answer if available, fallback to raw output
            final_output = result.get("final_answer") or result.get("output", "")
            _publish_terminal({
                "event_type": EventType.QUERY_COMPLETE.value,
                "session_id": managed.session_id,
                "step_number": 0,
                "data": {
                    "execution_id": execution_id,
                    "output": final_output,
                    "brief": result.get("brief", False),
                    "tables": result.get("datastore_tables", []),
                    "suggestions": result.get("suggestions", []),
                },
            })
            managed.status = SessionStatus.COMPLETED
        else:
            _publish_terminal({
                "event_type": EventType.QUERY_ERROR.value,
                "session_id": managed.session_id,
                "step_number": 0,
                "data": {
                    "execution_id": execution_id,
                    "error": result.get("error", "Unknown error"),
                },
            })
            managed.status = SessionStatus.ERROR

    except asyncio.CancelledError:
        # Query was cancelled
        _publish_terminal({
            "event_type": EventType.QUERY_CANCELLED.value,
            "session_id": managed.session_id,
            "step_number": 0,
            "data": {"execution_id": execution_id},
        })
        managed.status = SessionStatus.CANCELLED
        raise

    except Exception as e:
        logger.error(f"Async query execution error: {e}")
        _publish_terminal({
            "event_type": EventType.QUERY_ERROR.value,
            "session_id": managed.session_id,
            "step_number": 0,
            "data": {
                "execution_id": execution_id,
                "error": str(e),
            },
        })
        managed.status = SessionStatus.ERROR

    finally:
        managed.current_query = None
        managed.execution_id = None


@router.post("/{session_id}/query", response_model=QueryResponse)
async def submit_query(
    session_id: str,
    body: QueryRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> QueryResponse:
    """Submit a query for execution.

    The query is executed asynchronously. Subscribe to the GraphQL subscription
    to receive real-time progress events.

    Args:
        session_id: Session ID
        body: Query request with problem text
        session_manager: Injected session manager

    Returns:
        Query response with execution ID

    Raises:
        404: Session not found
        400: Session is busy
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    # If session is busy from a previous execution, cancel it and start fresh
    if managed.status in (SessionStatus.PLANNING, SessionStatus.EXECUTING, SessionStatus.AWAITING_APPROVAL):
        logger.info(
            f"Session {session_id} busy ({managed.status.value}), "
            f"cancelling previous execution for new query"
        )
        # Signal cancellation to the session
        if hasattr(managed.session, "_cancelled"):
            managed.session._cancelled = True
        if hasattr(managed.session, "_execution_context"):
            managed.session._execution_context.cancel()
        # Unblock any waiting approval/clarification events
        if managed.approval_event:
            managed.approval_response = {"approved": False, "feedback": "Cancelled by user"}
            managed.approval_event.set()
        if managed.clarification_event:
            managed.clarification_response = {"skip": True}
            managed.clarification_event.set()
        # Reset state
        managed.status = SessionStatus.IDLE
        managed.current_query = None
        managed.execution_id = None

    # Generate execution ID
    execution_id = str(uuid.uuid4())

    # Fast path: slash commands bypass async execution pipeline
    # Run synchronously and emit result directly to event queue
    # Exception: /reason needs the async pipeline for real-time event delivery
    stripped = body.problem.strip()
    _lower = stripped.lower()
    _async_commands = ("/redo", "/reason", "/replay")
    if stripped.startswith("/") and not any(_lower.startswith(c) for c in _async_commands):
        try:
            from constat.commands.registry import is_command
            if is_command(stripped):
                # Sync active domains to doc_tools before slash command execution
                if managed.session.doc_tools and hasattr(managed, 'active_domains'):
                    managed.session.doc_tools._active_domain_ids = managed.active_domains or []
                result = managed.session._handle_slash_command(stripped)
                output = result.get("output", "")
                status = "completed" if result.get("success", True) else "error"
                return QueryResponse(
                    execution_id=execution_id,
                    status=status,
                    message=output or (result.get("error", "Command failed") if status == "error" else "Command executed."),
                )
        except Exception as e:
            logger.warning(f"Slash command fast path failed, falling back to async: {e}")

    # Update session state (reset cancellation flag for fresh execution)
    if hasattr(managed.session, "_cancelled"):
        managed.session._cancelled = False
    managed.current_query = body.problem
    managed.execution_id = execution_id
    managed.status = SessionStatus.PLANNING

    # Override approval setting if client explicitly requested it
    if body.require_approval is not None:
        managed.session.session_config.require_approval = body.require_approval
        managed.session.session_config.auto_approve = not body.require_approval
        managed.session.session_config.force_approval = body.require_approval

    # Register event handler if not already registered
    handler = _create_event_handler(managed, session_manager)
    # Remove any existing handlers first to avoid duplicates
    managed.session._event_handlers = []
    managed.session.on_event(handler)

    # Start background execution using asyncio.create_task for better shutdown control
    task = asyncio.create_task(
        _execute_query_async(managed, body.problem, execution_id, body.is_followup, replay=body.replay, objective_index=body.objective_index, session_manager=session_manager)
    )
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)

    return QueryResponse(
        execution_id=execution_id,
        status="started",
        message="Query execution started. Subscribe to queryExecution for progress updates.",
    )


@router.post("/{session_id}/cancel")
async def cancel_execution(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Cancel the current execution.

    Attempts to cancel any running query execution.

    Args:
        session_id: Session ID
        session_manager: Injected session manager

    Returns:
        Cancellation status

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    # Signal cancellation to the session
    if hasattr(managed.session, "_cancelled"):
        managed.session._cancelled = True

    # Also signal to execution context if present
    if hasattr(managed.session, "_execution_context"):
        managed.session._execution_context.cancel()

    managed.status = SessionStatus.CANCELLED

    return {
        "status": "cancelling",
        "message": "Cancellation requested",
    }


@router.get("/{session_id}/plan", response_model=PlanResponse)
async def get_plan(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> PlanResponse:
    """Get the current execution plan.

    Args:
        session_id: Session ID
        session_manager: Injected session manager

    Returns:
        Current plan details

    Raises:
        404: Session not found or no plan exists
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    if not managed.session.plan:
        raise HTTPException(status_code=404, detail="No plan exists for this session")

    return _plan_to_response(managed.session.plan)


@router.post("/{session_id}/plan/approve", response_model=ApprovalResponse)
async def approve_plan(
    session_id: str,
    body: ApprovalRequest,
    session_manager: SessionManager = Depends(get_session_manager),
) -> ApprovalResponse:
    """Approve or reject the current plan.

    This endpoint is used when require_plan_approval is enabled.
    Call this after receiving the plan_ready event via GraphQL subscription.

    Args:
        session_id: Session ID
        body: Approval request with decision
        session_manager: Injected session manager

    Returns:
        Approval status

    Raises:
        404: Session not found
        400: Session not awaiting approval
    """
    managed = session_manager.get_session_or_none(session_id)
    if not managed:
        raise HTTPException(status_code=404, detail="Session not found")

    if managed.status != SessionStatus.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Session is not awaiting approval (status: {managed.status.value})",
        )

    # Store approval response and signal the waiting coroutine
    managed.approval_response = {
        "approved": body.approved,
        "feedback": body.feedback,
        "deleted_steps": body.deleted_steps,
        "edited_steps": [{"number": s.number, "goal": s.goal} for s in body.edited_steps] if body.edited_steps else None,
    }

    if managed.approval_event:
        managed.approval_event.set()

    if body.approved:
        managed.status = SessionStatus.EXECUTING
        deleted_count = len(body.deleted_steps) if body.deleted_steps else 0
        message = "Plan approved, execution continuing"
        if deleted_count > 0:
            message += f" ({deleted_count} step(s) removed)"
        return ApprovalResponse(status="approved", message=message)
    else:
        managed.status = SessionStatus.IDLE
        return ApprovalResponse(status="rejected", message=f"Plan rejected: {body.feedback}")

