# Copyright (c) 2025 Kenneth Stott
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

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

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

# Track active WebSocket connections for graceful shutdown
_active_websockets: set[WebSocket] = set()


async def shutdown_executor_async() -> None:
    """Shutdown the thread pool executor and cancel active tasks. Called during app shutdown."""
    # Close all active WebSocket connections first
    websockets_to_close = list(_active_websockets)
    for ws in websockets_to_close:
        try:
            await ws.close(code=1001, reason="Server shutting down")
        except (RuntimeError, ConnectionError, OSError):
            pass
    _active_websockets.clear()

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


def _create_approval_callback(managed: ManagedSession, loop: asyncio.AbstractEventLoop):
    """Create an approval callback that bridges sync session to async WebSocket.

    This callback is called from the synchronous session.solve() thread when
    plan approval is required. It:
    1. Updates status to awaiting_approval
    2. Emits plan_ready event via the event queue
    3. Blocks waiting for approval_event to be set (by WebSocket handler)
    4. Returns the approval response
    """
    def approval_callback(request: PlanApprovalRequest) -> PlanApprovalResponse:
        logger.debug(f"Approval callback called for session {managed.session_id}")

        # Update status
        managed.status = SessionStatus.AWAITING_APPROVAL

        # Create approval event for waiting
        managed.approval_event = asyncio.Event()
        managed.approval_response = None

        # Queue plan_ready event for WebSocket
        plan_data = {
            "problem": request.problem,
            "steps": request.steps,
            "reasoning": request.reasoning,
        }
        try:
            managed.event_queue.put_nowait({
                "event_type": EventType.PLAN_READY.value,
                "session_id": managed.session_id,
                "step_number": 0,
                "data": {"plan": plan_data},
            })
        except asyncio.QueueFull:
            logger.warning(f"Event queue full for session {managed.session_id}")

        # Wait for approval (blocking in thread, but event is set from async context)
        # Use run_coroutine_threadsafe to wait on the async event from sync code
        async def wait_for_approval():
            await managed.approval_event.wait()
            return managed.approval_response

        future = asyncio.run_coroutine_threadsafe(wait_for_approval(), loop)
        try:
            response = future.result(timeout=600)  # 10 minute timeout
        except Exception as e:
            logger.error(f"Error waiting for approval: {type(e).__name__}: {e}")
            return PlanApprovalResponse.approve()  # Default to approve on error

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


def _create_clarification_callback(managed: ManagedSession, loop: asyncio.AbstractEventLoop):
    """Create a clarification callback that bridges sync session to async WebSocket.

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

        # Send all questions at once for stepper UI
        clarification_data = {
            "original_question": request.original_question,
            "ambiguity_reason": request.ambiguity_reason,
            "questions": [
                {"text": q.text, "suggestions": q.suggestions}
                for q in request.questions
            ],
        }
        try:
            managed.event_queue.put_nowait({
                "event_type": EventType.CLARIFICATION_NEEDED.value,
                "session_id": managed.session_id,
                "step_number": 0,
                "data": clarification_data,
            })
        except asyncio.QueueFull:
            logger.warning(f"Event queue full for session {managed.session_id}")

        # Wait for all answers
        async def wait_for_clarification():
            await managed.clarification_event.wait()
            return managed.clarification_response

        future = asyncio.run_coroutine_threadsafe(wait_for_clarification(), loop)
        try:
            response = future.result(timeout=600)  # 10 minute timeout for all questions
        except Exception as e:
            logger.error(f"Error waiting for clarification: {e}")
            return ClarificationResponse(answers={}, skip=True)

        if response is None or response.get("skip"):
            return ClarificationResponse(answers={}, skip=True)

        managed.status = SessionStatus.PLANNING
        return ClarificationResponse(answers=response.get("answers", {}), skip=False)

    return clarification_callback


def _create_event_handler(managed: ManagedSession):
    """Create an event handler that queues events for WebSocket delivery."""

    def handle_event(event: StepEvent) -> None:
        """Handle session events by queuing them."""
        try:
            # Log dynamic_context events for debugging
            if event.event_type == "dynamic_context":
                logger.info(f"[EVENT_HANDLER] Received dynamic_context event: {event.data}")

            # Map session event types to API event types
            event_type_map = {
                "step_start": EventType.STEP_START,
                "generating": EventType.STEP_GENERATING,
                "executing": EventType.STEP_EXECUTING,
                "step_complete": EventType.STEP_COMPLETE,
                "step_error": EventType.STEP_ERROR,
                "step_failed": EventType.STEP_FAILED,
                "facts_extracted": EventType.FACTS_EXTRACTED,
                "progress": EventType.PROGRESS,
                "dynamic_context": EventType.DYNAMIC_CONTEXT,
                "planning_start": EventType.PLANNING_START,
                "proof_start": EventType.PROOF_START,
                "replanning": EventType.REPLANNING,
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
                "proof_complete": EventType.PROOF_COMPLETE,
                "proof_summary_ready": EventType.PROOF_SUMMARY_READY,
            }

            api_event_type = event_type_map.get(event.event_type, EventType.PROGRESS)

            # Sanitize event data — replace non-serializable types
            safe_data = event.data
            if safe_data:
                import pandas as pd
                sanitized = {}
                for k, v in safe_data.items():
                    if isinstance(v, pd.DataFrame):
                        sanitized[k] = f"DataFrame({len(v)} rows, {len(v.columns)} cols)"
                    else:
                        sanitized[k] = v
                safe_data = sanitized

            ws_event = StepEventWS(
                event_type=api_event_type,
                session_id=managed.session_id,
                step_number=event.step_number,
                data=safe_data,
            )

            # Put event on queue (non-blocking)
            try:
                managed.event_queue.put_nowait(ws_event.model_dump(mode="json"))
            except asyncio.QueueFull:
                logger.warning(f"Event queue full for session {managed.session_id}")

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    return handle_event


def _run_query(managed: ManagedSession, problem: str, loop: asyncio.AbstractEventLoop, is_followup: bool = False) -> dict[str, Any]:
    """Run a query synchronously (called from thread pool).

    Uses the ConstatAPI for solve/follow_up operations, ensuring
    consistent behavior with the REPL.

    Args:
        managed: The managed session
        problem: Query problem text
        loop: Event loop for async bridge
        is_followup: Whether this is a follow-up to a previous query

    Returns:
        Query result dict
    """
    try:
        # Register approval callback if plan approval is required
        if managed.session.session_config.require_approval:
            approval_callback = _create_approval_callback(managed, loop)
            managed.api.set_approval_callback(approval_callback)
            logger.debug(f"Registered approval callback for session {managed.session_id}")

        # Register clarification callback if clarifications are enabled
        if managed.session.session_config.ask_clarifications:
            clarification_callback = _create_clarification_callback(managed, loop)
            managed.session.set_clarification_callback(clarification_callback)
            logger.debug(f"Registered clarification callback for session {managed.session_id}")

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
) -> None:
    """Execute query in background and update session state.

    Args:
        managed: The managed session
        problem: Query problem text
        execution_id: Execution tracking ID
        is_followup: Whether this is a follow-up to a previous query
    """
    loop = asyncio.get_event_loop()

    try:
        # Run the synchronous solve() in thread pool
        # noinspection PyTypeChecker
        result = await loop.run_in_executor(
            _executor,
            lambda: _run_query(managed, problem, loop, is_followup)
        )

        # Queue completion event
        if result.get("success"):
            # Use synthesized final_answer if available, fallback to raw output
            final_output = result.get("final_answer") or result.get("output", "")
            managed.event_queue.put_nowait({
                "event_type": EventType.QUERY_COMPLETE.value,
                "session_id": managed.session_id,
                "step_number": 0,
                "data": {
                    "execution_id": execution_id,
                    "output": final_output,
                    "tables": result.get("datastore_tables", []),
                    "suggestions": result.get("suggestions", []),
                },
            })
            managed.status = SessionStatus.COMPLETED
        else:
            managed.event_queue.put_nowait({
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
        managed.event_queue.put_nowait({
            "event_type": EventType.QUERY_CANCELLED.value,
            "session_id": managed.session_id,
            "step_number": 0,
            "data": {"execution_id": execution_id},
        })
        managed.status = SessionStatus.CANCELLED
        raise

    except Exception as e:
        logger.error(f"Async query execution error: {e}")
        managed.event_queue.put_nowait({
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

    The query is executed asynchronously. Subscribe to the WebSocket
    endpoint to receive real-time progress events.

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
    managed = session_manager.get_session(session_id)

    # Check if session is busy
    if managed.status in (SessionStatus.PLANNING, SessionStatus.EXECUTING, SessionStatus.AWAITING_APPROVAL):
        raise HTTPException(
            status_code=400,
            detail=f"Session is busy (status: {managed.status.value})",
        )

    # Generate execution ID
    execution_id = str(uuid.uuid4())

    # Fast path: slash commands bypass async execution pipeline
    # Run synchronously and emit result directly to event queue
    stripped = body.problem.strip()
    if stripped.startswith("/") and not stripped.lower().startswith("/redo"):
        try:
            from constat.commands.registry import is_command
            if is_command(stripped):
                # Sync active domains to doc_tools before slash command execution
                if managed.session.doc_tools and hasattr(managed, 'active_domains'):
                    managed.session.doc_tools._active_domain_ids = managed.active_domains or []
                result = managed.session._handle_slash_command(stripped)
                output = result.get("output", "")
                if result.get("success", True):
                    managed.event_queue.put_nowait({
                        "event_type": EventType.QUERY_COMPLETE.value,
                        "session_id": managed.session_id,
                        "step_number": 0,
                        "data": {
                            "execution_id": execution_id,
                            "output": output,
                            "tables": [],
                            "suggestions": result.get("suggestions", []),
                        },
                    })
                else:
                    managed.event_queue.put_nowait({
                        "event_type": EventType.QUERY_ERROR.value,
                        "session_id": managed.session_id,
                        "step_number": 0,
                        "data": {
                            "execution_id": execution_id,
                            "error": output or result.get("error", "Command failed"),
                        },
                    })
                return QueryResponse(
                    execution_id=execution_id,
                    status="completed",
                    message=output[:200] if output else "Command executed.",
                )
        except Exception as e:
            logger.warning(f"Slash command fast path failed, falling back to async: {e}")

    # Update session state
    managed.current_query = body.problem
    managed.execution_id = execution_id
    managed.status = SessionStatus.PLANNING

    # Register event handler if not already registered
    handler = _create_event_handler(managed)
    # Remove any existing handlers first to avoid duplicates
    managed.session._event_handlers = []
    managed.session.on_event(handler)

    # Start background execution using asyncio.create_task for better shutdown control
    task = asyncio.create_task(
        _execute_query_async(managed, body.problem, execution_id, body.is_followup)
    )
    _active_tasks.add(task)
    task.add_done_callback(_active_tasks.discard)

    return QueryResponse(
        execution_id=execution_id,
        status="started",
        message="Query execution started. Connect to WebSocket for progress updates.",
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
    managed = session_manager.get_session(session_id)

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
    managed = session_manager.get_session(session_id)

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
    Call this after receiving the plan_ready event via WebSocket.

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
    managed = session_manager.get_session(session_id)

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


@router.websocket("/{session_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
):
    """WebSocket endpoint for real-time event streaming.

    Connect to this endpoint to receive real-time progress events
    during query execution. Events include:
    - planning_start: Planning has begun
    - plan_ready: Plan is ready (approval may be required)
    - step_start: Step execution starting
    - step_generating: Generating code for step
    - step_executing: Executing step code
    - step_complete: Step completed successfully
    - step_error: Step encountered an error
    - query_complete: Query execution completed
    - query_error: Query execution failed

    Commands can be sent via the WebSocket:
    - {"action": "approve"}: Approve the current plan
    - {"action": "reject", "data": {"feedback": "reason"}}: Reject plan
    - {"action": "cancel"}: Cancel execution
    - {"action": "clarify", "data": {"answer": "user answer"}}: Answer a clarification question
    - {"action": "skip_clarification"}: Skip clarification and proceed
    """
    await websocket.accept()
    _active_websockets.add(websocket)

    # Get session manager from app state
    session_manager: SessionManager = websocket.app.state.session_manager

    try:
        managed = session_manager.get_session(session_id)
    except KeyError:
        _active_websockets.discard(websocket)
        await websocket.close(code=4404, reason="Session not found")
        return

    # Reset stuck session status on reconnect (e.g., browser refresh mid-operation)
    if managed.status in (SessionStatus.PLANNING, SessionStatus.EXECUTING, SessionStatus.AWAITING_APPROVAL):
        logger.info(f"Resetting stuck session {session_id} from {managed.status.value} to idle")
        managed.status = SessionStatus.IDLE
        managed.current_query = None
        managed.execution_id = None
        managed.approval_event = None
        managed.approval_response = None
        # Clear stale events from previous execution to prevent duplicates
        while not managed.event_queue.empty():
            try:
                managed.event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # Send welcome message on connection
    welcome = WelcomeMessage.create()
    await websocket.send_json({
        "type": "event",
        "payload": {
            "event_type": "welcome",
            "session_id": session_id,
            "step_number": 0,
            "data": {
                "reliable_adjective": welcome.reliable_adjective,
                "honest_adjective": welcome.honest_adjective,
                "tagline": welcome.tagline,
                "suggestions": welcome.suggestions,
                "message_markdown": welcome.to_markdown(),
            },
        },
    })

    try:
        # Create tasks for sending events and receiving commands
        async def send_events():
            """Send events from queue to WebSocket."""
            while True:
                try:
                    event = await managed.event_queue.get()
                    await websocket.send_json({
                        "type": "event",
                        "payload": event,
                    })
                except asyncio.CancelledError:
                    break
                except Exception as send_err:
                    logger.error(f"Error sending event: {send_err}")
                    break

        async def receive_commands():
            """Receive and process commands from WebSocket."""
            while True:
                try:
                    data = await websocket.receive_json()
                    action = data.get("action")

                    if action == "approve":
                        managed.approval_response = {"approved": True, "feedback": None}
                        if managed.approval_event:
                            managed.approval_event.set()
                        await websocket.send_json({
                            "type": "ack",
                            "payload": {"action": "approve", "status": "ok"},
                        })

                    elif action == "reject":
                        feedback = data.get("data", {}).get("feedback", "Rejected by user")
                        managed.approval_response = {"approved": False, "feedback": feedback}
                        if managed.approval_event:
                            managed.approval_event.set()
                        await websocket.send_json({
                            "type": "ack",
                            "payload": {"action": "reject", "status": "ok"},
                        })

                    elif action == "cancel":
                        if hasattr(managed.session, "_cancelled"):
                            managed.session._cancelled = True
                        if hasattr(managed.session, "_execution_context"):
                            managed.session._execution_context.cancel()
                        await websocket.send_json({
                            "type": "ack",
                            "payload": {"action": "cancel", "status": "ok"},
                        })

                    elif action == "clarify":
                        # User answered all clarification questions
                        answers = data.get("data", {}).get("answers", {})
                        managed.clarification_response = {"answers": answers, "skip": False}
                        if managed.clarification_event:
                            managed.clarification_event.set()
                        await websocket.send_json({
                            "type": "ack",
                            "payload": {"action": "clarify", "status": "ok"},
                        })

                    elif action == "skip_clarification":
                        # User wants to skip clarification
                        managed.clarification_response = {"skip": True}
                        if managed.clarification_event:
                            managed.clarification_event.set()
                        await websocket.send_json({
                            "type": "ack",
                            "payload": {"action": "skip_clarification", "status": "ok"},
                        })

                    elif action == "autocomplete":
                        # Handle autocomplete requests for tables, columns, entities
                        ac_data = data.get("data", {})
                        context = ac_data.get("context")
                        prefix = ac_data.get("prefix", "").lower()
                        parent = ac_data.get("parent")
                        request_id = ac_data.get("request_id")

                        items = []
                        try:
                            if context == "table":
                                # Get tables from datastore
                                tables = managed.session.datastore.list_tables()
                                for t in tables:
                                    name = t.get("name", "")
                                    if name.lower().startswith(prefix):
                                        items.append({
                                            "label": name,
                                            "value": name,
                                            "description": t.get("description") or f"{t.get('row_count', 0)} rows",
                                        })

                            elif context == "column" and parent:
                                # Get columns for a specific table
                                schema = managed.session.datastore.get_table_schema(parent)
                                if schema:
                                    for col in schema:
                                        name = col.get("name", "")
                                        if name.lower().startswith(prefix):
                                            items.append({
                                                "label": name,
                                                "value": name,
                                                "description": col.get("type", ""),
                                            })

                            elif context == "entity":
                                # Get entity names from schema manager
                                if hasattr(managed.session, "schema_manager") and managed.session.schema_manager:
                                    entities = managed.session.schema_manager.get_schema_entities(include_columns=False)
                                    for entity in entities:
                                        if entity.lower().startswith(prefix):
                                            items.append({
                                                "label": entity,
                                                "value": entity,
                                                "description": "Table",
                                            })
                        except Exception as ac_err:
                            logger.warning(f"Autocomplete error: {ac_err}")

                        # Send autocomplete response
                        await websocket.send_json({
                            "type": "event",
                            "payload": {
                                "event_type": "autocomplete_response",
                                "session_id": session_id,
                                "step_number": 0,
                                "data": {
                                    "request_id": request_id,
                                    "items": items[:20],  # Limit to 20 items
                                },
                            },
                        })

                    else:
                        await websocket.send_json({
                            "type": "error",
                            "payload": {"message": f"Unknown action: {action}"},
                        })

                except asyncio.CancelledError:
                    break
                except WebSocketDisconnect:
                    break
                except Exception as recv_err:
                    logger.error(f"Error receiving command: {recv_err}")
                    break

        # Run both tasks concurrently
        send_task = asyncio.create_task(send_events())
        receive_task = asyncio.create_task(receive_commands())

        try:
            await asyncio.gather(send_task, receive_task)
        except asyncio.CancelledError:
            pass
        finally:
            send_task.cancel()
            receive_task.cancel()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        _active_websockets.discard(websocket)
        try:
            await websocket.close()
        except (RuntimeError, ConnectionError, OSError):
            pass  # Already closed
