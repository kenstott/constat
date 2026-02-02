# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""WebSocket handler for real-time event streaming.

This module provides utilities for managing WebSocket connections
and bridging synchronous Session events to asynchronous WebSocket delivery.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from constat.server.models import EventType, StepEventWS
from constat.session import StepEvent

logger = logging.getLogger(__name__)


@dataclass
class SessionWebSocket:
    """Manages WebSocket connection for a session.

    Bridges synchronous Session events to async WebSocket delivery
    using an asyncio Queue.

    Attributes:
        session_id: ID of the associated session
        event_queue: Queue for pending events to send
        connected: Whether WebSocket is connected
        approval_event: Event for signaling approval response
        approval_response: Response from approval request
    """

    session_id: str
    event_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1000))
    connected: bool = False
    approval_event: Optional[asyncio.Event] = None
    approval_response: Optional[dict] = None

    # Event type mapping from Session events to API events
    _EVENT_TYPE_MAP: dict[str, EventType] = field(default_factory=lambda: {
        "step_start": EventType.STEP_START,
        "generating": EventType.STEP_GENERATING,
        "executing": EventType.STEP_EXECUTING,
        "step_complete": EventType.STEP_COMPLETE,
        "step_error": EventType.STEP_ERROR,
        "step_failed": EventType.STEP_FAILED,
        "facts_extracted": EventType.FACTS_EXTRACTED,
        "fact_resolved": EventType.FACT_RESOLVED,
        "fact_start": EventType.FACT_START,
        "fact_planning": EventType.FACT_PLANNING,
        "fact_executing": EventType.FACT_EXECUTING,
        "fact_failed": EventType.FACT_FAILED,
        "proof_complete": EventType.PROOF_COMPLETE,
        "progress": EventType.PROGRESS,
        "planning_start": EventType.PLANNING_START,
        "table_created": EventType.TABLE_CREATED,
        "artifact_created": EventType.ARTIFACT_CREATED,
    })

    def handle_event(self, event: StepEvent) -> None:
        """Handle a Session event by queuing it for WebSocket delivery.

        This method is called synchronously from the Session thread
        and queues events for async delivery.

        Args:
            event: The StepEvent from Session
        """
        try:
            # Map session event type to API event type
            api_event_type = self._EVENT_TYPE_MAP.get(
                event.event_type,
                EventType.PROGRESS
            )

            ws_event = StepEventWS(
                event_type=api_event_type,
                session_id=self.session_id,
                step_number=event.step_number,
                timestamp=datetime.now(timezone.utc),
                data=event.data,
            )

            # Non-blocking put - drop if queue full
            try:
                self.event_queue.put_nowait(ws_event.model_dump(mode="json"))
            except asyncio.QueueFull:
                logger.warning(
                    f"Event queue full for session {self.session_id}, "
                    f"dropping event: {event.event_type}"
                )

        except Exception as e:
            logger.error(f"Error handling event for session {self.session_id}: {e}")

    def create_event_handler(self) -> Callable[[StepEvent], None]:
        """Create an event handler function for Session.on_event().

        Returns:
            A callable that can be registered with Session.on_event()
        """
        return self.handle_event

    async def wait_for_approval(self, timeout: float = 300.0) -> Optional[dict]:
        """Wait for approval response from WebSocket client.

        Args:
            timeout: Maximum seconds to wait for response

        Returns:
            Approval response dict or None if timed out
        """
        self.approval_event = asyncio.Event()
        self.approval_response = None

        try:
            await asyncio.wait_for(
                self.approval_event.wait(),
                timeout=timeout,
            )
            return self.approval_response
        except asyncio.TimeoutError:
            logger.warning(f"Approval timeout for session {self.session_id}")
            return None
        finally:
            self.approval_event = None

    def set_approval_response(self, approved: bool, feedback: Optional[str] = None) -> None:
        """Set the approval response and signal the waiting coroutine.

        Args:
            approved: Whether the plan was approved
            feedback: Optional feedback if rejected
        """
        self.approval_response = {
            "approved": approved,
            "feedback": feedback,
        }
        if self.approval_event:
            self.approval_event.set()

    async def get_next_event(self, timeout: Optional[float] = None) -> Optional[dict]:
        """Get the next event from the queue.

        Args:
            timeout: Maximum seconds to wait, or None to wait forever

        Returns:
            Event dict or None if timed out
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=timeout,
                )
            return await self.event_queue.get()
        except asyncio.TimeoutError:
            return None

    def queue_event(self, event_type: EventType, data: dict[str, Any]) -> None:
        """Queue a custom event for WebSocket delivery.

        Args:
            event_type: Type of event
            data: Event data
        """
        try:
            ws_event = StepEventWS(
                event_type=event_type,
                session_id=self.session_id,
                step_number=0,
                timestamp=datetime.now(timezone.utc),
                data=data,
            )
            self.event_queue.put_nowait(ws_event.model_dump(mode="json"))
        except asyncio.QueueFull:
            logger.warning(f"Event queue full for session {self.session_id}")

    def clear_queue(self) -> int:
        """Clear all pending events from the queue.

        Returns:
            Number of events cleared
        """
        count = 0
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count


class WebSocketManager:
    """Manages WebSocket connections across sessions.

    Provides centralized management of SessionWebSocket instances
    and utility methods for broadcasting events.
    """

    def __init__(self):
        """Initialize the WebSocket manager."""
        self._websockets: dict[str, SessionWebSocket] = {}

    def get_or_create(self, session_id: str) -> SessionWebSocket:
        """Get or create a SessionWebSocket for a session.

        Args:
            session_id: Session ID

        Returns:
            SessionWebSocket instance
        """
        if session_id not in self._websockets:
            self._websockets[session_id] = SessionWebSocket(session_id=session_id)
        return self._websockets[session_id]

    def get(self, session_id: str) -> Optional[SessionWebSocket]:
        """Get SessionWebSocket for a session if it exists.

        Args:
            session_id: Session ID

        Returns:
            SessionWebSocket instance or None
        """
        return self._websockets.get(session_id)

    def remove(self, session_id: str) -> bool:
        """Remove a SessionWebSocket.

        Args:
            session_id: Session ID

        Returns:
            True if removed, False if not found
        """
        if session_id in self._websockets:
            del self._websockets[session_id]
            return True
        return False

    def broadcast(self, event_type: EventType, data: dict[str, Any]) -> int:
        """Broadcast an event to all connected sessions.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            Number of sessions event was queued to
        """
        count = 0
        for ws in self._websockets.values():
            if ws.connected:
                ws.queue_event(event_type, data)
                count += 1
        return count

    def get_connected_count(self) -> int:
        """Get number of connected WebSockets.

        Returns:
            Count of connected sessions
        """
        return sum(1 for ws in self._websockets.values() if ws.connected)

    def get_stats(self) -> dict[str, Any]:
        """Get WebSocket manager statistics.

        Returns:
            Dict with connection stats
        """
        return {
            "total_sessions": len(self._websockets),
            "connected": self.get_connected_count(),
            "queue_sizes": {
                sid: ws.event_queue.qsize()
                for sid, ws in self._websockets.items()
            },
        }
