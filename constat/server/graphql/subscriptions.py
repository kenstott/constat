# Copyright (c) 2025 Kenneth Stott
# Canary: 29a27617-f0f7-4c3e-a09e-d775201b6b23
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL subscription resolvers for glossary and execution events."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator

import strawberry
from constat.server.graphql.session_context import GqlInfo as Info

from constat.server.graphql.types import ExecutionEventType, GlossaryChangeEvent

logger = logging.getLogger(__name__)


@strawberry.type
class GlossarySubscription:
    @strawberry.subscription
    async def glossary_changed(
        self, info: Info, session_id: str
    ) -> AsyncGenerator[GlossaryChangeEvent, None]:
        sm = info.context.session_manager
        queue = sm.subscribe_glossary(session_id)
        try:
            while True:
                event: GlossaryChangeEvent = await queue.get()
                yield event
        finally:
            sm.unsubscribe_glossary(session_id, queue)


# Backwards-compat alias: __init__.py previously imported `Subscription`
Subscription = GlossarySubscription


@strawberry.type
class ExecutionSubscription:
    @strawberry.subscription
    async def query_execution(
        self, info: Info, session_id: str
    ) -> AsyncGenerator[ExecutionEventType, None]:
        sm = info.context.session_manager

        # Wait for session (background init may still be running)
        managed = None
        for _ in range(30):
            managed = sm.get_session_or_none(session_id)
            if managed:
                break
            await asyncio.sleep(0.5)
        if not managed:
            raise ValueError(f"Session {session_id} not found")

        # Send welcome event
        from constat.messages import WelcomeMessage
        welcome = WelcomeMessage.create()
        yield ExecutionEventType(
            event_type="welcome",
            session_id=session_id,
            step_number=0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data={
                "reliable_adjective": welcome.reliable_adjective,
                "honest_adjective": welcome.honest_adjective,
                "tagline": welcome.tagline,
                "suggestions": welcome.suggestions,
                "message_markdown": welcome.to_markdown(),
            },
        )

        # Replay session_ready if it fired before subscription connected
        if managed._session_ready_event:
            evt = managed._session_ready_event
            yield ExecutionEventType(
                event_type=evt.get("event_type", ""),
                session_id=evt.get("session_id", session_id),
                step_number=evt.get("step_number", 0),
                timestamp=evt.get("timestamp", datetime.now(timezone.utc).isoformat()),
                data=evt.get("data", {}),
            )

        # Replay cached entity rebuild event if it fired before subscription connected
        if managed._entity_rebuild_event:
            evt = managed._entity_rebuild_event
            yield ExecutionEventType(
                event_type=evt.get("event_type", ""),
                session_id=evt.get("session_id", session_id),
                step_number=evt.get("step_number", 0),
                timestamp=evt.get("timestamp", datetime.now(timezone.utc).isoformat()),
                data=evt.get("data", {}),
            )

        # Replay glossary generation status
        if managed._glossary_generating:
            yield ExecutionEventType(
                event_type="glossary_rebuild_start",
                session_id=session_id,
                step_number=0,
                timestamp=datetime.now(timezone.utc).isoformat(),
                data={"session_id": session_id},
            )

        queue = sm.subscribe_execution(session_id)
        try:
            while True:
                event = await queue.get()
                yield ExecutionEventType(
                    event_type=event.get("event_type", ""),
                    session_id=event.get("session_id", session_id),
                    step_number=event.get("step_number", 0),
                    timestamp=str(event.get("timestamp", datetime.now(timezone.utc).isoformat())),
                    data=event.get("data") or {},
                )
        finally:
            sm.unsubscribe_execution(session_id, queue)
