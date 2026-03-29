# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL subscription resolvers for glossary change events."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator

import strawberry
from strawberry.types import Info

from constat.server.graphql.types import GlossaryChangeEvent

logger = logging.getLogger(__name__)


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def glossary_changed(
        self, info: Info, session_id: str
    ) -> AsyncGenerator[GlossaryChangeEvent, None]:
        sm = info.context["session_manager"]
        queue = sm.subscribe_glossary(session_id)
        try:
            while True:
                event: GlossaryChangeEvent = await queue.get()
                yield event
        finally:
            sm.unsubscribe_glossary(session_id, queue)
