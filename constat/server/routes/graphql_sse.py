# Copyright (c) 2025 Kenneth Stott
# Canary: 331bcac1-97fe-48fd-acfd-63f313977edd
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""SSE transport for GraphQL subscriptions.

Enables clients (e.g. Jupyter notebooks) that cannot use WebSockets
to consume GraphQL subscriptions over plain HTTP Server-Sent Events.

    GET /api/graphql/stream?query=subscription{...}&variables={}
    Accept: text/event-stream
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse

from constat.server.auth import authenticate_token
from constat.server.graphql import schema
from constat.server.graphql.session_context import GraphQLContext

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stream")
async def graphql_sse(request: Request, query: str, variables: str = "{}"):
    """Execute a GraphQL subscription and stream results as SSE."""
    parsed_vars = json.loads(variables)

    # Build context identical to the regular GraphQL endpoint
    auth_header = request.headers.get("authorization", "")
    token = auth_header[7:] if auth_header.startswith("Bearer ") else None
    server_config = request.app.state.server_config
    config = getattr(request.app.state, "config", None)
    try:
        user_id = authenticate_token(token, server_config)
    except Exception:
        user_id = None

    ctx = GraphQLContext(
        session_manager=request.app.state.session_manager,
        server_config=server_config,
        user_id=user_id,
        config=config,
    )

    async def event_generator():
        result = await schema.subscribe(query, variable_values=parsed_vars, context_value=ctx)
        try:
            async for item in result:
                if await request.is_disconnected():
                    break
                data = {"data": item.data} if hasattr(item, "data") else {"data": item}
                if hasattr(item, "errors") and item.errors:
                    data["errors"] = [{"message": str(e)} for e in item.errors]
                yield f"event: next\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.error("SSE subscription error: %s", exc)
            yield f"event: next\ndata: {json.dumps({'errors': [{'message': str(exc)}]})}\n\n"
        yield "event: complete\ndata:\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
