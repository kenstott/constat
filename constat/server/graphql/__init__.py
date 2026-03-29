# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL schema and router for the Constat API."""

from __future__ import annotations

import strawberry
from fastapi import Request, WebSocket
from strawberry.fastapi import GraphQLRouter

from constat.server.graphql.resolvers import Mutation, Query
from constat.server.graphql.subscriptions import Subscription


schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)


async def get_context(request: Request = None, ws: WebSocket = None) -> dict:
    conn = request or ws
    return {"session_manager": conn.app.state.session_manager}


graphql_router = GraphQLRouter(schema, context_getter=get_context)
