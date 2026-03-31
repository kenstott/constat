# Copyright (c) 2025 Kenneth Stott
# Canary: 06fd8a1c-db0f-471b-8cfc-951621f41957
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL schema and router for the Constat API.

Schema stitching: each domain contributes its own Query/Mutation/Subscription
fragment.  New domains add their types here via ``strawberry.merge_types``.

Current domains:
  - glossary  (resolvers.py)  — Query, Mutation
  - subscriptions (subscriptions.py) — Subscription (glossaryChanged)

To add a new domain:
  1. Create ``constat/server/graphql/<domain>.py`` with @strawberry.type classes
  2. Import them below and add to the appropriate merge_types call
"""

from __future__ import annotations

import logging

import strawberry
from fastapi import Request, WebSocket
from strawberry.fastapi import GraphQLRouter
from strawberry.tools import merge_types

from constat.server.auth import authenticate_token
from constat.server.graphql.session_context import GraphQLContext

# -- Domain resolver imports (one per file) ------------------------------------
from constat.server.graphql.auth_resolvers import Mutation as AuthMutation
from constat.server.graphql.auth_resolvers import Query as AuthQuery
from constat.server.graphql.resolvers import Mutation as GlossaryMutation
from constat.server.graphql.resolvers import Query as GlossaryQuery
from constat.server.graphql.session_resolvers import Mutation as SessionMutation
from constat.server.graphql.session_resolvers import Query as SessionQuery
from constat.server.graphql.state_resolvers import Mutation as StateMutation
from constat.server.graphql.state_resolvers import Query as StateQuery
from constat.server.graphql.data_resolvers import Mutation as DataMutation
from constat.server.graphql.data_resolvers import Query as DataQuery
from constat.server.graphql.source_resolvers import Mutation as SourceMutation
from constat.server.graphql.source_resolvers import Query as SourceQuery
from constat.server.graphql.domain_resolvers import Mutation as DomainMutation
from constat.server.graphql.domain_resolvers import Query as DomainQuery
from constat.server.graphql.execution_resolvers import Mutation as ExecutionMutation
from constat.server.graphql.execution_resolvers import Query as ExecutionQuery
from constat.server.graphql.learning_resolvers import Mutation as LearningMutation
from constat.server.graphql.learning_resolvers import Query as LearningQuery
from constat.server.graphql.fine_tune_resolvers import Mutation as FineTuneMutation
from constat.server.graphql.fine_tune_resolvers import Query as FineTuneQuery
from constat.server.graphql.feedback_resolvers import Mutation as FeedbackMutation
from constat.server.graphql.feedback_resolvers import Query as FeedbackQuery
from constat.server.graphql.testing_resolvers import Mutation as TestingMutation
from constat.server.graphql.testing_resolvers import Query as TestingQuery
from constat.server.graphql.public_resolvers import Query as PublicQuery
from constat.server.graphql.subscriptions import ExecutionSubscription, GlossarySubscription

logger = logging.getLogger(__name__)

# -- Schema stitching ----------------------------------------------------------
# merge_types composes multiple @strawberry.type classes into a single root type.
# As new domain resolver files are added (sessions.py, tables.py, etc.), import
# their Query/Mutation classes and add them to the tuple.

Query = merge_types("Query", (GlossaryQuery, AuthQuery, SessionQuery, StateQuery, DataQuery, SourceQuery, DomainQuery, ExecutionQuery, LearningQuery, FineTuneQuery, FeedbackQuery, TestingQuery, PublicQuery))
Mutation = merge_types("Mutation", (GlossaryMutation, AuthMutation, SessionMutation, StateMutation, DataMutation, SourceMutation, DomainMutation, ExecutionMutation, LearningMutation, FineTuneMutation, FeedbackMutation, TestingMutation))
Subscription = merge_types("Subscription", (GlossarySubscription, ExecutionSubscription))

schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)


def _extract_bearer_token(conn: Request | WebSocket) -> str | None:
    """Extract Bearer token from HTTP headers or WS connection params."""
    auth_header = conn.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


async def get_context(request: Request = None, ws: WebSocket = None) -> GraphQLContext:
    conn = request or ws
    server_config = conn.app.state.server_config
    config = getattr(conn.app.state, "config", None)
    token = _extract_bearer_token(conn)
    try:
        user_id = authenticate_token(token, server_config)
    except Exception:
        # Allow unauthenticated access (e.g. login mutation).
        # Resolvers that need auth check info.context.user_id.
        user_id = None
    return GraphQLContext(
        session_manager=conn.app.state.session_manager,
        server_config=server_config,
        user_id=user_id,
        config=config,
    )


graphql_router = GraphQLRouter(schema, context_getter=get_context)
