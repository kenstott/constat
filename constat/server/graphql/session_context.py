# Copyright (c) 2025 Kenneth Stott
# Canary: 3801d954-3a27-4aac-848f-158138779482
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Typed GraphQL context shared by all resolvers.

Every resolver receives this via ``info.context``.  Import the type alias
``GqlInfo`` to get autocomplete and type-checking on context fields::

    from constat.server.graphql.session_context import GqlInfo

    @strawberry.field
    async def my_field(self, info: GqlInfo) -> str:
        sm = info.context.session_manager
        uid = info.context.user_id
        ...
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strawberry.fastapi.context import BaseContext
from strawberry.types import Info

if TYPE_CHECKING:
    from constat.core.config import Config
    from constat.server.config import ServerConfig
    from constat.server.session_manager import SessionManager


class GraphQLContext(BaseContext):
    """Typed context available to every GraphQL resolver.

    Inherits ``request``, ``background_tasks``, ``response`` from BaseContext.
    Strawberry sets those automatically after the context getter returns.

    Attributes:
        session_manager: Server session manager instance.
        server_config: Server configuration.
        user_id: Authenticated user ID (``"default"`` when auth is disabled).
        config: Main application Config object.
    """

    def __init__(
        self,
        session_manager: SessionManager,
        server_config: ServerConfig,
        user_id: str,
        config: Config | None = None,
    ):
        super().__init__()
        self.session_manager = session_manager
        self.server_config = server_config
        self.user_id = user_id
        self.config = config


# Convenience alias: use ``GqlInfo`` instead of ``Info[GraphQLContext, None]``
GqlInfo = Info[GraphQLContext, None]
