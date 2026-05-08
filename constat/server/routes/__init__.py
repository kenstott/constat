# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""API route modules."""

from constat.server.routes.data import router as data_router
from constat.server.routes.queries import router as queries_router
from constat.server.routes.roles import router as roles_router
from constat.server.routes.schema import router as schema_router
from constat.server.routes.sessions import router as sessions_router

__all__ = ["sessions_router", "queries_router", "data_router", "schema_router", "roles_router"]
