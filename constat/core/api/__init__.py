# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core API for Constat business logic.

This module provides clean interfaces for business logic that can be used
by presentation layers (REPL, UI) without directly accessing storage or
internal implementation details.

Exports:
    EntityManager: Manages entity extraction lifecycle for sessions
    EntityExtractionResult: Result dataclass for entity operations
"""

from constat.core.api.entity_manager import (
    EntityManager,
    EntityExtractionResult,
    VectorStoreProtocol,
)

__all__ = [
    "EntityManager",
    "EntityExtractionResult",
    "VectorStoreProtocol",
]
