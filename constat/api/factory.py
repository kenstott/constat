# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Factory function for creating ConstatAPI instances.

Provides a clean entry point for creating API instances without
exposing implementation details.
"""

from pathlib import Path
from typing import Optional, Union

from constat.api.impl import ConstatAPIImpl
from constat.api.protocol import ConstatAPI
from constat.core.config import Config
from constat.session import Session, SessionConfig
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore


def create_api(
    config: Union[str, Path, Config],
    *,
    session_id: str,
    user_id: str = "default",
    verbose: bool = False,
    require_approval: bool = True,
    auto_approve: bool = False,
    enable_insights: bool = True,
    show_raw_output: bool = True,
) -> ConstatAPI:
    """Create a ConstatAPI instance.

    This is the primary entry point for creating API instances.
    It initializes all necessary components (Session, FactStore,
    LearningStore) and returns a clean API interface.

    Args:
        config: Path to config YAML file, or Config instance
        session_id: Client-provided session identifier
        user_id: User identifier for user-scoped storage
        verbose: Enable verbose output
        require_approval: Require approval before plan execution
        auto_approve: Auto-approve plans (for testing/scripts)
        enable_insights: Enable insight/synthesis generation
        show_raw_output: Show raw step output before synthesis

    Returns:
        ConstatAPI instance ready for use

    Example:
        ```python
        from constat.api import create_api

        api = create_api("config.yaml", session_id="abc123", user_id="test")
        result = api.solve("What tables are available?")
        print(result.answer)
        ```
    """
    # Load config if path provided
    if isinstance(config, (str, Path)):
        config = Config.load(config)

    # Create session config
    session_config = SessionConfig(
        verbose=verbose,
        require_approval=require_approval,
        auto_approve=auto_approve,
        enable_insights=enable_insights,
        show_raw_output=show_raw_output,
    )

    # Create session
    session = Session(
        config=config,
        session_id=session_id,
        session_config=session_config,
        user_id=user_id,
    )

    # Create stores
    fact_store = FactStore(user_id=user_id)
    learning_store = LearningStore(user_id=user_id)

    # Load persistent facts into session
    fact_store.load_into_session(session)

    # Create and return API implementation
    return ConstatAPIImpl(
        session=session,
        fact_store=fact_store,
        learning_store=learning_store,
    )
