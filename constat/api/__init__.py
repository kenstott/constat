# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ConstatAPI - Clean API boundary for REPL/UI consumers.

This module provides a clean API interface that encapsulates all business
logic, allowing REPL/UI to import from a single location without direct
access to internal implementation details.

Usage:
    ```python
    from constat.api import create_api, ConstatAPI

    # Create an API instance
    api = create_api("config.yaml", user_id="test")

    # Solve a problem
    result = api.solve("What tables are available?")
    print(result.answer)

    # Follow up
    result = api.follow_up("Show me the first 10 rows")
    print(result.answer)

    # Check if implementation satisfies protocol
    assert isinstance(api, ConstatAPI)  # Works due to Protocol
    ```
"""

# Detection utilities (for direct use if needed)
from constat.api.detection import (
    CORRECTION_PATTERNS,
    DISPLAY_OVERRIDE_PATTERNS,
    detect_display_overrides,
    detect_nl_correction,
)
# Factory function - primary entry point
from constat.api.factory import create_api
# Implementation
from constat.api.impl import ConstatAPIImpl
# Protocol definition
from constat.api.protocol import ConstatAPI, EventCallback
# Value objects (frozen dataclasses)
from constat.api.types import (
    # Detection results
    CorrectionDetection,
    DisplayOverrides,
    # Fact types
    Fact,
    # Learning types
    Learning,
    LearningCompactionResult,
    Rule,
    # Plan types
    SavedPlan,
    # Session state types
    ContextCompactionResult,
    ContextStats,
    SessionState,
    # Summarization results
    SummarizeResult,
    # Core operation results
    ArtifactInfo,
    FollowUpResult,
    ReplayResult,
    ResumeResult,
    SolveResult,
    StepInfo,
)

__all__ = [
    # Factory
    "create_api",
    # Protocol
    "ConstatAPI",
    "EventCallback",
    # Implementation
    "ConstatAPIImpl",
    # Value objects - Detection
    "CorrectionDetection",
    "DisplayOverrides",
    # Value objects - Facts
    "Fact",
    # Value objects - Learnings
    "Learning",
    "LearningCompactionResult",
    "Rule",
    # Value objects - Plans
    "SavedPlan",
    # Value objects - Session state
    "ContextCompactionResult",
    "ContextStats",
    "SessionState",
    # Value objects - Summarization
    "SummarizeResult",
    # Value objects - Core operations
    "ArtifactInfo",
    "FollowUpResult",
    "ReplayResult",
    "ResumeResult",
    "SolveResult",
    "StepInfo",
    # Detection utilities
    "CORRECTION_PATTERNS",
    "DISPLAY_OVERRIDE_PATTERNS",
    "detect_display_overrides",
    "detect_nl_correction",
]
