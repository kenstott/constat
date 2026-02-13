# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Feedback package — live terminal display for session execution."""

from constat.repl.feedback._display import FeedbackDisplay
from constat.repl.feedback._handler import SessionFeedbackHandler
from constat.repl.feedback._models import (
    SPINNER_FRAMES,
    PlanItem,
    StepDisplay,
    _left_align_markdown,
)
from constat.repl.feedback._plan_display import LivePlanExecutionDisplay
from constat.repl.feedback._status import PersistentStatusBar, StatusLine

__all__ = [
    "FeedbackDisplay",
    "SessionFeedbackHandler",
    "StatusLine",
    "PersistentStatusBar",
    "LivePlanExecutionDisplay",
    "SPINNER_FRAMES",
    "StepDisplay",
    "PlanItem",
    "_left_align_markdown",
]
