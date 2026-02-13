# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session orchestration for multi-step plan execution."""

from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse
from constat.session._types import (
    ApprovalCallback,
    ClarificationCallback,
    ClarificationQuestion,
    ClarificationRequest,
    ClarificationResponse,
    DetectedIntent,
    QuestionAnalysis,
    QuestionType,
    SessionConfig,
    StepEvent,
    create_session,
    is_meta_question,
)
from constat.session._core import CoreMixin
from constat.session._prompts import PromptsMixin
from constat.session._metadata import MetadataMixin
from constat.session._analysis import AnalysisMixin
from constat.session._resources import ResourcesMixin
from constat.session._plans import PlansMixin
from constat.session._intents import IntentsMixin
from constat.session._synthesis import SynthesisMixin
from constat.session._execution import ExecutionMixin
from constat.session._dag import DagMixin
from constat.session._auditable import AuditableMixin
from constat.session._follow_up import FollowUpMixin
from constat.session._solve import SolveMixin


class Session(
    SolveMixin,
    FollowUpMixin,
    AuditableMixin,
    DagMixin,
    ExecutionMixin,
    AnalysisMixin,
    SynthesisMixin,
    IntentsMixin,
    PlansMixin,
    ResourcesMixin,
    PromptsMixin,
    MetadataMixin,
    CoreMixin,
):
    """Orchestrates multi-step analytical plan execution."""


__all__ = [
    "Session",
    "SessionConfig",
    "StepEvent",
    "ApprovalCallback",
    "ClarificationCallback",
    "ClarificationQuestion",
    "ClarificationRequest",
    "ClarificationResponse",
    "QuestionType",
    "QuestionAnalysis",
    "DetectedIntent",
    "PlanApprovalRequest",
    "PlanApprovalResponse",
    "create_session",
    "is_meta_question",
]
