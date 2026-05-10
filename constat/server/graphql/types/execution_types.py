# Copyright (c) 2025 Kenneth Stott
# Canary: 4a18da44-0a21-494b-86c8-d4a67a65a5c4
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for query execution (Phase 7)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class ExecutionEventType:
    """Subscription event — mirrors StepEventWS shape with JSON data payload."""

    event_type: str
    session_id: str
    step_number: int
    timestamp: str
    data: JSON


@strawberry.type
class QuerySubmissionType:
    execution_id: str
    status: str
    message: Optional[str] = None


@strawberry.type
class PlanStepType:
    number: int
    goal: str
    status: str
    expected_inputs: list[str]
    expected_outputs: list[str]
    depends_on: list[int]
    code: Optional[str] = None
    domain: Optional[str] = None
    result: Optional[JSON] = None


@strawberry.type
class ExecutionPlanType:
    problem: str
    steps: list[PlanStepType]
    current_step: int
    completed_steps: list[int]
    failed_steps: list[int]
    is_complete: bool


@strawberry.type
class AutocompleteItemType:
    label: str
    value: str
    description: Optional[str] = None


@strawberry.type
class AutocompleteResultType:
    request_id: str
    items: list[AutocompleteItemType]


@strawberry.input
class EditedStepInput:
    number: int
    goal: str


@strawberry.input
class SubmitQueryInput:
    problem: str
    is_followup: bool = False
    require_approval: Optional[bool] = strawberry.UNSET
    replay: bool = False
    objective_index: Optional[int] = strawberry.UNSET


@strawberry.input
class ApprovePlanInput:
    approved: bool
    feedback: Optional[str] = strawberry.UNSET
    deleted_steps: Optional[list[int]] = strawberry.UNSET
    edited_steps: Optional[list[EditedStepInput]] = strawberry.UNSET


@strawberry.type
class ExecutionActionResultType:
    status: str
    message: Optional[str] = None
