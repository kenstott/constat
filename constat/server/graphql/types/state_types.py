# Copyright (c) 2025 Kenneth Stott
# Canary: ab5b64dc-c157-43c7-bf6a-d8f6de6381e6
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for session state (Phase 3)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class StepCodeType:
    step_number: int
    goal: str
    code: str
    prompt: Optional[str] = None
    model: Optional[str] = None


@strawberry.type
class StepCodeListType:
    steps: list[StepCodeType]
    total: int


@strawberry.type
class InferenceCodeType:
    inference_id: str
    name: str
    operation: str
    code: str
    attempt: int
    prompt: Optional[str] = None
    model: Optional[str] = None


@strawberry.type
class InferenceCodeListType:
    inferences: list[InferenceCodeType]
    total: int


@strawberry.type
class ScratchpadEntryType:
    step_number: int
    goal: str
    narrative: str
    tables_created: list[str]
    code: str
    user_query: str
    objective_index: Optional[int] = None


@strawberry.type
class ScratchpadType:
    entries: list[ScratchpadEntryType]
    total: int


@strawberry.type
class ExecutionOutputType:
    output: str
    suggestions: list[str]
    current_query: Optional[str] = None


@strawberry.type
class ProofTreeNodeType:
    name: str
    value: Optional[JSON] = None
    source: str
    reasoning: Optional[str] = None
    dependencies: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class ProofTreeType:
    facts: list[ProofTreeNodeType]
    execution_trace: list[JSON] = strawberry.field(default_factory=list)


@strawberry.type
class StoredMessageType:
    id: str
    type: str
    content: str
    timestamp: str
    step_number: Optional[int] = None
    is_final_insight: Optional[bool] = None
    step_duration_ms: Optional[int] = None
    role: Optional[str] = None
    skills: Optional[list[str]] = None


@strawberry.type
class MessagesType:
    messages: list[StoredMessageType]


@strawberry.type
class StoredProofFactType:
    id: str
    name: str
    description: Optional[str] = None
    status: str
    value: Optional[JSON] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    tier: Optional[str] = None
    strategy: Optional[str] = None
    formula: Optional[str] = None
    reason: Optional[str] = None
    dependencies: list[str] = strawberry.field(default_factory=list)
    elapsed_ms: Optional[int] = None


@strawberry.type
class ProofFactsType:
    facts: list[StoredProofFactType]
    summary: Optional[str] = None


@strawberry.type
class ObjectivesEntryType:
    type: str
    text: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    mode: Optional[str] = None
    guidance: Optional[str] = None
    ts: Optional[str] = None


@strawberry.type
class ActiveAgentType:
    name: str
    prompt: str


@strawberry.type
class ActiveSkillType:
    name: str
    prompt: str
    description: Optional[str] = None


@strawberry.type
class PromptContextType:
    system_prompt: str
    active_agent: Optional[ActiveAgentType] = None
    active_skills: list[ActiveSkillType] = strawberry.field(default_factory=list)


@strawberry.type
class DatabaseTableInfoType:
    name: str
    row_count: Optional[int] = None
    column_count: int


@strawberry.type
class DatabaseSchemaType:
    database: str
    tables: list[DatabaseTableInfoType]


@strawberry.type
class ApiFieldType:
    name: str
    type: str
    description: Optional[str] = None
    is_required: bool


@strawberry.type
class ApiEndpointType:
    name: str
    kind: str
    return_type: Optional[str] = None
    description: Optional[str] = None
    http_method: Optional[str] = None
    http_path: Optional[str] = None
    fields: list[ApiFieldType] = strawberry.field(default_factory=list)


@strawberry.type
class ApiSchemaType:
    name: str
    type: str
    description: Optional[str] = None
    endpoints: list[ApiEndpointType] = strawberry.field(default_factory=list)


@strawberry.type
class SaveResultType:
    status: str
    count: int


@strawberry.type
class UpdateSystemPromptResultType:
    status: str
    system_prompt: str


@strawberry.type
class PublicSessionSummaryType:
    session_id: str
    summary: Optional[str] = None
    status: str = "idle"
