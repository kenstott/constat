# Copyright (c) 2025 Kenneth Stott
# Canary: 07aac411-1651-40d2-a983-c55742ca3e6b
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for fine-tune, feedback, testing, and OAuth (Phase 9)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


# ---------------------------------------------------------------------------
# Fine-tune types
# ---------------------------------------------------------------------------


@strawberry.type
class FineTuneJobType:
    id: str
    name: str
    provider: str
    base_model: str
    fine_tuned_model_id: Optional[str]
    task_types: list[str]
    domain: Optional[str]
    status: str
    created: str
    exemplar_count: int
    metrics: Optional[JSON]
    training_data_path: Optional[str]


@strawberry.type
class FineTuneProviderType:
    name: str
    models: list[str]


@strawberry.input
class StartFineTuneInput:
    name: str
    provider: str
    base_model: str
    task_types: list[str]
    domain: Optional[str] = None
    include: Optional[list[str]] = None
    min_confidence: Optional[float] = None
    hyperparams: Optional[JSON] = None


# ---------------------------------------------------------------------------
# Feedback types
# ---------------------------------------------------------------------------


@strawberry.input
class FlagAnswerInput:
    session_id: str
    query_text: str
    answer_summary: str
    message: str
    glossary_term: Optional[str] = None
    suggested_definition: Optional[str] = None


@strawberry.type
class FlagAnswerResultType:
    learning_id: str
    glossary_suggestion_id: Optional[str]


@strawberry.type
class GlossarySuggestionType:
    learning_id: str
    term: str
    suggested_definition: str
    message: str
    created: str
    user_id: Optional[str]


@strawberry.type
class SuggestionActionResultType:
    status: str
    learning_id: str


# ---------------------------------------------------------------------------
# Testing types
# ---------------------------------------------------------------------------


@strawberry.type
class TestableDomainType:
    filename: str
    name: str
    question_count: int
    tags: list[str]


@strawberry.type
class GoldenQuestionExpectationType:
    terms: list[JSON]
    grounding: list[JSON]
    relationships: list[JSON]
    expected_outputs: list[JSON]
    end_to_end: Optional[JSON]
    suggested_question: Optional[str]
    objectives: list[str]
    step_hints: list[JSON]
    system_prompt: Optional[str]


@strawberry.type
class GoldenQuestionType:
    question: str
    tags: list[str]
    expect: JSON
    objectives: Optional[list[str]]
    system_prompt: Optional[str]
    index: Optional[int]
    warnings: Optional[list[str]] = None
    domain: Optional[str] = None


@strawberry.input
class GoldenQuestionExpectInput:
    proof_nodes: Optional[JSON] = None
    original_question: Optional[str] = None
    proof_summary: Optional[str] = None


@strawberry.input
class CreateGoldenQuestionInput:
    question: str
    tags: list[str]
    expect: JSON
    objectives: Optional[list[str]] = None
    system_prompt: Optional[str] = None


@strawberry.input
class UpdateGoldenQuestionInput:
    question: str
    tags: list[str]
    expect: JSON
    objectives: Optional[list[str]] = None
    system_prompt: Optional[str] = None


@strawberry.input
class MoveGoldenQuestionInput:
    target_domain: str
    validate_only: bool = False


# ---------------------------------------------------------------------------
# OAuth types
# ---------------------------------------------------------------------------


@strawberry.type
class EmailOAuthProvidersType:
    google: bool
    microsoft: bool
