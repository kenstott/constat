# Copyright (c) 2025 Kenneth Stott
# Canary: 0d67a594-4f42-4045-8487-30f6d7f4ea38
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for learnings, rules, skills, and agents (Phase 8)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class LearningInfoType:
    id: str
    content: str
    category: str
    source: Optional[str] = None
    context: Optional[JSON] = None
    applied_count: int = 0
    created_at: Optional[str] = None
    scope: Optional[JSON] = None


@strawberry.type
class RuleInfoType:
    id: str
    summary: str
    category: str
    confidence: float
    source_count: int
    tags: list[str]
    domain: Optional[str] = None
    source: Optional[str] = None
    scope: Optional[JSON] = None


@strawberry.type
class LearningListType:
    learnings: list[LearningInfoType]
    rules: list[RuleInfoType]


@strawberry.type
class CompactionResultType:
    status: str
    message: Optional[str] = None
    rules_created: int = 0
    rules_strengthened: int = 0
    rules_merged: int = 0
    learnings_archived: int = 0
    groups_found: int = 0
    skipped_low_confidence: int = 0
    errors: Optional[list[str]] = None


@strawberry.input
class CreateRuleInput:
    summary: str
    category: str = "user_correction"
    confidence: float = 0.9
    tags: list[str] = strawberry.field(default_factory=list)


@strawberry.input
class UpdateRuleInput:
    summary: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[list[str]] = None


@strawberry.type
class SkillInfoType:
    name: str
    description: Optional[str] = None
    prompt: Optional[str] = None
    filename: Optional[str] = None
    is_active: bool = False
    domain: Optional[str] = None
    source: Optional[str] = None


@strawberry.type
class SkillsListType:
    skills: list[SkillInfoType]
    active_skills: list[str]
    skills_dir: Optional[str] = None


@strawberry.type
class SkillContentType:
    name: str
    content: str
    path: str


@strawberry.type
class SetActiveSkillsResultType:
    status: str
    active_skills: list[str]


@strawberry.type
class AgentInfoType:
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None
    source: Optional[str] = None
    is_active: bool = False


@strawberry.type
class SetAgentResultType:
    success: bool
    current_agent: Optional[str] = None
    message: str


# --- Skill CRUD types ---

@strawberry.input
class CreateSkillInput:
    name: str
    prompt: str
    description: str = ""


@strawberry.input
class UpdateSkillInput:
    content: str  # Raw YAML content


@strawberry.input
class DraftSkillInput:
    name: str
    user_description: str


@strawberry.type
class DraftSkillResultType:
    name: str
    content: str
    description: str


@strawberry.input
class CreateSkillFromProofInput:
    name: str
    description: str = ""


@strawberry.type
class CreateSkillFromProofResultType:
    name: str
    content: str
    description: str
    has_script: bool


# --- Agent CRUD types ---

@strawberry.input
class CreateAgentInput:
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = strawberry.field(default_factory=list)


@strawberry.input
class UpdateAgentInput:
    prompt: str
    description: str = ""
    skills: list[str] = strawberry.field(default_factory=list)


@strawberry.input
class DraftAgentInput:
    name: str
    user_description: str


@strawberry.type
class DraftAgentResultType:
    name: str
    prompt: str
    description: str
    skills: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class AgentContentType:
    name: str
    prompt: str
    description: str
    skills: list[str] = strawberry.field(default_factory=list)
    path: str
