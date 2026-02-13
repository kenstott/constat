# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session types, dataclasses, constants, and free functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from constat.core.config import Config
from constat.execution.mode import Mode, PlanApprovalRequest, PlanApprovalResponse
from constat.prompts import load_prompt, load_yaml


META_QUESTION_PATTERNS = load_yaml("meta_question_patterns.yaml")["patterns"]


def is_meta_question(query: str) -> bool:
    """Check if query is a meta-question about capabilities."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in META_QUESTION_PATTERNS)


class QuestionType:
    DATA_ANALYSIS = "data_analysis"
    GENERAL_KNOWLEDGE = "general_knowledge"
    META_QUESTION = "meta_question"


@dataclass
class DetectedIntent:
    """A single detected intent with confidence."""
    intent: str
    confidence: float = 0.8
    extracted_value: Optional[str] = None


@dataclass
class QuestionAnalysis:
    """Combined result of question analysis (facts + classification + intent + mode)."""
    question_type: str
    extracted_facts: list = field(default_factory=list)
    cached_fact_answer: Optional[str] = None
    intents: list = field(default_factory=list)
    fact_modifications: list = field(default_factory=list)
    scope_refinements: list = field(default_factory=list)
    wants_brief: bool = False
    recommended_mode: Optional[str] = None
    mode_reasoning: Optional[str] = None


STEP_SYSTEM_PROMPT = load_prompt("step_system_prompt.md")
STEP_PROMPT_TEMPLATE = load_prompt("step_prompt_template.md")


ApprovalCallback = Callable[[PlanApprovalRequest], PlanApprovalResponse]


@dataclass
class ClarificationQuestion:
    """A single clarification question with optional suggested answers."""
    text: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ClarificationRequest:
    """Request for clarification before planning."""
    original_question: str
    ambiguity_reason: str
    questions: list[ClarificationQuestion]


@dataclass
class ClarificationResponse:
    """User's response to clarification request."""
    answers: dict[str, str]
    skip: bool = False


ClarificationCallback = Callable[[ClarificationRequest], ClarificationResponse]


@dataclass
class SessionConfig:
    """Configuration for a session."""
    max_retries_per_step: int = 10
    verbose: bool = False
    require_approval: bool = True
    max_replan_attempts: int = 3
    auto_approve: bool = False
    ask_clarifications: bool = True
    skip_clarification: bool = False
    enable_insights: bool = True
    show_raw_output: bool = True
    default_mode: Optional[Mode] = None


@dataclass
class StepEvent:
    """Event emitted during step execution."""
    event_type: str
    step_number: int
    data: dict = field(default_factory=dict)


def create_session(config_path: str, session_id: str) -> "Session":
    """Create a session from a config file path."""
    from constat.session import Session
    config = Config.from_yaml(config_path)
    return Session(config, session_id=session_id)
