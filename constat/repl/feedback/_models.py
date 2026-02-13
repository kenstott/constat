# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Shared models, constants, and utilities for the feedback package."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def _left_align_markdown(text: str) -> str:
    """Convert Markdown headers to bold text to avoid Rich's centering."""
    # Convert ## Header to **Header**
    text = re.sub(r'^#{1,6}\s+(.+)$', r'**\1**', text, flags=re.MULTILINE)
    return text


# Spinner frames for animation
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@dataclass
class StepDisplay:
    """Display state for a step."""
    number: int
    goal: str
    status: str = "pending"  # pending, running, generating, executing, completed, failed
    code: Optional[str] = None
    output: Optional[str] = None
    output_summary: Optional[str] = None  # Brief summary for display
    error: Optional[str] = None
    attempts: int = 0
    duration_ms: int = 0
    tables_created: list[str] = field(default_factory=list)
    status_message: str = ""  # Current status message for live display


@dataclass
class PlanItem:
    """A single item (premise or inference) in the execution plan."""
    fact_id: str  # P1, P2, I1, I2, etc.
    name: str
    item_type: str  # "premise" or "inference"
    status: str = "pending"  # pending, running, resolved, failed, blocked
    value: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0
    dependencies: list[str] = field(default_factory=list)  # List of fact_ids this depends on
