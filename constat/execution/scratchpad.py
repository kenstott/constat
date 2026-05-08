# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Scratchpad for state sharing between execution steps.

The scratchpad is a markdown-based document that persists across steps,
allowing each step to read previous results and write new findings.
It provides context continuity between steps.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScratchpadSection:
    """A section in the scratchpad."""
    name: str
    content: str
    step_number: Optional[int] = None


class Scratchpad:
    """
    Markdown scratchpad for inter-step communication.

    Structure:
        ## Context
        [Initial context and problem description]

        ## Step 1: [Goal]
        [Results from step 1]

        ## Step 2: [Goal]
        [Results from step 2]

        ## Notes
        [General observations, intermediate reasoning]
    """

    def __init__(self, initial_context: str = ""):
        """Initialize scratchpad with optional context."""
        self._sections: list[ScratchpadSection] = []
        if initial_context:
            self._sections.append(ScratchpadSection(
                name="Context",
                content=initial_context.strip()
            ))

    def add_step_result(
        self,
        step_number: int,
        goal: str,
        result: str,
        tables_created: Optional[list[str]] = None,
    ) -> None:
        """
        Add results from a completed step.

        Args:
            step_number: The step number (1-indexed)
            goal: The goal of this step
            result: The result/output to record
            tables_created: List of tables created in this step
        """
        content_parts = [result.strip()]

        if tables_created:
            content_parts.append("")
            content_parts.append("**Tables created:** " + ", ".join(f"`{t}`" for t in tables_created))

        self._sections.append(ScratchpadSection(
            name=f"Step {step_number}: {goal}",
            content="\n".join(content_parts),
            step_number=step_number
        ))

    def add_note(self, note: str) -> None:
        """Add a note to the Notes section."""
        notes_section = None
        for section in self._sections:
            if section.name == "Notes":
                notes_section = section
                break

        if notes_section:
            notes_section.content += "\n\n" + note.strip()
        else:
            self._sections.append(ScratchpadSection(
                name="Notes",
                content=note.strip()
            ))

    def get_context(self) -> str:
        """Get the initial context section."""
        for section in self._sections:
            if section.name == "Context":
                return section.content
        return ""

    def get_step_result(self, step_number: int) -> Optional[str]:
        """Get the result from a specific step."""
        for section in self._sections:
            if section.step_number == step_number:
                return section.content
        return None

    def get_recent_context(self, max_steps: int = 5) -> str:
        """
        Get recent context for LLM prompt.

        Returns context + last N step results to stay within token limits.
        """
        parts = []

        # Always include context
        context = self.get_context()
        if context:
            parts.append(f"## Context\n{context}")

        # Get step sections
        step_sections = [s for s in self._sections if s.step_number is not None]
        step_sections.sort(key=lambda s: s.step_number or 0)

        # Include last N steps
        for section in step_sections[-max_steps:]:
            parts.append(f"## {section.name}\n{section.content}")

        # Always include notes if present
        for section in self._sections:
            if section.name == "Notes":
                parts.append(f"## Notes\n{section.content}")

        return "\n\n".join(parts)

    def to_markdown(self) -> str:
        """Export the full scratchpad as markdown."""
        parts = []
        for section in self._sections:
            parts.append(f"## {section.name}\n{section.content}")
        return "\n\n".join(parts)

    @classmethod
    def from_markdown(cls, markdown: str) -> "Scratchpad":
        """Load scratchpad from markdown content."""
        scratchpad = cls()

        # Parse sections
        section_pattern = r"## (.+?)\n(.*?)(?=\n## |\Z)"
        matches = re.findall(section_pattern, markdown, re.DOTALL)

        for name, content in matches:
            name = name.strip()
            content = content.strip()

            # Extract step number if present
            step_match = re.match(r"Step (\d+):", name)
            step_number = int(step_match.group(1)) if step_match else None

            scratchpad._sections.append(ScratchpadSection(
                name=name,
                content=content,
                step_number=step_number
            ))

        return scratchpad

    def __str__(self) -> str:
        return self.to_markdown()

    def __repr__(self) -> str:
        return f"Scratchpad({len(self._sections)} sections)"
