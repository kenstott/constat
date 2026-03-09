# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for modular step system prompt: section parsing and selective composition."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest

from constat.core.models import TaskType
from constat.session._types import _parse_prompt_sections, STEP_PROMPT_SECTIONS


# ---------------------------------------------------------------------------
# _parse_prompt_sections
# ---------------------------------------------------------------------------

class TestParsePromptSections:

    def test_all_expected_tags_present(self):
        expected = {"core", "database", "api", "doc_tools", "llm_tools",
                    "llm_guide", "data_integrity", "skills", "pitfalls", "rules"}
        actual = {tag for tag, _, _ in STEP_PROMPT_SECTIONS}
        assert expected == actual

    def test_order_preserved(self):
        tags = [tag for tag, _, _ in STEP_PROMPT_SECTIONS]
        assert tags == [
            "core", "database", "api", "doc_tools", "llm_tools",
            "llm_guide", "data_integrity", "skills", "pitfalls", "rules",
        ]

    def test_no_empty_sections(self):
        for tag, content, _ in STEP_PROMPT_SECTIONS:
            assert content.strip(), f"Section '{tag}' is empty"

    def test_markers_not_in_content(self):
        for _, content, _ in STEP_PROMPT_SECTIONS:
            assert "<!-- @" not in content

    def test_all_base_sections_have_no_family(self):
        for tag, _, family in STEP_PROMPT_SECTIONS:
            assert family is None, f"Section '{tag}' has unexpected family '{family}'"

    def test_simple_parse(self):
        text = "line1\n<!-- @foo -->\nline2\n<!-- @bar -->\nline3"
        sections = _parse_prompt_sections(text)
        assert sections == [
            ("core", "line1", None),
            ("foo", "line2", None),
            ("bar", "line3", None),
        ]

    def test_family_parse(self):
        text = "default\n<!-- @pitfalls:ollama -->\nollama specific"
        sections = _parse_prompt_sections(text)
        assert sections == [
            ("core", "default", None),
            ("pitfalls", "ollama specific", "ollama"),
        ]

    def test_mixed_generic_and_family(self):
        text = (
            "<!-- @pitfalls -->\ngeneric pitfalls\n"
            "<!-- @pitfalls:ollama -->\nollama pitfalls\n"
            "<!-- @rules -->\nrules"
        )
        sections = _parse_prompt_sections(text)
        assert sections == [
            ("pitfalls", "generic pitfalls", None),
            ("pitfalls", "ollama pitfalls", "ollama"),
            ("rules", "rules", None),
        ]

    def test_no_markers(self):
        text = "just plain text\nno markers"
        sections = _parse_prompt_sections(text)
        assert sections == [("core", "just plain text\nno markers", None)]

    def test_ask_user_placeholder_in_core(self):
        core_sections = [(t, c) for t, c, _ in STEP_PROMPT_SECTIONS if t == "core"]
        assert any("{ask_user_docs}" in c for _, c in core_sections)


# ---------------------------------------------------------------------------
# _get_step_system_prompt — section filtering
# ---------------------------------------------------------------------------

def _make_mixin(*, databases=None, has_apis=False, has_documents=False, active_skills=None):
    """Build a minimal PromptsMixin-like object with required attributes."""
    from constat.session._prompts import PromptsMixin

    mixin = PromptsMixin()
    mixin.config = SimpleNamespace(databases=databases or {})
    mixin.resources = SimpleNamespace(
        has_apis=lambda: has_apis,
        has_documents=lambda: has_documents,
    )
    mixin.skill_manager = SimpleNamespace(
        active_skill_objects=active_skills or [],
    )
    return mixin


def _make_step(task_type, number=1):
    return SimpleNamespace(task_type=task_type, number=number)


class TestGetStepSystemPrompt:

    def test_user_input_minimal(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.USER_INPUT, number=1)
        prompt = mixin._get_step_system_prompt(step)

        # Should include core + rules
        assert "Your Task" in prompt
        assert "Output Format" in prompt
        # Should NOT include code-gen sections (check headings)
        assert "## Database Access Patterns" not in prompt
        assert "## LLM Primitive Selection Guide" not in prompt
        assert "## Trust Prior Step Results" not in prompt
        assert "## Common Pitfalls" not in prompt
        assert "## Using Skill Functions" not in prompt

    def test_user_input_has_widget_docs(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.USER_INPUT)
        prompt = mixin._get_step_system_prompt(step)
        assert 'widget="choice"' in prompt

    def test_python_step1_no_sources(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step)

        # Should have core, llm_tools, llm_guide, pitfalls, rules
        assert "llm_ask" in prompt
        assert "LLM Primitive Selection Guide" in prompt
        assert "Common Pitfalls" in prompt
        assert "Output Format" in prompt
        # Should NOT have data_integrity (step 1), database, api, skills
        assert "Trust Prior Step Results" not in prompt
        assert "Database Access Patterns" not in prompt
        assert "API Usage" not in prompt
        assert "Using Skill Functions" not in prompt

    def test_python_step2_with_databases(self):
        mixin = _make_mixin(databases={"sales": {}})
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=2)
        prompt = mixin._get_step_system_prompt(step)

        assert "Database Access Patterns" in prompt
        assert "Trust Prior Step Results" in prompt
        assert "llm_ask" in prompt

    def test_python_with_apis(self):
        mixin = _make_mixin(has_apis=True)
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step)
        assert "API Usage" in prompt

    def test_python_with_documents(self):
        mixin = _make_mixin(has_documents=True)
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step)
        assert "Document Tools" in prompt
        assert "doc_read" in prompt

    def test_python_with_skills(self):
        mixin = _make_mixin(active_skills=[SimpleNamespace(name="test-skill")])
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step)
        assert "Using Skill Functions" in prompt

    def test_sql_step1_with_databases(self):
        mixin = _make_mixin(databases={"sales": {}})
        step = _make_step(TaskType.SQL_GENERATION, number=1)
        prompt = mixin._get_step_system_prompt(step)

        assert "Database Access Patterns" in prompt
        assert "Common Pitfalls" in prompt
        # SQL should NOT have llm_tools or llm_guide sections
        assert "## LLM Primitive Selection Guide" not in prompt
        assert "## LLM Functions" not in prompt
        # Step 1 — no data_integrity
        assert "Trust Prior Step Results" not in prompt

    def test_sql_step2_has_data_integrity(self):
        mixin = _make_mixin(databases={"sales": {}})
        step = _make_step(TaskType.SQL_GENERATION, number=2)
        prompt = mixin._get_step_system_prompt(step)
        assert "Trust Prior Step Results" in prompt

    def test_full_python_all_sources(self):
        """Step 2+ PYTHON_ANALYSIS with all sources — every section present."""
        mixin = _make_mixin(
            databases={"db": {}},
            has_apis=True,
            has_documents=True,
            active_skills=[SimpleNamespace(name="s")],
        )
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=3)
        prompt = mixin._get_step_system_prompt(step)

        for section_name in ["Database Access Patterns", "API Usage", "Document Tools",
                             "LLM Functions", "LLM Primitive Selection Guide",
                             "Trust Prior Step Results", "Using Skill Functions",
                             "Common Pitfalls", "Output Format"]:
            assert section_name in prompt, f"Missing section: {section_name}"

    def test_user_input_much_smaller_than_full(self):
        full_mixin = _make_mixin(
            databases={"db": {}}, has_apis=True, has_documents=True,
            active_skills=[SimpleNamespace(name="s")],
        )
        full_step = _make_step(TaskType.PYTHON_ANALYSIS, number=3)
        full_prompt = full_mixin._get_step_system_prompt(full_step)

        ui_mixin = _make_mixin()
        ui_step = _make_step(TaskType.USER_INPUT)
        ui_prompt = ui_mixin._get_step_system_prompt(ui_step)

        # USER_INPUT should be at least 50% smaller
        assert len(ui_prompt) < len(full_prompt) * 0.5

    def test_ask_user_placeholder_replaced(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step)
        assert "{ask_user_docs}" not in prompt
        assert "ask_user(" in prompt


# ---------------------------------------------------------------------------
# Model-family-specific section overrides
# ---------------------------------------------------------------------------

class TestModelFamilySections:

    def test_no_family_uses_generic(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt_default = mixin._get_step_system_prompt(step)
        prompt_none = mixin._get_step_system_prompt(step, model_family=None)
        assert prompt_default == prompt_none

    def test_family_override_replaces_generic(self):
        """When a family-specific section exists, it replaces the generic one."""
        from constat.session._types import STEP_PROMPT_SECTIONS as real_sections
        from constat.session import _prompts

        # Temporarily inject a family-specific pitfalls override
        original = list(real_sections)
        real_sections.append(("pitfalls", "OLLAMA_SPECIFIC_PITFALLS", "ollama"))
        try:
            mixin = _make_mixin()
            step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)

            prompt_generic = mixin._get_step_system_prompt(step, model_family="anthropic")
            prompt_ollama = mixin._get_step_system_prompt(step, model_family="ollama")

            # Generic should have normal pitfalls, not ollama-specific
            assert "OLLAMA_SPECIFIC_PITFALLS" not in prompt_generic
            assert "Common Pitfalls" in prompt_generic

            # Ollama should have the override, not the generic
            assert "OLLAMA_SPECIFIC_PITFALLS" in prompt_ollama
            assert "## Common Pitfalls" not in prompt_ollama
        finally:
            # Restore original sections
            real_sections.clear()
            real_sections.extend(original)

    def test_unknown_family_uses_generic(self):
        mixin = _make_mixin()
        step = _make_step(TaskType.PYTHON_ANALYSIS, number=1)
        prompt = mixin._get_step_system_prompt(step, model_family="some_unknown_provider")
        # Should fall through to all generic sections
        assert "Common Pitfalls" in prompt
        assert "LLM Primitive Selection Guide" in prompt
