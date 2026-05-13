# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for follow_up() context provenance annotation.

Verifies that tables in the planner context are annotated with the source
question that produced them, so the planner can ignore tables from unrelated
prior objectives without a topic-shift classifier.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scratchpad_entries(entries: list[dict]) -> list[dict]:
    """Build scratchpad entry dicts matching the datastore schema."""
    return [
        {
            "step_number": e["step_number"],
            "goal": e.get("goal", f"Step {e['step_number']}"),
            "narrative": e.get("narrative", ""),
            "tables_created": e.get("tables_created", []),
            "code": e.get("code", ""),
            "user_query": e.get("user_query", ""),
            "objective_index": e.get("objective_index", 0),
        }
        for e in entries
    ]


def _make_tables(entries: list[dict]) -> list[dict]:
    """Build table list dicts matching list_tables() output."""
    return [
        {
            "name": e["name"],
            "step_number": e.get("step_number", 1),
            "row_count": e.get("row_count", 10),
            "created_at": None,
            "description": None,
            "is_published": False,
            "is_final_step": False,
            "version": 1,
            "version_count": 1,
            "is_view": False,
        }
        for e in entries
    ]


def _make_session(scratchpad: list[dict], tables: list[dict]) -> MagicMock:
    """Return a minimal mock Session with pre-populated datastore."""
    session = MagicMock()
    session.session_id = "test-session-123"

    ds = MagicMock()
    ds.get_state.return_value = []
    ds.get_all_state.return_value = {}
    ds.get_scratchpad.return_value = scratchpad
    ds.get_scratchpad_as_markdown.return_value = "(scratchpad)"
    ds.list_tables.return_value = tables
    ds.get_session_meta.return_value = scratchpad[0]["user_query"] if scratchpad else ""
    ds.set_state.return_value = None
    session.datastore = ds

    # Planner returns a minimal valid plan (no steps — enough to not crash)
    plan = MagicMock()
    plan.steps = []
    plan.get_execution_order.return_value = []
    planner_response = MagicMock()
    planner_response.plan = plan
    planner_response.reasoning = ""
    session.planner = MagicMock()
    session.planner.plan.return_value = planner_response

    # Stubs for everything follow_up() touches before the planner call
    session.fact_resolver.get_unresolved_facts.return_value = []
    session.history.log_user_input.return_value = None
    session.history.record_query.return_value = None
    session.history.complete_session.return_value = None
    session._emit_event = MagicMock()
    session._try_show_existing_data.return_value = None
    session._analyze_question.return_value = MagicMock(
        intents=[],
        fact_modifications=[],
        wants_brief=False,
        extracted_facts=[],
        cached_fact_answer=None,
        recommended_mode="exploratory",
    )
    session._sync_user_facts_to_planner.return_value = None
    session._sync_glossary_to_planner.return_value = None
    session._sync_available_agents_to_planner.return_value = None
    session._ensure_enhance_updates_source.side_effect = lambda _q, p, _t: p
    session.session_config.auto_approve = True
    session.session_config.require_approval = False
    session.session_config.force_approval = False
    session.session_config.enable_insights = False
    session.session_config.max_replan_attempts = 3

    return session


def _call_follow_up(session: MagicMock, question: str) -> str:
    """Call FollowUpMixin.follow_up() on a mock session and return planner prompt."""
    from constat.session._follow_up import FollowUpMixin

    # Bind the mixin method to our mock session
    follow_up = FollowUpMixin.follow_up.__get__(session, type(session))

    # Patch load_prompt so we can capture the formatted context
    captured = {}

    original_load = __import__(
        "constat.prompts", fromlist=["load_prompt"]
    ).load_prompt

    def _patched_load(name):
        tmpl = original_load(name)
        if name == "followup_context.md":
            class _CapturingTemplate(str):
                def format(self, **kwargs):
                    captured.update(kwargs)
                    return str.__new__(str, str(self)).format(**kwargs)
            return _CapturingTemplate(tmpl)
        return tmpl

    with patch("constat.session._follow_up.load_prompt", side_effect=_patched_load):
        try:
            follow_up(question, auto_classify=False)
        except Exception:
            pass  # We only care about the captured context, not the full result

    return captured.get("existing_tables_list", "")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFollowUpContextProvenance:

    def test_table_annotated_with_source_query(self):
        """Tables produced by a prior step are annotated with the originating question."""
        scratchpad = _make_scratchpad_entries([
            {"step_number": 1, "goal": "Count genres", "user_query": "How many genres are there?",
             "tables_created": ["genre_counts"]},
        ])
        tables = _make_tables([
            {"name": "genre_counts", "step_number": 1, "row_count": 5},
        ])
        session = _make_session(scratchpad, tables)

        table_list = _call_follow_up(session, "What are the top albums?")

        assert 'from: "How many genres are there?"' in table_list

    def test_tables_from_different_objectives_labeled_separately(self):
        """Tables from two distinct prior questions each carry their own label."""
        scratchpad = _make_scratchpad_entries([
            {"step_number": 1, "user_query": "Show sales by region", "tables_created": ["sales_by_region"]},
            {"step_number": 2, "user_query": "Show headcount by department", "tables_created": ["headcount"]},
        ])
        tables = _make_tables([
            {"name": "sales_by_region", "step_number": 1, "row_count": 20},
            {"name": "headcount", "step_number": 2, "row_count": 15},
        ])
        session = _make_session(scratchpad, tables)

        table_list = _call_follow_up(session, "What is the headcount in engineering?")

        assert 'from: "Show sales by region"' in table_list
        assert 'from: "Show headcount by department"' in table_list

    def test_tables_without_scratchpad_entry_have_no_annotation(self):
        """Internal/system tables with no scratchpad entry are listed without annotation."""
        scratchpad = _make_scratchpad_entries([])
        tables = _make_tables([
            {"name": "_constat_execution_history", "step_number": None, "row_count": 3},
        ])
        # Override step_number to None to simulate internal tables
        tables[0]["step_number"] = None
        session = _make_session(scratchpad, tables)

        table_list = _call_follow_up(session, "Anything")

        assert "from:" not in table_list
        assert "_constat_execution_history" in table_list

    def test_related_followup_same_query_shares_label(self):
        """Multiple steps from the same objective share one label on each table."""
        scratchpad = _make_scratchpad_entries([
            {"step_number": 1, "user_query": "Analyse Q1 revenue", "tables_created": ["raw_revenue"]},
            {"step_number": 2, "user_query": "Analyse Q1 revenue", "tables_created": ["revenue_by_product"]},
        ])
        tables = _make_tables([
            {"name": "raw_revenue", "step_number": 1, "row_count": 100},
            {"name": "revenue_by_product", "step_number": 2, "row_count": 30},
        ])
        session = _make_session(scratchpad, tables)

        table_list = _call_follow_up(session, "Add margin column to revenue_by_product")

        assert table_list.count('from: "Analyse Q1 revenue"') == 2

    def test_planner_receives_annotated_context(self):
        """The planner.plan() call receives the annotated table list, not the bare one."""
        scratchpad = _make_scratchpad_entries([
            {"step_number": 1, "user_query": "Count employees", "tables_created": ["employees"]},
        ])
        tables = _make_tables([{"name": "employees", "step_number": 1, "row_count": 200}])
        session = _make_session(scratchpad, tables)

        from constat.session._follow_up import FollowUpMixin
        follow_up = FollowUpMixin.follow_up.__get__(session, type(session))

        with patch("constat.session._follow_up.load_prompt") as mock_load:
            tmpl = MagicMock()
            tmpl.format.return_value = "formatted_prompt"
            mock_load.return_value = tmpl
            try:
                follow_up("Show revenue by region", auto_classify=False)
            except Exception:
                pass

        # Confirm format() was called with an annotated existing_tables_list
        call_kwargs = tmpl.format.call_args.kwargs if tmpl.format.called else {}
        tables_arg = call_kwargs.get("existing_tables_list", "")
        assert 'from: "Count employees"' in tables_arg
