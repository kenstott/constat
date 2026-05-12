# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for DQ constraint parsing and evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


class TestParseDqAnnotations:
    def test_empty_code(self):
        from constat.execution.executor import parse_dq_annotations
        assert parse_dq_annotations("") == {}

    def test_no_annotations(self):
        from constat.execution.executor import parse_dq_annotations
        code = "df = store.load_dataframe('orders')\nresult = df[df.active]"
        assert parse_dq_annotations(code) == {}

    def test_single_annotation(self):
        from constat.execution.executor import parse_dq_annotations
        code = "# @dq[orders]: len(df) > 0"
        assert parse_dq_annotations(code) == {"orders": ["len(df) > 0"]}

    def test_multiple_tables(self):
        from constat.execution.executor import parse_dq_annotations
        code = (
            "# @dq[orders]: len(df) > 0\n"
            "# @dq[customers]: df['id'].notna().all()\n"
        )
        result = parse_dq_annotations(code)
        assert result == {
            "orders": ["len(df) > 0"],
            "customers": ["df['id'].notna().all()"],
        }

    def test_multiple_constraints_same_table(self):
        from constat.execution.executor import parse_dq_annotations
        code = (
            "# @dq[orders]: len(df) > 0\n"
            "# @dq[orders]: df['amount'].notna().all()\n"
        )
        result = parse_dq_annotations(code)
        assert result == {"orders": ["len(df) > 0", "df['amount'].notna().all()"]}

    def test_inline_comment_not_matched(self):
        from constat.execution.executor import parse_dq_annotations
        code = "df = df[df.active]  # @dq[orders]: len(df) > 0"
        # inline (non-leading) comments should not match
        assert parse_dq_annotations(code) == {}

    def test_intermediate_table_annotation(self):
        from constat.execution.executor import parse_dq_annotations
        code = "# @dq[_joined_raw]: len(df) > 0"
        assert parse_dq_annotations(code) == {"_joined_raw": ["len(df) > 0"]}


class TestRunDqConstraints:
    def test_passing_constraint(self):
        from constat.execution.executor import run_dq_constraints
        df = pd.DataFrame({"id": [1, 2]})
        results = run_dq_constraints(["len(df) > 0"], "t", {"df": df})
        assert results[0]["passed"] is True

    def test_failing_constraint(self):
        from constat.execution.executor import run_dq_constraints
        df = pd.DataFrame({"id": []})
        results = run_dq_constraints(["len(df) > 0"], "t", {"df": df})
        assert results[0]["passed"] is False

    def test_error_in_constraint(self):
        from constat.execution.executor import run_dq_constraints
        results = run_dq_constraints(["undefined_var > 0"], "t", {})
        assert results[0]["passed"] is False
        assert "error" in results[0]

    def test_multiple_constraints_mixed(self):
        from constat.execution.executor import run_dq_constraints
        df = pd.DataFrame({"id": [1, None]})
        results = run_dq_constraints(
            ["len(df) > 0", "df['id'].notna().all()"],
            "t",
            {"df": df},
        )
        assert results[0]["passed"] is True
        assert results[1]["passed"] is False
