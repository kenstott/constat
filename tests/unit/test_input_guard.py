# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for build_input_guard."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

pytestmark = pytest.mark.unit


def _make_store(tables: dict[str, list[dict]]):
    """Mock store where load_dataframe returns a mock df with len = row count."""
    import pandas as pd
    store = MagicMock()
    def load(name):
        if name not in tables:
            return None
        return pd.DataFrame(tables[name])
    store.load_dataframe.side_effect = load
    return store


class TestBuildInputGuard:
    def test_empty_expected_inputs(self):
        from constat.execution.executor import build_input_guard
        assert build_input_guard([], {"orders"}) == ""

    def test_inputs_not_in_tables(self):
        from constat.execution.executor import build_input_guard
        # scalar state keys not in table registry — guard skipped
        assert build_input_guard(["start_date", "threshold"], set()) == ""

    def test_guard_skips_unknown_tables(self):
        from constat.execution.executor import build_input_guard
        guard = build_input_guard(["orders", "unknown"], {"orders"})
        assert "orders" in guard
        assert "unknown" not in guard

    def test_guard_raises_on_missing_table(self):
        from constat.execution.executor import build_input_guard
        guard = build_input_guard(["orders"], {"orders"})
        store = _make_store({})  # orders not in store
        ns = {"store": store}
        with pytest.raises(AssertionError, match="Required input missing"):
            exec(guard, ns)

    def test_guard_raises_on_empty_table(self):
        from constat.execution.executor import build_input_guard
        guard = build_input_guard(["orders"], {"orders"})
        store = _make_store({"orders": []})  # empty
        ns = {"store": store}
        with pytest.raises(AssertionError, match="Required input is empty"):
            exec(guard, ns)

    def test_guard_passes_on_non_empty_table(self):
        from constat.execution.executor import build_input_guard
        guard = build_input_guard(["orders"], {"orders"})
        store = _make_store({"orders": [{"id": 1}]})
        ns = {"store": store}
        exec(guard, ns)  # must not raise

    def test_guard_checks_all_declared_inputs(self):
        from constat.execution.executor import build_input_guard
        guard = build_input_guard(["orders", "customers"], {"orders", "customers"})
        assert "orders" in guard
        assert "customers" in guard
