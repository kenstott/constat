# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unit tests for PlanError, detect_structural_failure, and _error_fingerprint."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestErrorFingerprint:
    def test_keyerror_with_column(self):
        from constat.execution.executor import _error_fingerprint
        exc, term = _error_fingerprint("KeyError: 'customer_segment'")
        assert exc == "KeyError"
        assert term == "customer_segment"

    def test_attributeerror(self):
        from constat.execution.executor import _error_fingerprint
        exc, term = _error_fingerprint("AttributeError: 'NoneType' object has no attribute 'merge'")
        assert exc == "AttributeError"

    def test_no_quoted_term(self):
        from constat.execution.executor import _error_fingerprint
        exc, term = _error_fingerprint("RuntimeError: something went wrong")
        assert exc == "RuntimeError"
        assert term == ""

    def test_unknown_error(self):
        from constat.execution.executor import _error_fingerprint
        exc, term = _error_fingerprint("something completely different")
        assert exc == "Error"


class TestDetectStructuralFailure:
    def test_no_detection_below_threshold(self):
        from constat.execution.executor import detect_structural_failure
        errors = [("KeyError: 'x'", "model-a"), ("KeyError: 'x'", "model-b")]
        assert detect_structural_failure(errors, threshold=3) is None

    def test_no_detection_single_model_same_error(self):
        from constat.execution.executor import detect_structural_failure
        # Same fingerprint but only one model — should escalate, not give up
        errors = [("KeyError: 'customer_segment'", "model-a")] * 3
        assert detect_structural_failure(errors, threshold=3) is None

    def test_detects_same_error_across_two_models(self):
        from constat.execution.executor import detect_structural_failure
        errors = [
            ("KeyError: 'customer_segment'", "model-a"),
            ("KeyError: 'customer_segment'", "model-a"),
            ("KeyError: 'customer_segment'", "model-b"),
        ]
        result = detect_structural_failure(errors, threshold=3)
        assert result is not None
        assert "KeyError" in result
        assert "customer_segment" in result
        assert "2 different models" in result

    def test_no_detection_varied_errors(self):
        from constat.execution.executor import detect_structural_failure
        errors = [
            ("KeyError: 'customer_segment'", "model-a"),
            ("TypeError: unsupported operand", "model-b"),
            ("KeyError: 'customer_segment'", "model-b"),
        ]
        assert detect_structural_failure(errors, threshold=3) is None

    def test_detects_from_tail(self):
        from constat.execution.executor import detect_structural_failure
        errors = [
            ("TypeError: something else", "model-a"),
            ("KeyError: 'col_x'", "model-a"),
            ("KeyError: 'col_x'", "model-a"),
            ("KeyError: 'col_x'", "model-b"),
        ]
        result = detect_structural_failure(errors, threshold=3)
        assert result is not None
        assert "col_x" in result

    def test_empty_history(self):
        from constat.execution.executor import detect_structural_failure
        assert detect_structural_failure([], threshold=3) is None


class TestPlanError:
    def test_plan_error_is_exception(self):
        from constat.execution.executor import PlanError
        with pytest.raises(PlanError, match="missing column"):
            raise PlanError("missing column")

    def test_executor_captures_plan_error(self):
        from constat.execution.executor import PlanError, PythonExecutor
        executor = PythonExecutor()
        result = executor.execute(
            "raise PlanError('orders has no segment column')",
            {"PlanError": PlanError},
        )
        assert result.success is False
        assert result.plan_error == "orders has no segment column"
        assert result.runtime_error is None  # not a runtime error

    def test_plan_error_not_retried_as_runtime_error(self):
        from constat.execution.executor import PlanError, PythonExecutor
        executor = PythonExecutor()
        result = executor.execute(
            "raise PlanError('structurally impossible')",
            {"PlanError": PlanError},
        )
        assert result.plan_error is not None
        assert result.runtime_error is None
