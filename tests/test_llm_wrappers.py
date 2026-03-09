# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for list-returning LLM primitive wrappers."""

from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# llm_map
# ---------------------------------------------------------------------------

class TestLlmMapWrapper:

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_dedup_and_expand(self, mock_raw):
        """Duplicates are deduped for raw call, then expanded back."""
        mock_raw.return_value = {"a": "X", "b": "Y"}
        from constat.llm.wrappers import llm_map

        result = llm_map(["a", "b", "a"], ["X", "Y"], "src", "tgt")
        mock_raw.assert_called_once_with(["a", "b"], ["X", "Y"], "src", "tgt", reason=False, score=False)
        assert result == ["X", "Y", "X"]

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_default_returns_list_str(self, mock_raw):
        mock_raw.return_value = {"cat": "feline", "dog": "canine"}
        from constat.llm.wrappers import llm_map

        result = llm_map(["cat", "dog"], ["feline", "canine"], "animals")
        assert result == ["feline", "canine"]
        assert all(isinstance(v, str) for v in result)

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_reason_returns_list_dict(self, mock_raw):
        mock_raw.return_value = {
            "cat": {"value": "feline", "reason": "is a cat", "score": 0.9},
        }
        from constat.llm.wrappers import llm_map

        result = llm_map(["cat"], ["feline"], "animals", reason=True)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["value"] == "feline"

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_empty_input(self, mock_raw):
        from constat.llm.wrappers import llm_map

        result = llm_map([], ["X"], "src")
        assert result == []
        mock_raw.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_single_element(self, mock_raw):
        mock_raw.return_value = {"hello": "world"}
        from constat.llm.wrappers import llm_map

        result = llm_map(["hello"], ["world"], "src")
        assert result == ["world"]

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_str_coercion(self, mock_raw):
        """Non-string values are coerced to str for key safety."""
        mock_raw.return_value = {"1": "one", "2": "two"}
        from constat.llm.wrappers import llm_map

        result = llm_map([1, 2, 1], ["one", "two"], "numbers")
        mock_raw.assert_called_once_with(["1", "2"], ["one", "two"], "numbers", "", reason=False, score=False)
        assert result == ["one", "two", "one"]


# ---------------------------------------------------------------------------
# llm_classify
# ---------------------------------------------------------------------------

class TestLlmClassifyWrapper:

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_dedup_and_expand(self, mock_raw):
        mock_raw.return_value = {"a": "pos", "b": "neg"}
        from constat.llm.wrappers import llm_classify

        result = llm_classify(["a", "b", "a"], ["pos", "neg"], "ctx")
        mock_raw.assert_called_once_with(["a", "b"], ["pos", "neg"], "ctx", reason=False, score=False)
        assert result == ["pos", "neg", "pos"]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_default_returns_list_with_none(self, mock_raw):
        mock_raw.return_value = {"x": "bug", "y": None}
        from constat.llm.wrappers import llm_classify

        result = llm_classify(["x", "y"], ["bug", "feature"])
        assert result == ["bug", None]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_empty_input(self, mock_raw):
        from constat.llm.wrappers import llm_classify

        result = llm_classify([], ["a"])
        assert result == []
        mock_raw.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_single_element(self, mock_raw):
        mock_raw.return_value = {"hello": "greeting"}
        from constat.llm.wrappers import llm_classify

        result = llm_classify(["hello"], ["greeting", "farewell"])
        assert result == ["greeting"]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_reason_returns_list_dict(self, mock_raw):
        mock_raw.return_value = {
            "x": {"value": "bug", "reason": "looks like a bug", "score": 0.8},
        }
        from constat.llm.wrappers import llm_classify

        result = llm_classify(["x"], ["bug", "feature"], reason=True)
        assert isinstance(result[0], dict)
        assert result[0]["value"] == "bug"


# ---------------------------------------------------------------------------
# llm_score
# ---------------------------------------------------------------------------

class TestLlmScoreWrapper:

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_dedup_and_expand(self, mock_raw):
        mock_raw.return_value = [(0.5, "ok"), (0.9, "great")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["a", "b", "a"])
        mock_raw.assert_called_once_with(["a", "b"], 0.0, 1.0, "Rate each text")
        assert result == [0.5, 0.9, 0.5]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_default_returns_list_float(self, mock_raw):
        mock_raw.return_value = [(3.0, "high"), (1.0, "low")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["good", "bad"], min_val=0.0, max_val=5.0)
        assert result == [3.0, 1.0]
        assert all(isinstance(v, float) for v in result)

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_none_score(self, mock_raw):
        mock_raw.return_value = [(None, "cannot score")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["unclear"])
        assert result == [None]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_reason_returns_list_dict(self, mock_raw):
        mock_raw.return_value = [(0.8, "well written")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["text"], reason=True)
        assert len(result) == 1
        assert result[0] == {"score": 0.8, "reasoning": "well written"}

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_empty_input(self, mock_raw):
        from constat.llm.wrappers import llm_score

        result = llm_score([])
        assert result == []
        mock_raw.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_single_element(self, mock_raw):
        mock_raw.return_value = [(0.7, "decent")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["one"])
        assert result == [0.7]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_reason_with_duplicates(self, mock_raw):
        mock_raw.return_value = [(0.5, "mid")]
        from constat.llm.wrappers import llm_score

        result = llm_score(["same", "same", "same"], reason=True)
        assert len(result) == 3
        assert all(r == {"score": 0.5, "reasoning": "mid"} for r in result)
