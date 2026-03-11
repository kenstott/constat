# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for forgiving LLM primitive wrappers.

Each wrapper must handle: str, list, Series, ndarray, single value, empty,
duplicates, int/float coercion — and always return the simplest type.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# llm_map
# ---------------------------------------------------------------------------

class TestLlmMap:

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_list_dedup_expand(self, mock):
        mock.return_value = {"a": "X", "b": "Y"}
        from constat.llm.wrappers import llm_map
        assert llm_map(["a", "b", "a"], ["X", "Y"], "s", "t") == ["X", "Y", "X"]
        mock.assert_called_once_with(["a", "b"], ["X", "Y"], "s", "t")

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_single_str_returns_scalar(self, mock):
        mock.return_value = {"hello": "world"}
        from constat.llm.wrappers import llm_map
        assert llm_map("hello", ["world"]) == "world"

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_single_element_list_returns_scalar(self, mock):
        mock.return_value = {"x": "y"}
        from constat.llm.wrappers import llm_map
        assert llm_map(["x"], ["y"]) == "y"

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_series_input(self, mock):
        mock.return_value = {"cat": "feline", "dog": "canine"}
        from constat.llm.wrappers import llm_map
        s = pd.Series(["cat", "dog", "cat"])
        assert llm_map(s, ["feline", "canine"]) == ["feline", "canine", "feline"]

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_ndarray_input(self, mock):
        mock.return_value = {"1": "one", "2": "two"}
        from constat.llm.wrappers import llm_map
        arr = np.array([1, 2, 1])
        assert llm_map(arr, ["one", "two"]) == ["one", "two", "one"]

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_int_coercion(self, mock):
        mock.return_value = {"1": "one", "2": "two"}
        from constat.llm.wrappers import llm_map
        assert llm_map([1, 2], ["one", "two"]) == ["one", "two"]

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_empty(self, mock):
        from constat.llm.wrappers import llm_map
        assert llm_map([], ["X"]) == []
        mock.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_map")
    def test_reason_kwarg_ignored(self, mock):
        """reason=True is silently ignored — always returns str."""
        mock.return_value = {"a": "X"}
        from constat.llm.wrappers import llm_map
        result = llm_map(["a", "a"], ["X"], reason=True)
        assert result == ["X", "X"]
        assert all(isinstance(v, str) for v in result)

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_allow_none_routes_to_classify(self, mock):
        mock.return_value = {"a": "pos", "b": None}
        from constat.llm.wrappers import llm_map
        result = llm_map(["a", "b"], ["pos", "neg"], allow_none=True)
        assert result == ["pos", None]
        mock.assert_called_once()

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_allow_none_scalar(self, mock):
        mock.return_value = {"x": "bug"}
        from constat.llm.wrappers import llm_map
        assert llm_map("x", ["bug", "feature"], allow_none=True) == "bug"

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_allow_none_dedup(self, mock):
        mock.return_value = {"a": "pos", "b": "neg"}
        from constat.llm.wrappers import llm_map
        assert llm_map(["a", "b", "a"], ["pos", "neg"], allow_none=True) == ["pos", "neg", "pos"]


# ---------------------------------------------------------------------------
# llm_classify (backwards-compat alias)
# ---------------------------------------------------------------------------

class TestLlmClassify:

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_list_dedup_expand(self, mock):
        mock.return_value = {"a": "pos", "b": "neg"}
        from constat.llm.wrappers import llm_classify
        assert llm_classify(["a", "b", "a"], ["pos", "neg"], "ctx") == ["pos", "neg", "pos"]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_single_str_returns_scalar(self, mock):
        mock.return_value = {"hi": "greeting"}
        from constat.llm.wrappers import llm_classify
        assert llm_classify("hi", ["greeting", "farewell"]) == "greeting"

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_single_element_list_returns_scalar(self, mock):
        mock.return_value = {"x": "bug"}
        from constat.llm.wrappers import llm_classify
        assert llm_classify(["x"], ["bug", "feature"]) == "bug"

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_none_for_unclassifiable(self, mock):
        mock.return_value = {"x": "bug", "y": None}
        from constat.llm.wrappers import llm_classify
        assert llm_classify(["x", "y"], ["bug"]) == ["bug", None]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_series_input(self, mock):
        mock.return_value = {"a": "pos", "b": "neg"}
        from constat.llm.wrappers import llm_classify
        s = pd.Series(["a", "b"])
        assert llm_classify(s, ["pos", "neg"]) == ["pos", "neg"]

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_empty(self, mock):
        from constat.llm.wrappers import llm_classify
        assert llm_classify([], ["a"]) == []
        mock.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_classify")
    def test_reason_kwarg_ignored(self, mock):
        mock.return_value = {"a": "pos"}
        from constat.llm.wrappers import llm_classify
        result = llm_classify(["a", "a"], ["pos"], reason=True, score=True)
        assert result == ["pos", "pos"]


# ---------------------------------------------------------------------------
# llm_score
# ---------------------------------------------------------------------------

class TestLlmScore:

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_list_dedup_expand(self, mock):
        mock.return_value = [(0.5, "ok"), (0.9, "great")]
        from constat.llm.wrappers import llm_score
        assert llm_score(["a", "b", "a"]) == [0.5, 0.9, 0.5]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_single_str_returns_scalar(self, mock):
        mock.return_value = [(0.8, "good")]
        from constat.llm.wrappers import llm_score
        assert llm_score("hello", 0, 1, "Rate") == 0.8

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_single_element_list_returns_scalar(self, mock):
        mock.return_value = [(0.7, "decent")]
        from constat.llm.wrappers import llm_score
        assert llm_score(["one"]) == 0.7

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_single_none_returns_none(self, mock):
        mock.return_value = [(None, "cannot score")]
        from constat.llm.wrappers import llm_score
        assert llm_score(["unclear"]) is None

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_series_input(self, mock):
        mock.return_value = [(0.3, "low"), (0.9, "high")]
        from constat.llm.wrappers import llm_score
        s = pd.Series(["bad", "good"])
        assert llm_score(s, 0, 1, "Rate") == [0.3, 0.9]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_ndarray_input(self, mock):
        mock.return_value = [(0.5, "mid")]
        from constat.llm.wrappers import llm_score
        arr = np.array(["same", "same", "same"])
        assert llm_score(arr) == [0.5, 0.5, 0.5]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_empty(self, mock):
        from constat.llm.wrappers import llm_score
        assert llm_score([]) == []
        mock.assert_not_called()

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_reason_kwarg_ignored(self, mock):
        """reason=True silently ignored — always returns float."""
        mock.return_value = [(0.8, "good"), (0.2, "bad")]
        from constat.llm.wrappers import llm_score
        result = llm_score(["x", "y"], reason=True)
        assert result == [0.8, 0.2]
        assert all(isinstance(v, float) for v in result)

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_apply_pattern_produces_floats(self, mock):
        """The .apply() anti-pattern produces usable float values."""
        mock.return_value = [(0.8, "positive")]
        from constat.llm.wrappers import llm_score
        df = pd.DataFrame({"text": ["hello", "world", "test"]})
        df["score"] = df["text"].apply(lambda x: llm_score([x], 0, 1, "Rate"))
        assert df["score"].dtype == float
        assert list(df["score"]) == [0.8, 0.8, 0.8]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_apply_str_pattern_produces_floats(self, mock):
        """Calling with bare string in .apply() also works."""
        mock.return_value = [(0.6, "neutral")]
        from constat.llm.wrappers import llm_score
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        df["score"] = df["text"].apply(lambda x: llm_score(x, 0, 1, "Rate"))
        assert df["score"].dtype == float

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_explain_list(self, mock):
        mock.return_value = [(0.5, "ok"), (0.9, "great")]
        from constat.llm.wrappers import llm_score
        result = llm_score(["a", "b", "a"], explain=True)
        assert result == [(0.5, "ok"), (0.9, "great"), (0.5, "ok")]

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_explain_scalar(self, mock):
        mock.return_value = [(0.8, "good")]
        from constat.llm.wrappers import llm_score
        result = llm_score("hello", 0, 1, "Rate", explain=True)
        assert result == (0.8, "good")

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_explain_none_score(self, mock):
        mock.return_value = [(None, "cannot score")]
        from constat.llm.wrappers import llm_score
        result = llm_score(["unclear"], explain=True)
        assert result == (None, "cannot score")  # single-element list → scalar

    @patch("constat.llm.wrappers._raw_llm_score")
    def test_explain_empty(self, mock):
        from constat.llm.wrappers import llm_score
        assert llm_score([], explain=True) == []
        mock.assert_not_called()
