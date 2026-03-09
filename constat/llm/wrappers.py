# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Forgiving LLM primitive wrappers.

Accept anything (str, list, Series, ndarray), deduplicate internally,
return the simplest possible type based on how they were called:
  - Single value in  → scalar out  (str / float / None)
  - Multiple values  → list out    (list[str] / list[float|None])

No reason/score kwargs — reasoning is captured in the event stream.
Generated code gets exactly one way to call each function.
"""

from __future__ import annotations

import logging

from constat.llm import (
    llm_map as _raw_llm_map,
    llm_classify as _raw_llm_classify,
    llm_score as _raw_llm_score,
)

logger = logging.getLogger(__name__)


def _to_str_list(values) -> tuple[list[str], bool]:
    """Normalize any input to list[str]. Returns (str_list, was_scalar)."""
    if isinstance(values, str):
        return [values], True
    # pandas Series / numpy ndarray
    if hasattr(values, 'tolist'):
        items = values.tolist()
        return [str(v) for v in items], len(items) == 1
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values], len(values) == 1
    # Single non-string scalar
    return [str(values)], True


def llm_map(values, allowed, source_desc="values", target_desc="", **_kw):
    """Map values to an allowed set. Always returns str (scalar) or list[str]."""
    str_values, scalar = _to_str_list(values)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_map(unique, allowed, source_desc, target_desc)
    result = [raw.get(v, raw.get(v)) for v in str_values]
    if scalar:
        return result[0]
    return result


def llm_classify(values, categories, context="", **_kw):
    """Classify values into categories. Always returns str|None (scalar) or list[str|None]."""
    str_values, scalar = _to_str_list(values)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_classify(unique, categories, context)
    result = [raw.get(v) for v in str_values]
    if scalar:
        return result[0]
    return result


def llm_score(texts, min_val=0.0, max_val=1.0, instruction="Rate each text", **_kw):
    """Score texts on a numeric scale. Always returns float|None (scalar) or list[float|None]."""
    str_values, scalar = _to_str_list(texts)
    if not str_values:
        return []
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_score(unique, min_val, max_val, instruction)
    score_map = dict(zip(unique, raw))
    # raw returns list[tuple[float|None, str]] — extract score only
    result = [score_map[v][0] for v in str_values]
    if scalar:
        return result[0]
    return result
