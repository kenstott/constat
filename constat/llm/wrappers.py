# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""List-returning LLM primitive wrappers.

These wrappers accept full columns (with duplicates), deduplicate internally,
call the raw primitives on uniques, and return input-aligned lists.
This eliminates the .map() pattern that causes failures with smaller models.
"""

from __future__ import annotations

from constat.llm import (
    llm_map as _raw_llm_map,
    llm_classify as _raw_llm_classify,
    llm_score as _raw_llm_score,
)


def llm_map(
    values: list,
    allowed: list[str],
    source_desc: str = "values",
    target_desc: str = "",
    *,
    reason: bool = False,
    score: bool = False,
) -> list[str] | list[dict]:
    """Map values to an allowed set, returning an input-aligned list.

    Deduplicates internally — pass the full column (duplicates OK).

    Returns:
        list[str] by default — direct column assignment.
        list[dict] when reason or score is True, with keys "value", "reason", "score".
    """
    if not values:
        return []
    str_values = [str(v) for v in values]
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_map(unique, allowed, source_desc, target_desc, reason=reason, score=score)
    return [raw[v] for v in str_values]


def llm_classify(
    values: list,
    categories: list[str],
    context: str = "",
    *,
    reason: bool = False,
    score: bool = False,
) -> list[str | None] | list[dict]:
    """Classify values into categories, returning an input-aligned list.

    Deduplicates internally — pass the full column (duplicates OK).

    Returns:
        list[str | None] by default — None for unclassifiable.
        list[dict] when reason or score is True.
    """
    if not values:
        return []
    str_values = [str(v) for v in values]
    unique = list(dict.fromkeys(str_values))
    raw = _raw_llm_classify(unique, categories, context, reason=reason, score=score)
    return [raw.get(str(v)) for v in str_values]


def llm_score(
    texts: list[str],
    min_val: float = 0.0,
    max_val: float = 1.0,
    instruction: str = "Rate each text",
    *,
    reason: bool = False,
) -> list[float | None] | list[dict]:
    """Score texts on a numeric scale, returning an input-aligned list.

    Deduplicates internally — pass the full column (duplicates OK).

    Returns:
        list[float | None] by default — scores only, direct column assignment.
        list[dict] when reason=True, with keys "score", "reasoning".
    """
    if not texts:
        return []
    unique = list(dict.fromkeys(texts))
    raw = _raw_llm_score(unique, min_val, max_val, instruction)
    score_map = dict(zip(unique, raw))
    if reason:
        return [{"score": score_map[v][0], "reasoning": score_map[v][1]} for v in texts]
    return [score_map[v][0] for v in texts]
