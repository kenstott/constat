# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Display override detection from natural language.

Detects user preferences for output formatting embedded in queries.
"""

import re

from constat.api.types import DisplayOverrides

# Display override patterns: (pattern, setting, value, is_persistent)
# Persistent changes are detected by "always", "from now on", "stop", "never"
# Single-turn overrides apply only to the current query
DISPLAY_OVERRIDE_PATTERNS: list[tuple[str, str, bool, bool]] = [
    # Persistent changes (detected by "always", "from now on", "stop", "never")
    (r"\b(always|from now on)\s+(show|display|include)\s+raw\b", "raw", True, True),
    (r"\b(stop|never|don't|do not)\s+(show|display|include)\s+raw\b", "raw", False, True),
    (r"\b(always|from now on)\s+(be\s+)?verbose\b", "verbose", True, True),
    (r"\b(stop|never|don't|do not)\s+(be\s+)?verbose\b", "verbose", False, True),
    (r"\b(always|from now on)\s+(show|give|provide)\s+(insights?|synthesis)\b", "insights", True, True),
    (r"\b(stop|never|don't|do not)\s+(show|give|provide)\s+(insights?|synthesis)\b", "insights", False, True),

    # Single-turn overrides (apply only to this query)
    (r"^(briefly|concisely|quick(ly)?)\b", "insights", False, False),
    (r"\b(with(out)?\s+)?raw\s+(output|results)\b", "raw", True, False),
    (r"\b(hide|no|skip|without)\s+raw\b", "raw", False, False),
    (r"\b(verbose(ly)?|in\s+detail|detailed)\b", "verbose", True, False),
    (r"\bjust\s+the\s+(answer|result)\b", "raw", False, False),
]


def detect_display_overrides(text: str) -> DisplayOverrides:
    """Detect display preference overrides in natural language.

    Scans the input text for patterns indicating user preferences for
    output formatting (raw output, verbose mode, insights).

    Args:
        text: User input text to analyze

    Returns:
        DisplayOverrides with persistent and single_turn settings detected
    """
    persistent: dict[str, bool] = {}
    single_turn: dict[str, bool] = {}
    text_lower = text.lower()

    for pattern, setting, value, is_persistent in DISPLAY_OVERRIDE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            if is_persistent:
                persistent[setting] = value
            else:
                single_turn[setting] = value

    return DisplayOverrides(
        persistent=persistent,
        single_turn=single_turn,
    )
