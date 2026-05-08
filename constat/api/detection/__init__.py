# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Detection utilities for natural language patterns."""

from constat.api.detection.correction import (
    CORRECTION_PATTERNS,
    detect_nl_correction,
)
from constat.api.detection.display import (
    DISPLAY_OVERRIDE_PATTERNS,
    detect_display_overrides,
)

__all__ = [
    "CORRECTION_PATTERNS",
    "DISPLAY_OVERRIDE_PATTERNS",
    "detect_display_overrides",
    "detect_nl_correction",
]
