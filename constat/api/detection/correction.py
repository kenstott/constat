# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Natural language correction detection.

Detects correction patterns in user input for learning capture.
"""

import re

from constat.api.types import CorrectionDetection

# Correction patterns: (pattern, correction_type)
# Detects corrections in natural conversation
CORRECTION_PATTERNS: list[tuple[str, str]] = [
    # Explicit wrong
    (r"\bthat'?s\s+(wrong|incorrect|not\s+right)\b", "explicit_wrong"),
    (r"\byou\s+(got|have)\s+it\s+wrong\b", "you_wrong"),
    (r"\bthat'?s\s+not\s+(how|what|correct)\b", "not_correct"),

    # "Actually" corrections
    (r"\bactually[,]?\s+(.+)\s+(means|is|should\s+be)\b", "actually_means"),
    (r"\bno[,]?\s+(.+)\s+(means|is|should\s+be)\b", "no_means"),

    # Domain terminology
    (r"\b(when\s+I\s+say|by)\s+['\"]?(\w+)['\"]?[,]?\s*I\s+mean\b", "i_mean"),
    (r"\bin\s+(our|this)\s+(context|company)[,]?\s+(\w+)\s+(means|refers)\b", "domain_term"),

    # Assumptions
    (r"\bdon'?t\s+assume\s+(.+)\b", "dont_assume"),
    (r"\bnever\s+assume\s+(.+)\b", "never_assume"),
]


def detect_nl_correction(text: str) -> CorrectionDetection:
    """Detect if user input contains a correction pattern.

    Scans the input text for patterns indicating the user is correcting
    a previous response or providing domain-specific terminology.

    Args:
        text: User input text to analyze

    Returns:
        CorrectionDetection with detected flag, correction_type, and matched_text
    """
    for pattern, correction_type in CORRECTION_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return CorrectionDetection(
                detected=True,
                correction_type=correction_type,
                matched_text=match.group(0),
            )

    return CorrectionDetection(detected=False)
