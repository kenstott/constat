# Copyright (c) 2025 Kenneth Stott
# Canary: ec4e337d-b4ad-4dd8-b15f-6c8be0b85a48
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""REPL test suite entry point.

Tests are split by concern into:
  - test_repl_commands.py   — command parsing and individual command handlers
  - test_repl_execution.py  — solve, session lifecycle, initialization, cleanup
  - test_repl_display.py    — StatusLine, FeedbackDisplay, /remember output
"""

from __future__ import annotations
