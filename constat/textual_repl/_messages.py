# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Textual Message subclasses for inter-widget communication."""

from __future__ import annotations

from textual.message import Message


class ShowApprovalUI(Message):
    """Message to trigger showing the approval UI on the main thread."""
    pass


class ShowClarificationUI(Message):
    """Message to trigger showing the clarification UI on the main thread."""
    pass


class SolveComplete(Message):
    """Message posted when solve operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class ProveComplete(Message):
    """Message posted when prove operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class ConsolidateComplete(Message):
    """Message posted when consolidate operation completes."""
    def __init__(self, result: dict) -> None:
        self.result = result
        super().__init__()


class DocumentAddComplete(Message):
    """Message posted when document addition completes."""
    def __init__(self, success: bool, message: str) -> None:
        self.success = success
        self.message = message
        super().__init__()


class SessionEvent(Message):
    """Message posted when a session event occurs (proof tree, steps, etc.)."""
    def __init__(self, event) -> None:
        self.event = event
        super().__init__()
