# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Learning and corrections system for Constat.

This package provides:
- LearningCompactor: Compacts raw learnings into rules
- FineTuneRegistry: YAML-backed model registry
- FineTuneManager: Fine-tune lifecycle orchestration
"""

from constat.learning.compactor import LearningCompactor, CompactionResult
from constat.learning.exemplar_generator import ExemplarGenerator, ExemplarResult
from constat.learning.fine_tune_registry import FineTuneRegistry, FineTunedModel
from constat.learning.fine_tune_manager import FineTuneManager

__all__ = [
    "LearningCompactor", "CompactionResult",
    "ExemplarGenerator", "ExemplarResult",
    "FineTuneRegistry", "FineTunedModel", "FineTuneManager",
]
