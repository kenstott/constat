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
"""

from constat.learning.compactor import LearningCompactor, CompactionResult
from constat.learning.exemplar_generator import ExemplarGenerator, ExemplarResult

__all__ = ["LearningCompactor", "CompactionResult", "ExemplarGenerator", "ExemplarResult"]
