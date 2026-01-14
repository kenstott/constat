"""Learning and corrections system for Constat.

This package provides:
- LearningCompactor: Compacts raw learnings into rules
"""

from constat.learning.compactor import LearningCompactor, CompactionResult

__all__ = ["LearningCompactor", "CompactionResult"]
