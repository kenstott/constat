# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Auto-compaction logic for learnings.

Automatically triggers learning compaction when thresholds are exceeded.
"""

import logging
from typing import TYPE_CHECKING, Optional

from constat.api.types import LearningCompactionResult
from constat.learning.compactor import LearningCompactor

if TYPE_CHECKING:
    from constat.session import Session
    from constat.storage.learnings import LearningStore

logger = logging.getLogger(__name__)


def maybe_auto_compact(
    session: "Session",
    learning_store: "LearningStore",
) -> Optional[LearningCompactionResult]:
    """Check if auto-compaction should trigger and run it.

    Auto-compaction triggers when the number of unpromoted learnings
    exceeds LearningCompactor.AUTO_COMPACT_THRESHOLD.

    Args:
        session: Session instance to get LLM provider from
        learning_store: LearningStore instance to compact

    Returns:
        LearningCompactionResult if compaction was run, None otherwise
    """
    stats = learning_store.get_stats()
    unpromoted = stats.get("unpromoted", 0)

    if unpromoted < LearningCompactor.AUTO_COMPACT_THRESHOLD:
        return None

    logger.info(f"Auto-compacting {unpromoted} learnings...")

    try:
        # Get a model for compaction (use general task routing)
        model_spec = session.router.routing_config.get_models_for_task("general")[0]
        llm = session.router._get_provider(model_spec)
        compactor = LearningCompactor(learning_store, llm)
        result = compactor.compact()

        logger.info(
            f"Compaction complete: {result.rules_created} rules created, "
            f"{result.learnings_archived} learnings archived"
        )

        return LearningCompactionResult(
            rules_created=result.rules_created,
            rules_strengthened=result.rules_strengthened,
            rules_merged=result.rules_merged,
            learnings_archived=result.learnings_archived,
            learnings_expired=result.learnings_expired,
            groups_found=result.groups_found,
            skipped_low_confidence=result.skipped_low_confidence,
            errors=tuple(result.errors),
        )

    except Exception as e:
        logger.warning(f"Auto-compact failed: {e}")
        return None


def should_auto_compact(learning_store: "LearningStore") -> bool:
    """Check if auto-compaction should trigger.

    Args:
        learning_store: LearningStore instance to check

    Returns:
        True if unpromoted learnings exceed threshold
    """
    stats = learning_store.get_stats()
    return stats.get("unpromoted", 0) >= LearningCompactor.AUTO_COMPACT_THRESHOLD


def get_compaction_stats(learning_store: "LearningStore") -> dict:
    """Get statistics relevant to compaction.

    Args:
        learning_store: LearningStore instance

    Returns:
        Dict with unpromoted count, threshold, and should_compact flag
    """
    stats = learning_store.get_stats()
    unpromoted = stats.get("unpromoted", 0)
    threshold = LearningCompactor.AUTO_COMPACT_THRESHOLD

    return {
        "unpromoted": unpromoted,
        "threshold": threshold,
        "should_compact": unpromoted >= threshold,
        "total_learnings": stats.get("total_learnings", 0),
        "total_rules": stats.get("total_rules", 0),
    }
