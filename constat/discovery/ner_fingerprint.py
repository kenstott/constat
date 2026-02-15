# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fingerprint caching for entity extraction.

Computes a hash of (chunk_ids + term_set) so that NER can be skipped
when the session's scope hasn't changed (same chunks, same entities).
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# In-memory cache: session_id → fingerprint
_session_fingerprints: dict[str, str] = {}


def compute_ner_fingerprint(
    chunk_ids: list[str],
    schema_terms: list[str],
    api_terms: list[str],
    business_terms: Optional[list[str]] = None,
) -> str:
    """Compute a fingerprint for the NER extraction scope.

    Args:
        chunk_ids: Sorted list of chunk IDs visible to the session
        schema_terms: Schema entity names
        api_terms: API entity names
        business_terms: Glossary/relationship terms

    Returns:
        Hex digest fingerprint
    """
    parts = [
        "|".join(sorted(chunk_ids)),
        "|".join(sorted(schema_terms)),
        "|".join(sorted(api_terms)),
        "|".join(sorted(business_terms or [])),
    ]
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def should_skip_ner(session_id: str, fingerprint: str) -> bool:
    """Check if NER can be skipped for this session (scope unchanged).

    Args:
        session_id: Session ID
        fingerprint: Current scope fingerprint

    Returns:
        True if the fingerprint matches (skip NER), False otherwise
    """
    cached = _session_fingerprints.get(session_id)
    if cached == fingerprint:
        logger.info(f"NER fingerprint cache hit for session {session_id} — skipping extraction")
        return True
    return False


def update_ner_fingerprint(session_id: str, fingerprint: str) -> None:
    """Store the NER fingerprint after successful extraction.

    Args:
        session_id: Session ID
        fingerprint: Fingerprint to cache
    """
    _session_fingerprints[session_id] = fingerprint
    logger.debug(f"Updated NER fingerprint for session {session_id}: {fingerprint}")


def invalidate_ner_fingerprint(session_id: str) -> None:
    """Invalidate the cached fingerprint (e.g., on scope change).

    Args:
        session_id: Session ID to invalidate
    """
    _session_fingerprints.pop(session_id, None)
    logger.debug(f"Invalidated NER fingerprint for session {session_id}")


def clear_all_fingerprints() -> None:
    """Clear all cached fingerprints."""
    _session_fingerprints.clear()
