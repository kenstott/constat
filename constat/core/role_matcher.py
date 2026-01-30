# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Embedding-based role matching for dynamic role selection.

Matches user queries to roles based on semantic similarity between
the query and role descriptions. Uses the same embedding model as
intent classification for consistency.

The matcher:
1. Encodes role descriptions at initialization
2. For each query, computes similarity to all role descriptions
3. Returns the best matching role if above threshold, else None
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from constat.embedding_loader import EmbeddingModelLoader
from constat.core.roles import Role, RoleManager

logger = logging.getLogger(__name__)

# Similarity threshold for role matching
# Lower than intent classification (0.80) since role descriptions are broader
ROLE_MATCH_THRESHOLD = 0.45


@dataclass
class RoleMatch:
    """Result of role matching."""
    role: Role
    similarity: float

    @property
    def name(self) -> str:
        return self.role.name


class RoleMatcher:
    """Matches user queries to roles using embedding similarity.

    Uses BAAI/bge-large-en-v1.5 model (same as ConceptDetector and IntentClassifier)
    for semantic matching between queries and role descriptions.

    Example:
        matcher = RoleMatcher(role_manager)
        matcher.initialize()

        match = matcher.match("analyze quarterly revenue trends")
        if match:
            print(f"Matched role: {match.role.name} ({match.similarity:.2f})")
    """

    def __init__(
        self,
        role_manager: RoleManager,
        threshold: float = ROLE_MATCH_THRESHOLD,
    ):
        """Initialize the role matcher.

        Args:
            role_manager: RoleManager instance with loaded roles
            threshold: Minimum cosine similarity to match a role (default 0.45)
        """
        self._role_manager = role_manager
        self._threshold = threshold

        # Lazy-loaded model and embeddings
        self._model: Optional[object] = None
        self._role_embeddings: Optional[dict[str, np.ndarray]] = None
        self._initialized = False

    def initialize(self) -> None:
        """Precompute embeddings for all role descriptions.

        Called lazily on first match. Embeddings are cached for fast matching.
        """
        if self._initialized:
            return

        # Use shared embedding model loader
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Compute embeddings for each role's description
        self._role_embeddings = {}

        for role_name in self._role_manager.list_roles():
            role = self._role_manager.get_role(role_name)
            if not role:
                continue

            # Use description if available, otherwise use first line of prompt
            text_to_embed = role.description if role.description else role.prompt.split('\n')[0]

            if not text_to_embed.strip():
                logger.warning(f"Role '{role_name}' has no description or prompt, skipping")
                continue

            embedding = self._model.encode(
                text_to_embed,
                normalize_embeddings=True,
            )
            self._role_embeddings[role_name] = embedding

        logger.info(f"RoleMatcher initialized with {len(self._role_embeddings)} roles")
        self._initialized = True

    def reload(self) -> None:
        """Reload role embeddings after role changes."""
        self._initialized = False
        self._role_embeddings = None
        self.initialize()

    def match(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[RoleMatch]:
        """Match a query to the best-fitting role.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold

        Returns:
            RoleMatch if a role matches above threshold, None otherwise
        """
        if not self._initialized:
            self.initialize()

        if not self._role_embeddings:
            return None

        threshold = threshold if threshold is not None else self._threshold

        # Encode the query
        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )

        # Find best matching role
        best_role: Optional[Role] = None
        best_similarity = 0.0

        for role_name, role_embedding in self._role_embeddings.items():
            # Cosine similarity (embeddings are normalized, so dot product works)
            similarity = float(np.dot(role_embedding, query_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_role = self._role_manager.get_role(role_name)

        # Check threshold
        if best_role is None or best_similarity < threshold:
            logger.debug(
                f"No role matched for query (best: {best_similarity:.2f}, threshold: {threshold})"
            )
            return None

        logger.info(f"Matched role '{best_role.name}' with similarity {best_similarity:.2f}")
        return RoleMatch(role=best_role, similarity=best_similarity)

    def match_all(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> list[RoleMatch]:
        """Get all roles that match above threshold, sorted by similarity.

        Useful for debugging or showing alternatives to the user.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold

        Returns:
            List of RoleMatch objects sorted by similarity (highest first)
        """
        if not self._initialized:
            self.initialize()

        if not self._role_embeddings:
            return []

        threshold = threshold if threshold is not None else self._threshold

        # Encode the query
        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )

        # Compute similarities for all roles
        matches = []
        for role_name, role_embedding in self._role_embeddings.items():
            similarity = float(np.dot(role_embedding, query_embedding))

            if similarity >= threshold:
                role = self._role_manager.get_role(role_name)
                if role:
                    matches.append(RoleMatch(role=role, similarity=similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches

    @property
    def threshold(self) -> float:
        """Get the current similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the similarity threshold."""
        self._threshold = value

    @property
    def is_initialized(self) -> bool:
        """Check if the matcher has been initialized."""
        return self._initialized

    @property
    def role_count(self) -> int:
        """Get the number of roles with embeddings."""
        return len(self._role_embeddings) if self._role_embeddings else 0
