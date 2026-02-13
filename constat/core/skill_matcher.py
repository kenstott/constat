# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Embedding-based skill matching for dynamic skill selection.

Matches user queries to skills based on semantic similarity between
the query and skill descriptions. Follows the Agent Skills standard
where skills are automatically invoked when relevant to the query.

The matcher:
1. Encodes skill descriptions at initialization
2. For each query, computes similarity to all skill descriptions
3. Returns all matching skills above threshold (can be multiple)
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from constat.core.skills import Skill, SkillManager
from constat.embedding_loader import EmbeddingModelLoader

logger = logging.getLogger(__name__)

# Similarity threshold for skill matching
# Skills can have more specific descriptions, so threshold is slightly higher
SKILL_MATCH_THRESHOLD = 0.50


@dataclass
class SkillMatch:
    """Result of skill matching."""
    skill: Skill
    similarity: float

    @property
    def name(self) -> str:
        return self.skill.name


class SkillMatcher:
    """Matches user queries to skills using embedding similarity.

    Uses BAAI/bge-large-en-v1.5 model for semantic matching between
    queries and skill descriptions.

    Unlike roles (single selection), skills can return multiple matches
    since multiple skills may be relevant to a single query.

    Example:
        matcher = SkillMatcher(skill_manager)
        matcher.initialize()

        matches = matcher.match("analyze quarterly revenue trends")
        for match in matches:
            print(f"Matched skill: {match.skill.name} ({match.similarity:.2f})")
    """

    def __init__(
        self,
        skill_manager: SkillManager,
        threshold: float = SKILL_MATCH_THRESHOLD,
        max_skills: int = 3,
    ):
        """Initialize the skill matcher.

        Args:
            skill_manager: SkillManager instance with loaded skills
            threshold: Minimum cosine similarity to match a skill (default 0.50)
            max_skills: Maximum number of skills to return (default 3)
        """
        self._skill_manager = skill_manager
        self._threshold = threshold
        self._max_skills = max_skills

        # Lazy-loaded model and embeddings
        self._model: Optional[object] = None
        self._skill_embeddings: Optional[dict[str, np.ndarray]] = None
        self._initialized = False

    def initialize(self) -> None:
        """Precompute embeddings for all skill descriptions.

        Called lazily on first match. Embeddings are cached for fast matching.
        """
        if self._initialized:
            return

        # Use shared embedding model loader
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Compute embeddings for each skill's description
        self._skill_embeddings = {}

        for skill in self._skill_manager.get_all_skills():
            # Skip skills that have disabled model invocation
            if skill.disable_model_invocation:
                logger.debug(f"Skill '{skill.name}' has disable_model_invocation=True, skipping")
                continue

            # Use description for matching
            text_to_embed = skill.description
            if not text_to_embed.strip():
                logger.warning(f"Skill '{skill.name}' has no description, skipping")
                continue

            embedding = self._model.encode(
                text_to_embed,
                normalize_embeddings=True,
            )
            self._skill_embeddings[skill.name] = embedding

        logger.info(f"SkillMatcher initialized with {len(self._skill_embeddings)} skills")
        self._initialized = True

    def reload(self) -> None:
        """Reload skill embeddings after skill changes."""
        self._initialized = False
        self._skill_embeddings = None
        self.initialize()

    def match(
        self,
        query: str,
        threshold: Optional[float] = None,
        max_skills: Optional[int] = None,
    ) -> list[SkillMatch]:
        """Match a query to relevant skills.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold
            max_skills: Optional override for max skills to return

        Returns:
            List of SkillMatch objects sorted by similarity (highest first)
        """
        if not self._initialized:
            self.initialize()

        if not self._skill_embeddings:
            return []

        threshold = threshold if threshold is not None else self._threshold
        max_skills = max_skills if max_skills is not None else self._max_skills

        # Encode the query
        query_embedding = self._model.encode(  # noinspection PyUnresolvedReferences
            query,
            normalize_embeddings=True,
        )

        # Find all matching skills above threshold
        matches = []
        for skill_name, skill_embedding in self._skill_embeddings.items():
            # Cosine similarity (embeddings are normalized, so dot product works)
            similarity = float(np.dot(skill_embedding, query_embedding))

            if similarity >= threshold:
                skill = self._skill_manager.get_skill(skill_name)
                if skill:
                    matches.append(SkillMatch(skill=skill, similarity=similarity))

        # Sort by similarity descending and limit
        matches.sort(key=lambda x: x.similarity, reverse=True)
        matches = matches[:max_skills]

        if matches:
            skill_names = [m.name for m in matches]
            logger.info(f"Matched skills: {skill_names}")
        else:
            logger.debug(f"No skills matched for query (threshold: {threshold})")

        return matches

    def get_combined_prompt(
        self,
        query: str,
        threshold: Optional[float] = None,
        max_skills: Optional[int] = None,
    ) -> str:
        """Get combined prompt content from all matching skills.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold
            max_skills: Optional override for max skills

        Returns:
            Combined prompt content from matching skills, or empty string
        """
        matches = self.match(query, threshold, max_skills)

        if not matches:
            return ""

        prompts = []
        for match in matches:
            prompts.append(f"## {match.skill.name}\n{match.skill.prompt}")

        return "\n\n".join(prompts)

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
    def skill_count(self) -> int:
        """Get the number of skills with embeddings."""
        return len(self._skill_embeddings) if self._skill_embeddings else 0
