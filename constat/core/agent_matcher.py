# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Embedding-based agent matching for dynamic agent selection.

Matches user queries to agents based on semantic similarity between
the query and agent descriptions. Uses the same embedding model as
intent classification for consistency.

The matcher:
1. Encodes agent descriptions at initialization
2. For each query, computes similarity to all agent descriptions
3. Returns the best matching agent if above threshold, else None
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from constat.core.agents import Agent, AgentManager
from constat.embedding_loader import EmbeddingModelLoader

logger = logging.getLogger(__name__)

# Similarity threshold for agent matching
# Lower than intent classification (0.80) since agent descriptions are broader
AGENT_MATCH_THRESHOLD = 0.45


@dataclass
class AgentMatch:
    """Result of agent matching."""
    agent: Agent
    similarity: float

    @property
    def name(self) -> str:
        return self.agent.name


class AgentMatcher:
    """Matches user queries to agents using embedding similarity.

    Uses BAAI/bge-large-en-v1.5 model (same as ConceptDetector and IntentClassifier)
    for semantic matching between queries and agent descriptions.

    Example:
        matcher = AgentMatcher(agent_manager)
        matcher.initialize()

        match = matcher.match("analyze quarterly revenue trends")
        if match:
            print(f"Matched agent: {match.agent.name} ({match.similarity:.2f})")
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        threshold: float = AGENT_MATCH_THRESHOLD,
    ):
        """Initialize the agent matcher.

        Args:
            agent_manager: AgentManager instance with loaded agents
            threshold: Minimum cosine similarity to match an agent (default 0.45)
        """
        self._agent_manager = agent_manager
        self._threshold = threshold

        # Lazy-loaded model and embeddings
        self._model: Optional[object] = None
        self._agent_embeddings: Optional[dict[str, np.ndarray]] = None
        self._initialized = False

    def initialize(self) -> None:
        """Precompute embeddings for all agent descriptions.

        Called lazily on first match. Embeddings are cached for fast matching.
        """
        if self._initialized:
            return

        # Use shared embedding model loader
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Compute embeddings for each agent's description
        self._agent_embeddings = {}

        for agent_name in self._agent_manager.list_agents():
            agent = self._agent_manager.get_agent(agent_name)
            if not agent:
                continue

            # Use description if available, otherwise use first line of prompt
            text_to_embed = agent.description if agent.description else agent.prompt.split('\n')[0]

            if not text_to_embed.strip():
                logger.warning(f"Agent '{agent_name}' has no description or prompt, skipping")
                continue

            # noinspection PyUnresolvedReferences
            embedding = self._model.encode(
                text_to_embed,
                normalize_embeddings=True,
            )
            self._agent_embeddings[agent_name] = embedding

        logger.info(f"AgentMatcher initialized with {len(self._agent_embeddings)} agents")
        self._initialized = True

    def reload(self) -> None:
        """Reload agent embeddings after agent changes."""
        self._initialized = False
        self._agent_embeddings = None
        self.initialize()

    def match(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> Optional[AgentMatch]:
        """Match a query to the best-fitting agent.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold

        Returns:
            AgentMatch if an agent matches above threshold, None otherwise
        """
        if not self._initialized:
            self.initialize()

        if not self._agent_embeddings:
            return None

        threshold = threshold if threshold is not None else self._threshold

        # Encode the query
        # noinspection PyUnresolvedReferences
        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )

        # Find best matching agent
        best_agent: Optional[Agent] = None
        best_similarity = 0.0

        for agent_name, agent_embedding in self._agent_embeddings.items():
            # Cosine similarity (embeddings are normalized, so dot product works)
            similarity = float(np.dot(agent_embedding, query_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_agent = self._agent_manager.get_agent(agent_name)

        # Check threshold
        if best_agent is None or best_similarity < threshold:
            logger.debug(
                f"No agent matched for query (best: {best_similarity:.2f}, threshold: {threshold})"
            )
            return None

        logger.info(f"Matched agent '{best_agent.name}' with similarity {best_similarity:.2f}")
        return AgentMatch(agent=best_agent, similarity=best_similarity)

    def match_all(
        self,
        query: str,
        threshold: Optional[float] = None,
    ) -> list[AgentMatch]:
        """Get all agents that match above threshold, sorted by similarity.

        Useful for debugging or showing alternatives to the user.

        Args:
            query: User's natural language query
            threshold: Optional override for similarity threshold

        Returns:
            List of AgentMatch objects sorted by similarity (highest first)
        """
        if not self._initialized:
            self.initialize()

        if not self._agent_embeddings:
            return []

        threshold = threshold if threshold is not None else self._threshold

        # Encode the query
        # noinspection PyUnresolvedReferences
        query_embedding = self._model.encode(
            query,
            normalize_embeddings=True,
        )

        # Compute similarities for all agents
        matches = []
        for agent_name, agent_embedding in self._agent_embeddings.items():
            similarity = float(np.dot(agent_embedding, query_embedding))

            if similarity >= threshold:
                agent = self._agent_manager.get_agent(agent_name)
                if agent:
                    matches.append(AgentMatch(agent=agent, similarity=similarity))

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
    def agent_count(self) -> int:
        """Get the number of agents with embeddings."""
        return len(self._agent_embeddings) if self._agent_embeddings else 0
