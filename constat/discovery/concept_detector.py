# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Embedding-based concept detection for conditional prompt injection.

Detects relevant prompt concepts in user queries using semantic similarity.
Uses BAAI/bge-large-en-v1.5 model for high-quality semantic matching.

Concepts are defined by exemplar sentences, not keywords. This approach is:
- More robust to paraphrasing and language variation
- Multilingual-capable with appropriate model
- Lower maintenance than keyword lists
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from constat.embedding_loader import EmbeddingModelLoader
from constat.execution.prompt_sections import (
    PROMPT_SECTIONS,
    PromptSection,
    PromptTarget,
)


@dataclass
class DetectedConcept:
    """A concept detected in the query with its similarity score."""

    concept_id: str
    """The unique identifier of the detected concept."""

    similarity: float
    """Cosine similarity score (0-1) of the best-matching exemplar."""

    section: PromptSection
    """The full PromptSection object for this concept."""


class ConceptDetector:
    """
    Detects relevant prompt concepts in user queries using embedding similarity.

    Uses BAAI/bge-large-en-v1.5 model for high-quality semantic matching.
    Concepts are defined by exemplar sentences, not keywords.

    The detector precomputes embeddings for all exemplars at initialization,
    making query-time detection fast (~5ms).

    Example:
        detector = ConceptDetector()
        detector.initialize()

        # Detect concepts for a query
        concepts = detector.detect("create a sales dashboard", target="step")
        # Returns: [DetectedConcept(concept_id='dashboard_layout', similarity=0.82, ...)]

        # Get injectable content.
        content = detector.get_sections_for_prompt("create a dashboard", "step")
    """

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    DEFAULT_THRESHOLD = 0.55

    def __init__(
        self,
        model: Optional["SentenceTransformer"] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """Initialize the concept detector.

        Args:
            model: Optional preloaded SentenceTransformer model.
                   If None, loads the model on first use.
            threshold: Minimum cosine similarity to consider a concept detected.
                      Default 0.55 balances precision and recall.
        """
        self._model = model
        self._threshold = threshold

        # Precomputed embeddings: (n_total_exemplars, 384)
        self._exemplar_embeddings: Optional[np.ndarray] = None
        # Maps exemplar index -> concept_id
        self._exemplar_to_concept: list[str] = []

        self._initialized = False

    def initialize(self) -> None:
        """Precompute embeddings for all concept exemplars.

        Called once at startup. Embeddings are cached for fast query-time detection.
        This takes ~100ms on first call (model loading + encoding).
        """
        if self._initialized:
            return

        # Use shared embedding model loader (may already be loaded)
        if self._model is None:
            self._model = EmbeddingModelLoader.get_instance().get_model()

        # Collect all exemplars
        all_exemplars = []
        self._exemplar_to_concept = []

        for concept_id, section in PROMPT_SECTIONS.items():
            for exemplar in section.exemplars:
                all_exemplars.append(exemplar)
                self._exemplar_to_concept.append(concept_id)

        # Compute embeddings in batch
        if all_exemplars:
            self._exemplar_embeddings = self._model.encode(
                all_exemplars,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Pre-normalize for cosine similarity
            )

        self._initialized = True

    def detect(
        self,
        query: str,
        target: Optional[PromptTarget] = None,
        threshold: Optional[float] = None,
    ) -> list[DetectedConcept]:
        """Detect relevant concepts in a query.

        Args:
            query: The user's query or step goal
            target: Optional filter to only return concepts for a specific prompt
                   ('engine', 'planner', or 'step')
            threshold: Optional override for similarity threshold

        Returns:
            List of DetectedConcept objects, sorted by similarity (highest first)
        """
        if not self._initialized:
            self.initialize()

        if self._exemplar_embeddings is None or len(self._exemplar_to_concept) == 0:
            return []

        threshold = threshold if threshold is not None else self._threshold

        # Embed the query
        query_embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Compute similarities (dot product since normalized = cosine similarity)
        similarities = np.dot(self._exemplar_embeddings, query_embedding.T).flatten()

        # Find concepts that exceed threshold (take max similarity per concept)
        concept_max_sim: dict[str, float] = {}
        for idx, sim in enumerate(similarities):
            concept_id = self._exemplar_to_concept[idx]
            if sim >= threshold:
                if concept_id not in concept_max_sim or sim > concept_max_sim[concept_id]:
                    concept_max_sim[concept_id] = float(sim)

        # Build results, optionally filtering by target
        results = []
        for concept_id, sim in concept_max_sim.items():
            section = PROMPT_SECTIONS[concept_id]

            # Filter by target if specified
            if target and target not in section.targets:
                continue

            results.append(
                DetectedConcept(
                    concept_id=concept_id,
                    similarity=sim,
                    section=section,
                )
            )

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results

    def get_sections_for_prompt(
        self,
        query: str,
        target: PromptTarget,
        threshold: Optional[float] = None,
    ) -> str:
        """Get concatenated section content for a specific prompt type.

        This is the main entry point for prompt building. Returns all relevant
        section content joined together, ready for injection into the prompt.

        Args:
            query: The user's query
            target: The prompt type ('engine', 'planner', or 'step')
            threshold: Optional override for similarity threshold

        Returns:
            Concatenated section content string, or empty string if none detected
        """
        detected = self.detect(query, target=target, threshold=threshold)

        if not detected:
            return ""

        # Concatenate sections with clear separation
        sections = []
        for concept in detected:
            sections.append(concept.section.content.strip())

        return "\n\n".join(sections)

    @property
    def is_initialized(self) -> bool:
        """Check if the detector has been initialized."""
        return self._initialized

    @property
    def threshold(self) -> float:
        """Get the current similarity threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the similarity threshold."""
        self._threshold = value
