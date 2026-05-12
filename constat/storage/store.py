# Copyright (c) 2025 Kenneth Stott
# Canary: 06234e90-b227-4ee3-b0ba-fab8485ef781
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Composed store that delegates to RelationalStore and DuckDBVectorBackend.

Cross-layer methods (vector search + relational entity lookup) live here.
"""

import logging

import numpy as np

from constat.discovery.models import ChunkEntity, EnrichedChunk
from constat.storage.duckdb_backend import DuckDBVectorBackend
from constat.storage.relational import RelationalStore

logger = logging.getLogger(__name__)


class Store:
    """Composes RelationalStore + DuckDBVectorBackend."""

    def __init__(self, relational: RelationalStore, vector: DuckDBVectorBackend):
        self.relational = relational
        self.vector = vector

    # ------------------------------------------------------------------
    # Cross-layer: vector search + entity enrichment
    # ------------------------------------------------------------------

    def search_enriched(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
        query_text: str | None = None,
    ) -> list[EnrichedChunk]:
        results = self.vector.search(query_embedding, limit, domain_ids, session_id, query_text=query_text)

        enriched = []
        for chunk_id, score, chunk in results:
            entities = []
            if session_id:
                entities = self.relational.get_entities_for_chunk(chunk_id, session_id)
            enriched.append(EnrichedChunk(
                chunk=chunk,
                score=score,
                entities=entities,
            ))
        return enriched

    # ------------------------------------------------------------------
    # Cross-layer: search_similar_entities (vector search → entity lookup)
    # ------------------------------------------------------------------

    def search_similar_entities(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        min_similarity: float = 0.3,
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        results = self.vector.search(query_embedding, limit=limit * 3, domain_ids=domain_ids, session_id=session_id)

        entity_best: dict[str, dict] = {}
        for chunk_id, similarity, _chunk in results:
            if similarity < min_similarity:
                continue

            rows = self.relational.get_non_ignored_entities_for_chunk(chunk_id, session_id)

            for eid, entity_name, entity_type in rows:
                if eid not in entity_best or similarity > entity_best[eid]["similarity"]:
                    entity_best[eid] = {
                        "id": eid,
                        "name": entity_name,
                        "type": entity_type,
                        "similarity": similarity,
                    }

        sorted_entities = sorted(entity_best.values(), key=lambda x: x["similarity"], reverse=True)
        return sorted_entities[:limit]
