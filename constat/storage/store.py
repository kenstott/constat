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
    # Cross-layer: entity extraction (needs chunks from vector + entity writes to relational)
    # ------------------------------------------------------------------

    def extract_entities_for_session(
        self,
        session_id: str,
        domain_ids: list[str] | None = None,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
        stop_list: set[str] | None = None,
    ) -> int:
        from constat.discovery.entity_extractor import EntityExtractor

        chunks = self.vector.get_all_chunks(domain_ids)
        if not chunks:
            logger.debug(f"No chunks found for session {session_id}")
            return 0

        logger.info(f"Extracting entities from {len(chunks)} chunks for session {session_id}")

        extractor = EntityExtractor(
            session_id=session_id,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
            stop_list=stop_list,
        )

        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            results = extractor.extract(chunk)
            for entity, link in results:
                all_links.append(link)

        entities = extractor.get_all_entities()
        new_ids = {e.id for e in entities}

        # Remove stale entities (exist in DB but not in new extraction)
        old_ids = self.relational.get_entity_ids_for_session(session_id)
        stale_ids = old_ids - new_ids
        if stale_ids:
            self.relational.remove_entities_by_ids(stale_ids)
            logger.info(f"Removed {len(stale_ids)} stale entities for session {session_id}")

        if entities:
            self.relational.add_entities(entities, session_id)
            # Re-link all chunk-entity relationships (links are cheap, ensures correctness)
            self.relational.clear_chunk_entity_links_for_ids(new_ids)
            self.relational.link_chunk_entities(all_links)
            logger.info(f"Extracted {len(entities)} entities ({len(new_ids - old_ids)} new, {len(stale_ids)} removed) from {len(chunks)} chunks")

        return len(entities)

    def extract_entities_for_domain(
        self,
        session_id: str,
        domain_id: str,
        schema_terms: list[str] | None = None,
        api_terms: list[str] | None = None,
        business_terms: list[str] | None = None,
        stop_list: set[str] | None = None,
    ) -> int:
        from constat.discovery.entity_extractor import EntityExtractor

        chunks = self.vector.get_domain_chunks(domain_id)
        if not chunks:
            logger.debug(f"No chunks found for domain {domain_id}")
            return 0

        logger.info(f"Extracting entities from {len(chunks)} chunks for domain {domain_id} in session {session_id}")

        extractor = EntityExtractor(
            session_id=session_id,
            domain_id=domain_id,
            schema_terms=schema_terms,
            api_terms=api_terms,
            business_terms=business_terms,
            stop_list=stop_list,
        )

        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            results = extractor.extract(chunk)
            for entity, link in results:
                all_links.append(link)

        entities = extractor.get_all_entities()
        if entities:
            self.relational.add_entities(entities, session_id)
            self.relational.link_chunk_entities(all_links)
            logger.info(f"Extracted {len(entities)} entities from {len(chunks)} chunks for domain {domain_id}")

        return len(entities)

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
