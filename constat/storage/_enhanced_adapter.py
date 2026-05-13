# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Adapters that wire constat's storage layer into chonk's EnhancedSearch.

EnhancedSearch expects:
  store.search(embedding, limit, query_text)  — seed search
  store.vector.get_all_chunks()              — structural neighbor lookup
  store.vector._conn                         — direct DuckDB for embedding fetch

entity_index expects:
  get_entities_for_chunk(chunk_id) -> [(entity_id, score)]
  get_chunks_for_entity(entity_id, top_n) -> [(chunk_id, score)]

Structural expansion is disabled: constat's _generate_chunk_id includes section
in its hash while chonk's does not, so re-derived IDs would not match.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from chonk.search import EnhancedSearch

if TYPE_CHECKING:
    from chonk.models import ScoredChunk
    from constat.discovery.models import DocumentChunk
    from constat.storage.duckdb_backend import DuckDBVectorBackend
    from constat.storage.relational import RelationalStore

logger = logging.getLogger(__name__)


class _EnhancedStoreAdapter:
    """Satisfies chonk's Store interface: .search() + .vector attribute."""

    def __init__(
        self,
        vector: "DuckDBVectorBackend",
        domain_ids: list[str] | None = None,
        session_id: str | None = None,
        chunk_types: list[str] | None = None,
    ) -> None:
        self.vector = vector
        self._domain_ids = domain_ids
        self._session_id = session_id
        self._chunk_types = chunk_types

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 5,
        query_text: str | None = None,
        chunk_types: list[str] | None = None,
        namespaces: list[str] | None = None,
        domain_ids: list[str] | None = None,
    ) -> list[tuple[str, float, "DocumentChunk"]]:
        return self.vector.search(
            query_embedding,
            limit=limit,
            domain_ids=self._domain_ids,
            session_id=self._session_id,
            query_text=query_text,
            chunk_types=chunk_types if chunk_types is not None else self._chunk_types,
        )


class _RelationalEntityIndex:
    """Wraps constat's RelationalStore as an EnhancedSearch EntityIndex.

    Both methods delegate to DB queries so no in-memory build step is needed.
    """

    def __init__(
        self,
        relational: "RelationalStore",
        session_id: str,
        domain_ids: list[str] | None = None,
    ) -> None:
        self._rel = relational
        self._session_id = session_id
        self._domain_ids = domain_ids

    def get_entities_for_chunk(self, chunk_id: str) -> list[tuple[str, float]]:
        try:
            entities = self._rel.get_entities_for_chunk(chunk_id, self._session_id)
            return [(e.id, 1.0) for e in entities]
        except Exception as exc:
            logger.debug("get_entities_for_chunk failed for %s: %s", chunk_id, exc)
            return []

    def get_chunks_for_entity(
        self, entity_id: str, top_n: int = 3
    ) -> list[tuple[str, float]]:
        try:
            rows = self._rel.get_chunks_for_entity(
                entity_id, limit=top_n, domain_ids=self._domain_ids
            )
            return [(chunk_id, float(confidence)) for chunk_id, _, confidence in rows]
        except Exception as exc:
            logger.debug("get_chunks_for_entity failed for %s: %s", entity_id, exc)
            return []


def run_enhanced_search(
    vector: "DuckDBVectorBackend",
    relational: "RelationalStore | None",
    query_embedding: np.ndarray,
    limit: int,
    query_text: str | None,
    domain_ids: list[str] | None,
    session_id: str | None,
    chunk_types: list[str] | None,
    lane_entity_min_sim: float | None = None,
    mode: str = "vector_first",
    relationship_index=None,
    return_scored: bool = False,
    active_domains: list[str] | None = None,
) -> list[tuple[str, float, "DocumentChunk"]] | list["ScoredChunk"]:
    """Run EnhancedSearch with pre-filtering by chunk_type via chonk's native support.

    When the chonk global Store has been built by warmup, uses it natively for
    domain-isolated search with resolve_session(). Falls back to the adapter
    shim otherwise.

    Args:
        mode: "vector_first" (default), "graph_first", or "global".
        relationship_index: Optional RelationshipIndex for graph_first mode.
        return_scored: If True, return ScoredChunk objects (preserves provenance).
        active_domains: Domain names for resolve_session() (e.g. ["sales-analytics"]).
    """
    from constat.storage._chonk_registry import get_global_store_readonly

    # Derive active domain names from constat domain_ids (strip __base__).
    if active_domains is None and domain_ids:
        active_domains = [d for d in domain_ids if d != "__base__"]

    chonk_store = get_global_store_readonly()
    if chonk_store is not None and active_domains:
        chonk_domain_ids = chonk_store.resolve_domain_ids(
            [("global", name) for name in active_domains],
            include_global=False,
        )
        searcher = EnhancedSearch(  # type: ignore[call-arg]
            store=chonk_store,  # type: ignore[arg-type]
            relationship_index=relationship_index,
            structural_expansion=True,
            entity_expansion=False,
            cluster_expansion=False,
            lane_entity_min_sim=lane_entity_min_sim,
        )
        scored = searcher.search(  # type: ignore[call-arg]
            query_embedding, k=limit, query_text=query_text, mode=mode,
            domain_ids=chonk_domain_ids,
        )
        if return_scored:
            return scored
        return [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]

    # Fallback: use constat adapter shim
    adapter = _EnhancedStoreAdapter(vector, domain_ids, session_id, chunk_types=chunk_types)

    entity_index: _RelationalEntityIndex | None = None
    if relational is not None and session_id:
        entity_index = _RelationalEntityIndex(relational, session_id, domain_ids)

    searcher = EnhancedSearch(  # type: ignore[call-arg]
        store=adapter,  # type: ignore[arg-type]
        entity_index=entity_index,  # type: ignore[arg-type]
        relationship_index=relationship_index,
        structural_expansion=False,
        entity_expansion=(entity_index is not None),
        cluster_expansion=False,
        lane_entity_min_sim=lane_entity_min_sim,
    )

    scored = searcher.search(query_embedding, k=limit, query_text=query_text, mode=mode)  # type: ignore[call-arg]
    if return_scored:
        return scored
    return [(sc.chunk_id, sc.score, sc.chunk) for sc in scored]
