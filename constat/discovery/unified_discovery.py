# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unified discovery tools with progressive disclosure pattern.

Provides a layered discovery approach:
- Level 0: discover() - Unified entry point, semantic search across all sources
- Level 1: get_context() - Get document chunks mentioning an entity
- Level 2: Type-specific details (get_table_schema, get_document, get_api_schema)
- Level 3: find_related() - Cross-reference entities
"""

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

from constat.discovery.models import EnrichedChunk

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from constat.discovery.vector_store import VectorStore
    from constat.catalog.schema_manager import SchemaManager
    from constat.catalog.api_schema_manager import APISchemaManager

logger = logging.getLogger(__name__)


class UnifiedDiscovery:
    """Unified discovery with progressive disclosure.

    Provides a single entry point for discovering relevant resources
    across all sources (documents, schemas, APIs) using semantic search.
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        schema_manager: Optional["SchemaManager"] = None,
        api_schema_manager: Optional["APISchemaManager"] = None,
        embed_fn: Optional[callable] = None,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ):
        """Initialize unified discovery.

        Args:
            vector_store: Vector store for entity/chunk search
            schema_manager: Schema manager for table details
            api_schema_manager: API schema manager for endpoint details
            embed_fn: Function to embed text queries
            project_ids: Active project IDs for scoping
            session_id: Current session ID for scoping
        """
        self._vector_store = vector_store
        self._schema_manager = schema_manager
        self._api_schema_manager = api_schema_manager
        self._embed_fn = embed_fn
        self._project_ids = project_ids
        self._session_id = session_id

    def set_scope(
        self,
        project_ids: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Update the discovery scope."""
        self._project_ids = project_ids
        self._session_id = session_id

    def discover(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list["EnrichedChunk"]:
        """Unified discovery across all sources.

        Searches chunks by semantic similarity and returns enriched chunks
        with their related entities. Entities indicate if chunk relates to
        a document, API, or database.

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns:
            List of EnrichedChunk (chunk + score + related entities)
        """
        if not self._embed_fn:
            raise ValueError("embed_fn required for discover()")

        # Embed the query
        query_embedding = self._embed_fn(query)
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)

        # Search chunks and enrich with entities in one call
        enriched_chunks = self._vector_store.search_enriched(
            query_embedding=query_embedding,
            limit=limit,
            project_ids=self._project_ids,
            session_id=self._session_id,
        )

        # Filter by min_score
        return [ec for ec in enriched_chunks if ec.score >= min_score]

    def get_context(
        self,
        entity_name: str,
        limit: int = 5,
    ) -> dict:
        """Level 1: Get document chunks mentioning an entity.

        Returns document excerpts, source references, and related entities.

        Args:
            entity_name: Name of entity to get context for
            limit: Maximum chunks to return

        Returns:
            Dict with entity info and relevant chunks
        """
        # Find the entity using find_entity_by_name
        entity = self._vector_store.find_entity_by_name(
            name=entity_name,
            project_ids=self._project_ids,
            session_id=self._session_id,
        )

        if not entity:
            # Try fuzzy match via semantic search on chunks
            if self._embed_fn:
                results = self.discover(entity_name, limit=1, min_score=0.7)
                if results and results[0].entities:
                    # Use first related entity from the chunk
                    entity_name = results[0].entities[0].name
                    entity = self._vector_store.find_entity_by_name(
                        name=entity_name,
                        project_ids=self._project_ids,
                        session_id=self._session_id,
                    )

        if not entity:
            return {
                "entity": None,
                "chunks": [],
                "related_entities": [],
                "message": f"No entity found matching '{entity_name}'",
            }

        # Get chunks mentioning this entity
        # Returns: list of (chunk_id, DocumentChunk, mention_count, confidence)
        chunk_results = self._vector_store.get_chunks_for_entity(
            entity_id=entity.id,
            limit=limit,
            project_ids=self._project_ids,
            session_id=self._session_id,
        )

        # Convert to dict format for _get_cooccurring_entities
        chunks_for_related = [{"chunk_id": chunk_id} for chunk_id, _, _, _ in chunk_results]

        # Get related entities (co-occurring in same chunks)
        related = self._get_cooccurring_entities(entity.id, chunks_for_related)

        return {
            "entity": {
                "name": entity.name,
                "type": entity.type,
                "source": getattr(entity, "source", None),
                "metadata": entity.metadata or {},
            },
            "chunks": [
                {
                    "content": chunk.content,
                    "document": chunk.document_name,
                    "section": chunk.section,
                    "relevance": mention_count,
                }
                for chunk_id, chunk, mention_count, confidence in chunk_results
            ],
            "related_entities": related,
        }

    def _get_cooccurring_entities(
        self,
        entity_id: str,
        chunks: list[dict],
    ) -> list[dict]:
        """Find entities that co-occur in the same chunks."""
        if not chunks:
            return []

        chunk_ids = [c.get("chunk_id") for c in chunks if c.get("chunk_id")]
        if not chunk_ids:
            return []

        try:
            placeholders = ",".join(["?" for _ in chunk_ids])
            result = self._vector_store._conn.execute(
                f"""
                SELECT DISTINCT e.name, e.type, COUNT(*) as co_occurrences
                FROM chunk_entities ce
                JOIN entities e ON ce.entity_id = e.id
                WHERE ce.chunk_id IN ({placeholders})
                  AND e.id != ?
                GROUP BY e.id, e.name, e.type
                ORDER BY co_occurrences DESC
                LIMIT 5
                """,
                chunk_ids + [entity_id],
            ).fetchall()

            return [
                {"name": row[0], "type": row[1], "co_occurrences": row[2]}
                for row in result
            ]
        except Exception as e:
            logger.warning(f"Error finding co-occurring entities: {e}")
            return []

    def find_related(
        self,
        entity_name: str,
        limit: int = 5,
    ) -> list[dict]:
        """Level 3: Find related entities.

        Finds entities that:
        1. Co-occur in same document chunks
        2. Have similar embeddings (semantic similarity)
        3. Have foreign key relationships (for tables)

        Args:
            entity_name: Name of entity to find relations for
            limit: Maximum related entities to return

        Returns:
            List of related entities with relationship type
        """
        results = []

        # Find the source entity
        entity = self._vector_store.find_entity_by_name(
            name=entity_name,
            project_ids=self._project_ids,
            session_id=self._session_id,
        )

        if not entity:
            return []

        # 1. Co-occurring entities (from same chunks)
        chunk_results = self._vector_store.get_chunks_for_entity(
            entity_id=entity.id,
            limit=10,
            project_ids=self._project_ids,
            session_id=self._session_id,
        )
        chunks_for_related = [{"chunk_id": chunk_id} for chunk_id, _, _, _ in chunk_results]
        cooccurring = self._get_cooccurring_entities(entity.id, chunks_for_related)
        for r in cooccurring:
            results.append({
                "name": r["name"],
                "type": r["type"],
                "relationship": "co-occurrence",
                "strength": r["co_occurrences"],
            })

        # 2. Semantically similar entities - use query by name if no stored embedding
        if self._embed_fn:
            query_embedding = self._embed_fn(entity_name)
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)

            similar = self._vector_store.search_similar_entities(
                query_embedding=query_embedding,
                limit=limit + 1,  # +1 to exclude self
                min_similarity=0.6,
                project_ids=self._project_ids,
                session_id=self._session_id,
            )
            for s in similar:
                if s["id"] != entity.id and s["name"].lower() != entity_name.lower():
                    # Skip if already in results
                    if not any(r["name"].lower() == s["name"].lower() for r in results):
                        results.append({
                            "name": s["name"],
                            "type": s["type"],
                            "relationship": "semantic_similarity",
                            "strength": s["similarity"],
                        })

        # 3. Foreign key relationships (for tables)
        if entity.type == "table" and self._schema_manager:
            fk_related = self._get_fk_relations(entity.name)
            for r in fk_related:
                if not any(x["name"].lower() == r["name"].lower() for x in results):
                    results.append(r)

        # Sort by strength and limit
        results.sort(key=lambda item: item.get("strength", 0), reverse=True)
        return results[:limit]

    def _get_fk_relations(self, table_name: str) -> list[dict]:
        """Get foreign key relationships for a table."""
        if not self._schema_manager:
            return []

        try:
            schema = self._schema_manager.get_table_schema(table_name)
            if not schema:
                return []

            relations = []

            # Parse foreign keys from schema
            fk_info = schema.get("foreign_keys", [])
            for fk in fk_info:
                ref_table = fk.get("references_table", "")
                if ref_table:
                    relations.append({
                        "name": ref_table,
                        "type": "table",
                        "relationship": "foreign_key",
                        "strength": 1.0,
                    })

            return relations
        except Exception as e:
            logger.warning(f"Error getting FK relations for {table_name}: {e}")
            return []


# Tool definitions for planner - optimized for minimal tokens
UNIFIED_DISCOVERY_TOOLS = [
    {
        "name": "discover",
        "description": "START HERE. Semantic search across tables, APIs, documents. Returns ranked matches with type/score/summary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
                "types": {"type": "array", "items": {"type": "string"}, "description": "Filter: table|column|api_endpoint|business_term|concept"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_context",
        "description": "Get document chunks mentioning entity. Returns excerpts + related entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["entity_name"],
        },
    },
    {
        "name": "find_related",
        "description": "Find related entities via co-occurrence, similarity, or FK links.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {"type": "string", "description": "Entity name"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["entity_name"],
        },
    },
]