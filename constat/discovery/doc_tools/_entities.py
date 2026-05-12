# Copyright (c) 2025 Kenneth Stott
# Canary: 527dfaf2-4f38-4715-8ace-c39abdc3f252
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity extraction mixin stub — entity extraction now handled by chonk."""

import logging

from constat.discovery.models import DocumentChunk, ChunkEntity

logger = logging.getLogger(__name__)


def _deduplicate_chunk_links(all_links: list[ChunkEntity]) -> list[ChunkEntity]:
    """Deduplicate chunk-entity links by (chunk_id, entity_id)."""
    unique_links: dict[tuple, ChunkEntity] = {}
    for link in all_links:
        key = (link.chunk_id, link.entity_id)
        if key not in unique_links:
            unique_links[key] = link
        else:
            existing = unique_links[key]
            unique_links[key] = ChunkEntity(
                chunk_id=link.chunk_id,
                entity_id=link.entity_id,
                mention_count=existing.mention_count + link.mention_count,
                confidence=max(existing.confidence, link.confidence),
                mention_text=existing.mention_text or link.mention_text,
            )
    return list(unique_links.values())


# noinspection PyUnresolvedReferences
class _EntityMixin:
    """Entity mixin — spaCy NER removed; chonk is the extraction source."""

    def set_schema_entities(self, entities: set[str] | list[str]) -> None:
        new_entities = list(entities) if isinstance(entities, set) else entities
        if set(new_entities) == set(self._schema_entities or []):
            return
        # noinspection PyAttributeOutsideInit
        self._schema_entities = new_entities

    def set_openapi_entities(self, operations: list[str], schemas: list[str]) -> None:
        # noinspection PyAttributeOutsideInit
        self._openapi_operations = operations
        # noinspection PyAttributeOutsideInit
        self._openapi_schemas = schemas

    def set_graphql_entities(self, types: list[str], fields: list[str]) -> None:
        # noinspection PyAttributeOutsideInit
        self._graphql_types = types
        # noinspection PyAttributeOutsideInit
        self._graphql_fields = fields

    def extract_entities_for_session(
        self,
        session_id: str,
        domain_ids: list[str],
        schema_entities: list[str],
        api_entities: list[str] | None = None,
        business_terms: list[str] | None = None,
        entity_terms: dict[str, list[str]] | None = None,
    ) -> int:
        return 0

    def process_metadata_through_ner(
        self,
        metadata_texts: list[tuple[str, str]],
        source_type: str = "schema",
    ) -> None:
        pass

    def _get_session_visible_chunks(self, domain_ids: list[str]) -> list[DocumentChunk]:
        if not hasattr(self._vector_store, 'get_visible_chunks_with_metadata'):
            return self._vector_store.get_chunks()

        from constat.discovery.vector_store import DuckDBVectorStore

        domain_join, chunk_filter, params = DuckDBVectorStore.embeddings_domain_join_filter(domain_ids, alias="e")

        result = self._vector_store.get_visible_chunks_with_metadata(chunk_filter, params, domain_join=domain_join)

        chunks = []
        skipped = 0
        for row in result:
            chunk_id, doc_name, content, section, chunk_idx, domain_id = row
            if not content:
                skipped += 1
                continue
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                domain_id=domain_id,
            )
            chunk._chunk_id = chunk_id
            chunks.append(chunk)

        if skipped:
            logger.warning(f"_get_session_visible_chunks: skipped {skipped} chunks with NULL/empty content")
        return chunks
