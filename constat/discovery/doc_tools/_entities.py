# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Entity extraction and linking mixin for DocumentDiscoveryTools."""

import logging
from collections import defaultdict

from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.models import DocumentChunk, ChunkEntity

logger = logging.getLogger(__name__)


def _deduplicate_chunk_links(all_links: list[ChunkEntity]) -> list[ChunkEntity]:
    """Deduplicate chunk-entity links by (chunk_id, entity_id), merging counts and confidence."""
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


def _extract_links_from_chunks(
    extractor: EntityExtractor,
    chunks: list[DocumentChunk],
) -> list[ChunkEntity]:
    """Extract entity links from chunks using the given extractor."""
    all_links: list[ChunkEntity] = []
    for chunk in chunks:
        extractions = extractor.extract(chunk)
        for entity, link in extractions:
            all_links.append(link)
    return all_links


# noinspection PyUnresolvedReferences
class _EntityMixin:
    """Entity extraction/linking methods for DocumentDiscoveryTools."""

    def _collect_api_terms(self) -> list[str] | None:
        """Combine API terms from all sources (OpenAPI + GraphQL)."""
        terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))
        return terms if terms else None

    def set_schema_entities(self, entities: set[str] | list[str]) -> None:
        """Set database schema entities (table names, column names) for pattern matching.

        Only stores entity names for later use by extract_entities_for_session().
        Does NOT trigger extraction — the caller is responsible for that.

        Args:
            entities: Set or list of entity names
        """
        new_entities = list(entities) if isinstance(entities, set) else entities

        # Check if entities actually changed
        if set(new_entities) == set(self._schema_entities or []):
            logger.debug(f"set_schema_entities: no change, skipping (have {len(new_entities)} entities)")
            return

        logger.info(f"set_schema_entities: updating from {len(self._schema_entities or [])} to {len(new_entities)} entities")
        logger.debug(f"set_schema_entities: new entities include: {list(new_entities)[:10]}...")
        # noinspection PyAttributeOutsideInit
        self._schema_entities = new_entities

    def extract_entities_for_session(
        self,
        session_id: str,
        domain_ids: list[str],
        schema_entities: list[str],
        api_entities: list[str] | None = None,
        business_terms: list[str] | None = None,
    ) -> int:
        """Run entity extraction for a session's visible documents.

        Called at session creation to build chunk-entity links using the
        session's entity catalog. Links are stored with session_id.

        Args:
            session_id: Session ID for storing links
            domain_ids: List of loaded domain IDs
            schema_entities: Schema entity names (tables, columns)
            api_entities: API entity names (endpoints, schemas)
            business_terms: Glossary/relationship terms for NER

        Returns:
            Number of chunk-entity links created
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return 0

        # Update internal entity lists for extraction
        # noinspection PyAttributeOutsideInit
        self._schema_entities = schema_entities or []
        if api_entities:
            # noinspection PyAttributeOutsideInit
            self._openapi_operations = api_entities
            # noinspection PyAttributeOutsideInit
            self._openapi_schemas = api_entities

        # Get chunks visible to this session (base + domains)
        # Base chunks have domain_id='__base__' or NULL
        # Domain chunks have domain_id in domain_ids
        logger.info(f"extract_entities_for_session({session_id}): looking for chunks with domain_ids={domain_ids}")
        chunks = self._get_session_visible_chunks(domain_ids)
        if not chunks:
            logger.warning(f"extract_entities_for_session({session_id}): no visible chunks found!")
            # Debug: check what's in the database
            if hasattr(self._vector_store, '_conn'):
                try:
                    count = self._vector_store._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                    by_domain = self._vector_store._conn.execute("SELECT domain_id, COUNT(*) FROM embeddings GROUP BY domain_id").fetchall()
                    logger.warning(f"extract_entities_for_session: total embeddings={count}, by_domain={by_domain}")
                except Exception as e:
                    logger.warning(f"extract_entities_for_session: failed to check embeddings: {e}")
            return 0

        logger.info(f"extract_entities_for_session({session_id}): extracting from {len(chunks)} chunks")

        # Group chunks by domain_id so each EntityExtractor gets the correct domain
        chunks_by_domain: dict[str | None, list[DocumentChunk]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_domain[chunk.domain_id].append(chunk)

        all_links: list[ChunkEntity] = []
        all_entities = []

        for domain_id, domain_chunks in chunks_by_domain.items():
            extractor = EntityExtractor(
                session_id=session_id,
                domain_id=domain_id,
                schema_terms=self._schema_entities,
                api_terms=self._collect_api_terms(),
                business_terms=business_terms,
            )
            links = _extract_links_from_chunks(extractor, domain_chunks)
            all_links.extend(links)
            all_entities.extend(extractor.get_all_entities())
            logger.debug(f"extract_entities_for_session({session_id}): domain={domain_id} -> {len(domain_chunks)} chunks, {len(links)} links")

        if all_entities:
            self._vector_store.add_entities(all_entities, session_id=session_id)

        if all_links:
            deduped = _deduplicate_chunk_links(all_links)
            self._vector_store.link_chunk_entities(deduped)
            logger.info(f"extract_entities_for_session({session_id}): created {len(deduped)} links")

            # Reconcile glossary term domains — move terms to follow
            # their entity when a data source changed domains.
            if hasattr(self._vector_store, 'reconcile_glossary_domains'):
                self._vector_store.reconcile_glossary_domains(session_id)

            return len(deduped)

        return 0

    def _get_session_visible_chunks(self, domain_ids: list[str]) -> list[DocumentChunk]:
        """Get chunks visible to a session (base + loaded domains).

        Args:
            domain_ids: List of loaded domain IDs

        Returns:
            List of DocumentChunk objects
        """
        if not hasattr(self._vector_store, '_conn'):
            return self._vector_store.get_chunks()

        from constat.discovery.vector_store import DuckDBVectorStore

        chunk_filter, params = DuckDBVectorStore.chunk_visibility_filter(domain_ids)

        result = self._vector_store._conn.execute(
            f"""
            SELECT chunk_id, document_name, content, section, chunk_index, domain_id
            FROM embeddings
            WHERE {chunk_filter}
            """,
            params,
        ).fetchall()

        chunks = []
        for row in result:
            chunk_id, doc_name, content, section, chunk_idx, domain_id = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
                domain_id=domain_id,
            )
            # Store chunk_id for linking (hacky but needed for entity extraction)
            chunk._chunk_id = chunk_id
            chunks.append(chunk)

        return chunks

    def process_metadata_through_ner(
        self,
        metadata_texts: list[tuple[str, str]],
        source_type: str = "schema",
    ) -> None:
        """Process schema/API metadata through NER for cross-datasource entity linking.

        Creates pseudo-chunks from metadata text (names, descriptions) and runs
        entity extraction to find entities that appear across 'datasources'.

        Args:
            metadata_texts: List of (source_name, text) tuples
            source_type: Source type for the chunks ("schema" or "api")
        """
        if not metadata_texts:
            return

        if not hasattr(self._vector_store, 'add_entities'):
            return

        logger.info(f"Processing {len(metadata_texts)} {source_type} metadata items through NER")

        # Create pseudo-chunks from metadata
        chunks = []
        for source_name, text in metadata_texts:
            if text and text.strip():
                chunk = DocumentChunk(
                    document_name=f"__{source_type}_metadata__",
                    content=text,
                    section=source_name,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

        if not chunks:
            return

        # Run entity extraction on metadata chunks
        # Use "__metadata__" as session_id for metadata processing
        extractor = EntityExtractor(
            session_id="__metadata__",
            schema_terms=self._schema_entities,
            api_terms=self._collect_api_terms(),
        )

        all_links = _extract_links_from_chunks(extractor, chunks)

        entities = extractor.get_all_entities()
        logger.debug(f"Metadata NER: {len(entities)} entities, {len(all_links)} links from {len(chunks)} metadata items")

        if entities:
            # Add all entities to vector store
            self._vector_store.add_entities(entities, session_id="__metadata__")

        if all_links:
            self._vector_store.link_chunk_entities(_deduplicate_chunk_links(all_links))

    def set_openapi_entities(
        self,
        operations: list[str],
        schemas: list[str],
    ) -> None:
        """Set OpenAPI entities for pattern matching in documents.

        Args:
            operations: List of operation/endpoint names
            schemas: List of schema definition names
        """
        # noinspection PyAttributeOutsideInit
        self._openapi_operations = operations
        # noinspection PyAttributeOutsideInit
        self._openapi_schemas = schemas

    def set_graphql_entities(
        self,
        types: list[str],
        fields: list[str],
    ) -> None:
        """Set GraphQL entities for pattern matching in documents.

        Args:
            types: List of type names
            fields: List of field/operation names
        """
        # noinspection PyAttributeOutsideInit
        self._graphql_types = types
        # noinspection PyAttributeOutsideInit
        self._graphql_fields = fields

    def _extract_and_store_entities_session(
        self,
        chunks: list[DocumentChunk],
        session_id: str,
    ) -> None:
        """Extract entities from chunks and store them with session_id.

        Uses spaCy NER for named entity extraction plus pattern matching
        for database, OpenAPI, and GraphQL schemas.

        Args:
            chunks: Document chunks to extract entities from
            session_id: Session ID for session-scoped entities
        """
        self._extract_and_store_entities_scoped(chunks, session_id=session_id)

    def _extract_and_store_entities_domain(
        self,
        chunks: list[DocumentChunk],
        domain_id: str,
    ) -> None:
        """Extract entities from chunks and store them with domain_id.

        Uses spaCy NER for named entity extraction plus pattern matching
        for database, OpenAPI, and GraphQL schemas.

        Args:
            chunks: Document chunks to extract entities from
            domain_id: Domain ID for domain-scoped entities
        """
        self._extract_and_store_entities_scoped(chunks, session_id=domain_id, domain_id=domain_id)

    def _extract_and_store_entities_scoped(
        self,
        chunks: list[DocumentChunk],
        session_id: str,
        domain_id: str | None = None,
    ) -> None:
        """Extract entities from chunks and store them with the given scope.

        Args:
            chunks: Document chunks to extract entities from
            session_id: Session or domain ID for scoped storage
            domain_id: Optional domain ID passed to EntityExtractor
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return

        scope_label = f"domain {domain_id}" if domain_id else f"session {session_id}"
        logger.debug(f"Extracting entities from {len(chunks)} chunks for {scope_label}")

        # Create extractor with all known schema entities
        extractor_kwargs = dict(
            session_id=session_id,
            schema_terms=self._schema_entities,
            api_terms=self._collect_api_terms(),
        )
        if domain_id:
            extractor_kwargs["domain_id"] = domain_id
        extractor = EntityExtractor(**extractor_kwargs)

        all_links: list[ChunkEntity] = []

        # Extract entities from all chunks using spaCy NER
        # Skip glossary/relationship chunks — they're indexed for search only,
        # running NER on them creates circular entity references.
        for chunk in chunks:
            if chunk.document_name.startswith(("glossary:", "relationship:")):
                continue
            extractions = extractor.extract(chunk)
            logger.debug(f"[ENTITY] Chunk '{chunk.section}' -> {len(extractions)} entities")
            for entity, link in extractions:
                all_links.append(link)

        entities = extractor.get_all_entities()
        logger.debug(f"Extracted {len(entities)} entities, {len(all_links)} links")
        if entities:
            self._vector_store.add_entities(entities, session_id=session_id)

        if all_links:
            self._vector_store.link_chunk_entities(_deduplicate_chunk_links(all_links))
