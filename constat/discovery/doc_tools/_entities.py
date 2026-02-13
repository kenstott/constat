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

from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.models import DocumentChunk, ChunkEntity

logger = logging.getLogger(__name__)


class _EntityMixin:
    """Entity extraction/linking methods for DocumentDiscoveryTools."""

    def set_schema_entities(self, entities: set[str] | list[str]) -> None:
        """Set database schema entities (table names, column names) for pattern matching.

        When schema entities change, entity extraction is re-run on existing
        documents to link schema terms to document references.

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
        self._schema_entities = new_entities

        # Re-extract entities from existing documents if we have indexed chunks
        chunk_count = self._vector_store.count()
        if chunk_count > 0 and hasattr(self._vector_store, 'add_entities'):
            logger.info(f"set_schema_entities: re-extracting entities from {chunk_count} chunks")
            # Clear existing entity links (but keep entities from other sources)
            if hasattr(self._vector_store, 'clear_chunk_entity_links'):
                self._vector_store.clear_chunk_entity_links()

            # Get all chunks and re-extract entities
            chunks = self._vector_store.get_chunks()
            if chunks:
                self._extract_and_store_entities(chunks, self._schema_entities)
                logger.info(f"set_schema_entities: extraction complete")
        else:
            logger.debug(f"set_schema_entities: no chunks to re-extract ({chunk_count} chunks)")

    def extract_entities_for_session(
        self,
        session_id: str,
        project_ids: list[str],
        schema_entities: list[str],
        api_entities: list[str] | None = None,
    ) -> int:
        """Run entity extraction for a session's visible documents.

        Called at session creation to build chunk-entity links using the
        session's entity catalog. Links are stored with session_id.

        Args:
            session_id: Session ID for storing links
            project_ids: List of loaded project IDs
            schema_entities: Schema entity names (tables, columns)
            api_entities: API entity names (endpoints, schemas)

        Returns:
            Number of chunk-entity links created
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return 0

        # Update internal entity lists for extraction
        self._schema_entities = schema_entities or []
        if api_entities:
            self._openapi_operations = api_entities
            self._openapi_schemas = api_entities

        # Get chunks visible to this session (base + projects)
        # Base chunks have project_id='__base__' or NULL
        # Project chunks have project_id in project_ids
        logger.info(f"extract_entities_for_session({session_id}): looking for chunks with project_ids={project_ids}")
        chunks = self._get_session_visible_chunks(project_ids)
        if not chunks:
            logger.warning(f"extract_entities_for_session({session_id}): no visible chunks found!")
            # Debug: check what's in the database
            if hasattr(self._vector_store, '_conn'):
                try:
                    count = self._vector_store._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                    by_proj = self._vector_store._conn.execute("SELECT project_id, COUNT(*) FROM embeddings GROUP BY project_id").fetchall()
                    logger.warning(f"extract_entities_for_session: total embeddings={count}, by_project={by_proj}")
                except Exception as e:
                    logger.warning(f"extract_entities_for_session: failed to check embeddings: {e}")
            return 0

        logger.info(f"extract_entities_for_session({session_id}): extracting from {len(chunks)} chunks")

        # Combine API terms from all sources
        api_terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))

        # Create extractor with session's entity catalog
        extractor = EntityExtractor(
            session_id=session_id,
            schema_terms=self._schema_entities,
            api_terms=api_terms if api_terms else None,
        )

        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        # Store entities - Entity model now has semantic_type instead of metadata
        entities = extractor.get_all_entities()
        if entities:
            # Add all entities to vector store (session_id is required)
            self._vector_store.add_entities(entities, session_id=session_id)

        # Store links WITH session_id
        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links = {}
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
            self._vector_store.link_chunk_entities(list(unique_links.values()))
            logger.info(f"extract_entities_for_session({session_id}): created {len(unique_links)} links")
            return len(unique_links)

        return 0

    def _get_session_visible_chunks(self, project_ids: list[str]) -> list[DocumentChunk]:
        """Get chunks visible to a session (base + loaded projects).

        Args:
            project_ids: List of loaded project IDs

        Returns:
            List of DocumentChunk objects
        """
        if not hasattr(self._vector_store, '_conn'):
            return self._vector_store.get_chunks()

        # Query chunks where project_id is NULL, '__base__', or in project_ids
        conditions = ["project_id IS NULL", "project_id = '__base__'"]
        params = []

        if project_ids:
            placeholders = ",".join(["?" for _ in project_ids])
            conditions.append(f"project_id IN ({placeholders})")
            params.extend(project_ids)

        where_clause = " OR ".join(conditions)

        result = self._vector_store._conn.execute(
            f"""
            SELECT chunk_id, document_name, content, section, chunk_index
            FROM embeddings
            WHERE {where_clause}
            """,
            params,
        ).fetchall()

        chunks = []
        for row in result:
            chunk_id, doc_name, content, section, chunk_idx = row
            chunk = DocumentChunk(
                document_name=doc_name,
                content=content,
                section=section,
                chunk_index=chunk_idx,
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
        entity extraction to find entities that appear across datasources.

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

        # Combine API terms
        api_terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))

        # Run entity extraction on metadata chunks
        # Use "__metadata__" as session_id for metadata processing
        extractor = EntityExtractor(
            session_id="__metadata__",
            schema_terms=self._schema_entities,
            api_terms=api_terms if api_terms else None,
        )

        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        entities = extractor.get_all_entities()
        logger.debug(f"Metadata NER: {len(entities)} entities, {len(all_links)} links from {len(chunks)} metadata items")

        if entities:
            # Add all entities to vector store
            self._vector_store.add_entities(entities, session_id="__metadata__")

        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links = {}
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
            self._vector_store.link_chunk_entities(list(unique_links.values()))

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
        self._openapi_operations = operations
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
        self._graphql_types = types
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
        if not hasattr(self._vector_store, 'add_entities'):
            return

        logger.debug(f"Extracting entities from {len(chunks)} chunks for session {session_id}")

        # Combine API terms
        api_terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))

        # Create extractor with all known schema entities
        extractor = EntityExtractor(
            session_id=session_id,
            schema_terms=self._schema_entities,
            api_terms=api_terms if api_terms else None,
        )

        all_links: list[ChunkEntity] = []

        # Extract entities from all chunks using spaCy NER
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            logger.debug(f"[ENTITY] Chunk '{chunk.section}' -> {len(extractions)} entities")

            for entity, link in extractions:
                all_links.append(link)

        entities = extractor.get_all_entities()
        logger.debug(f"Extracted {len(entities)} entities, {len(all_links)} links")
        if entities:
            self._vector_store.add_entities(entities, session_id=session_id)

        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links = {}
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
            self._vector_store.link_chunk_entities(list(unique_links.values()))

    def _extract_and_store_entities_project(
        self,
        chunks: list[DocumentChunk],
        project_id: str,
    ) -> None:
        """Extract entities from chunks and store them with project_id.

        Uses spaCy NER for named entity extraction plus pattern matching
        for database, OpenAPI, and GraphQL schemas.

        Args:
            chunks: Document chunks to extract entities from
            project_id: Project ID for project-scoped entities
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return

        logger.debug(f"Extracting entities from {len(chunks)} chunks for project {project_id}")

        # Combine API terms
        api_terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))

        # Create extractor with all known schema entities
        # Use project_id as session_id for project-scoped extraction
        extractor = EntityExtractor(
            session_id=project_id,
            project_id=project_id,
            schema_terms=self._schema_entities,
            api_terms=api_terms if api_terms else None,
        )

        all_links: list[ChunkEntity] = []

        # Extract entities from all chunks using spaCy NER
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            logger.debug(f"[ENTITY] Chunk '{chunk.section}' -> {len(extractions)} entities")

            for entity, link in extractions:
                all_links.append(link)

        entities = extractor.get_all_entities()
        logger.debug(f"Extracted {len(entities)} entities, {len(all_links)} links")
        if entities:
            self._vector_store.add_entities(entities, session_id=project_id)

        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links = {}
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
            self._vector_store.link_chunk_entities(list(unique_links.values()))
