# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core mixin for DocumentDiscoveryTools — init, document management, indexing, chunking."""

import hashlib
import logging
import threading
from pathlib import Path
from typing import Optional

from constat.core.config import Config, DocumentConfig
from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.models import (
    ChunkEntity,
    DocumentChunk,
    LoadedDocument,
)
from constat.discovery.vector_store import (
    VectorStoreBackend,
    create_vector_store,
)
from constat.embedding_loader import EmbeddingModelLoader
from ._file_extractors import (
    _extract_pdf_text,
    _extract_pdf_text_from_bytes,
    _extract_docx_text,
    _extract_docx_text_from_bytes,
    _extract_xlsx_text,
    _extract_xlsx_text_from_bytes,
    _extract_pptx_text,
    _extract_pptx_text_from_bytes,
    _detect_format,
    _detect_format_from_content_type,
)
from ._schema_inference import _expand_file_paths, _infer_structured_schema

logger = logging.getLogger(__name__)


class _CoreMixin:
    """Init, document management, indexing, and chunking for DocumentDiscoveryTools."""

    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    # bge-large-en-v1.5 has max_seq_length=512 tokens (~2048 chars)
    # Use 1500 chars to stay within limit while maximizing context
    CHUNK_SIZE = 1500
    CACHE_FILENAME = "doc_index_cache.json"

    def __init__(
        self,
        config: Config,
        cache_dir: Optional[Path] = None,
        vector_store: Optional[VectorStoreBackend] = None,
        schema_entities: Optional[list[str]] = None,
        allowed_documents: Optional[set[str]] = None,
        skip_auto_index: bool = False,
    ):
        """Initialize document discovery tools.

        Args:
            config: Config with document definitions
            cache_dir: Directory for caching document metadata
            vector_store: Vector store backend for semantic search
            schema_entities: Schema entities for entity extraction
            allowed_documents: Set of allowed document names. If None, all documents
                are visible. If empty set, no documents are visible.
            skip_auto_index: If True, skip automatic indexing of config documents
        """
        self._skip_auto_index = skip_auto_index
        self.config = config
        self.allowed_documents = allowed_documents
        self._loaded_documents: dict[str, LoadedDocument] = {}

        # Use shared embedding model loader (may already be loading in background)
        self._model = EmbeddingModelLoader.get_instance().get_model()
        # Lock for thread-safe access to embedding model (not thread-safe for concurrent encode)
        self._model_lock = threading.Lock()

        # Schema entities for entity extraction (table names, column names)
        self._schema_entities: list[str] = schema_entities or []

        # OpenAPI entities (operations, schemas)
        self._openapi_operations: list[str] = []
        self._openapi_schemas: list[str] = []

        # GraphQL entities (types, fields)
        self._graphql_types: list[str] = []
        self._graphql_fields: list[str] = []

        # Cache directory for persisting document metadata
        if cache_dir:
            self._cache_dir = cache_dir
        else:
            self._cache_dir = Path.cwd() / ".constat"
        self._cache_file = self._cache_dir / self.CACHE_FILENAME

        # Initialize vector store
        if vector_store:
            self._vector_store = vector_store
        else:
            self._vector_store = self._create_vector_store()

        # Active project IDs for automatic search filtering
        # Set this when projects are loaded so searches automatically include project docs
        self._active_project_ids: list[str] = []

        # Note: No cleanup needed - data is scoped by project_id/session_id

        # Index documents that aren't already indexed (incremental)
        # Skip if caller will handle indexing (e.g., warmup with hash-based invalidation)
        print(f"[DOC_INIT] skip_auto_index={self._skip_auto_index}, documents={list(self.config.documents.keys()) if self.config.documents else []}")
        if not self._skip_auto_index and self.config.documents:
            unindexed_docs = self._get_unindexed_documents()
            print(f"[DOC_INIT] unindexed_docs={unindexed_docs}")
            if unindexed_docs:
                logger.info(f"[DOC_INIT] Indexing {len(unindexed_docs)} documents...")
                for name in unindexed_docs:
                    try:
                        self._load_document(name)
                    except Exception as e:
                        logger.warning(f"[DOC_INIT] Failed to load {name}: {e}")
                # Incrementally add new documents (don't clear existing chunks)
                self._index_loaded_documents(self._schema_entities)
                logger.info(f"[DOC_INIT] Indexing complete, count={self._vector_store.count()}")

    def _is_document_allowed(self, doc_name: str) -> bool:
        """Check if a document is allowed based on permissions."""
        if self.allowed_documents is None:
            return True  # No filtering
        return doc_name in self.allowed_documents

    def _get_unindexed_documents(self) -> list[str]:
        """Get list of configured documents that are not yet indexed.

        Returns:
            List of document names that need indexing
        """
        if not self.config.documents:
            return []

        # Get document names that already have chunks in the vector store
        try:
            indexed_docs = set()
            result = self._vector_store._conn.execute("""
                SELECT DISTINCT document_name
                FROM embeddings
                WHERE source = 'document'
            """).fetchall()
            indexed_docs = {row[0] for row in result}
        except Exception:
            # If query fails, assume nothing is indexed
            indexed_docs = set()

        # Return documents in config that are not yet indexed
        return [name for name in self.config.documents if name not in indexed_docs]

    def _index_loaded_documents(self, schema_entities: Optional[list[str]] = None) -> None:
        """Add document chunks to the vector store (incremental).

        This is the core indexing method for the DOCUMENT resource type.
        It adds chunks for all loaded documents without clearing existing data.

        Vector store chunk naming:
        - Documents: document_name = "<doc_name>" (e.g., "business_rules")
        - Schema (from SchemaManager): document_name = "schema:<db>.<table>"
        - APIs (from APISchemaManager): document_name = "api:<api_name>.<endpoint>"

        Args:
            schema_entities: Optional list of known schema entity names for entity extraction
        """
        print(f"[DOC_INDEX] _loaded_documents={list(self._loaded_documents.keys())}")
        chunks = []
        for name, doc in self._loaded_documents.items():
            doc_chunks = self._chunk_document(name, doc.content)
            print(f"[DOC_INDEX] {name}: {len(doc_chunks)} chunks")
            chunks.extend(doc_chunks)

        if not chunks:
            print("[DOC_INDEX] No chunks to index!")
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Add to vector store
        print(f"[DOC_INDEX] Adding {len(chunks)} chunks with source='document'")
        self._vector_store.add_chunks(chunks, embeddings, source="document")

        # Extract and store entities
        self._extract_and_store_entities(chunks, schema_entities)

    def _create_vector_store(self) -> VectorStoreBackend:
        """Create vector store based on config."""
        storage_config = self.config.storage
        vs_config = storage_config.vector_store if storage_config else None

        backend = vs_config.backend if vs_config else "duckdb"
        db_path = vs_config.db_path if vs_config else None

        return create_vector_store(backend=backend, db_path=db_path)

    def _add_document_internal(
        self,
        name: str,
        content: str,
        doc_format: str = "text",
        description: str = "",
        project_id: str | None = None,
        session_id: str | None = None,
        skip_entity_extraction: bool = False,
    ) -> bool:
        """Internal method to add a document with full control over flags.

        Args:
            name: Document name
            content: Document content
            doc_format: Format (text, markdown, etc.)
            description: Optional description
            project_id: Project this document belongs to (for project filtering)
            session_id: Session this document was added in (for session filtering)
            skip_entity_extraction: If True, skip NER (done later at session creation)

        Returns:
            True if indexed successfully
        """
        from datetime import datetime

        # Remove existing document with same name to avoid duplicate key errors
        if name in self._loaded_documents:
            self.remove_document(name)
        # Always try to delete from vector store in case chunks exist
        if hasattr(self._vector_store, 'delete_by_document'):
            self._vector_store.delete_by_document(name)

        # Create a minimal config for the document
        doc_config = DocumentConfig(
            type="inline",
            content=content,
            description=description or "",
            format=doc_format,
        )

        # Extract sections for markdown
        sections = []
        if doc_format in ("markdown", "md"):
            for line in content.split("\n"):
                if line.startswith("#"):
                    sections.append(line.lstrip("#").strip())

        # Store the loaded document
        self._loaded_documents[name] = LoadedDocument(
            name=name,
            config=doc_config,
            content=content,
            format=doc_format,
            sections=sections,
            loaded_at=datetime.now().isoformat(),
        )

        # Chunk and embed the document
        chunks = self._chunk_document(name, content)
        logger.debug(f"Document '{name}': {len(chunks)} chunks generated")
        if not chunks:
            return True

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Add to vector store with project_id/session_id for filtering
        if hasattr(self._vector_store, 'add_chunks'):
            self._vector_store.add_chunks(
                chunks,
                embeddings,
                source="document",
                session_id=session_id,
                project_id=project_id,
            )

        # Extract entities with appropriate scope (unless skipped for later session-level extraction)
        if not skip_entity_extraction:
            if session_id:
                # Session-added documents get session_id for session filtering
                self._extract_and_store_entities_session(chunks, session_id)
            elif project_id:
                # Project documents get project_id for project filtering
                self._extract_and_store_entities_project(chunks, project_id)
            else:
                # Base documents (no project_id, no session_id) - permanent
                self._extract_and_store_entities(chunks, self._schema_entities)

        return True

    def add_document_from_file(
        self,
        file_path: str,
        name: str | None = None,
        description: str = "",
        project_id: str | None = None,
        session_id: str | None = None,
        skip_entity_extraction: bool = False,
    ) -> tuple[bool, str]:
        """Add a document from a file path.

        Args:
            file_path: Path to the document file
            name: Optional name (defaults to filename without extension)
            description: Optional description
            project_id: Optional project ID for project-scoped documents
            session_id: Optional session ID for session-scoped documents
            skip_entity_extraction: If True, skip NER (done later at session creation)

        Returns:
            Tuple of (success, message)
        """
        path = Path(file_path)
        if not path.exists():
            return False, f"File not found: {file_path}"

        # Use filename as name if not provided
        if not name:
            name = path.stem.replace(" ", "_").replace("-", "_").lower()

        # Determine format and extract content
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                content = _extract_pdf_text(path)
                doc_format = "text"
            elif suffix == ".docx":
                content = _extract_docx_text(path)
                doc_format = "text"
            elif suffix == ".xlsx":
                content = _extract_xlsx_text(path)
                doc_format = "text"
            elif suffix == ".pptx":
                content = _extract_pptx_text(path)
                doc_format = "text"
            elif suffix in (".md", ".markdown"):
                content = path.read_text()
                doc_format = "markdown"
            elif suffix in (".json",):
                content = path.read_text()
                doc_format = "json"
            elif suffix in (".yaml", ".yml"):
                content = path.read_text()
                doc_format = "yaml"
            else:
                content = path.read_text()
                doc_format = "text"
        except Exception as e:
            return False, f"Failed to read file: {e}"

        if not content.strip():
            return False, "File is empty"

        # Add document with specified scope
        success = self._add_document_internal(
            name=name,
            content=content,
            doc_format=doc_format,
            description=description or f"Document from {path.name}",
            project_id=project_id,
            session_id=session_id,
            skip_entity_extraction=skip_entity_extraction,
        )

        if success:
            return True, f"Added document '{name}' ({len(content):,} chars)"
        return False, "Failed to index document"

    def remove_document(self, name: str) -> bool:
        """Remove a document and its vectors from the index.

        Args:
            name: Document name to remove

        Returns:
            True if removed, False if not found
        """
        if name not in self._loaded_documents:
            return False

        del self._loaded_documents[name]

        # Remove from vector store
        if hasattr(self._vector_store, 'delete_by_document'):
            self._vector_store.delete_by_document(name)

        return True

    def get_session_documents(self) -> dict[str, dict]:
        """Get documents added during this session (not in config).

        Returns:
            Dict of {name: {format, char_count, loaded_at}} for session docs
        """
        # Get names of documents from config (documents is a dict keyed by name)
        config_doc_names = set(self.config.documents.keys()) if self.config.documents else set()

        # Find documents that were added but not in config
        session_docs = {}
        for name, doc in self._loaded_documents.items():
            if name not in config_doc_names:
                session_docs[name] = {
                    "format": doc.format,
                    "char_count": len(doc.content),
                    "loaded_at": doc.loaded_at,
                }

        return session_docs

    def refresh(self, force_full: bool = False) -> dict:
        """Refresh documents, using incremental update by default.

        Args:
            force_full: If True, force full rebuild of DOCUMENT chunks only
                       (preserves schema and API chunks from other managers)

        Returns:
            Dict with refresh statistics: {added, updated, removed, unchanged}
        """
        if force_full:
            self._loaded_documents.clear()

            # Clear only DOCUMENT chunks, preserve schema:* and api:* from other managers
            try:
                self._vector_store._conn.execute("""
                    DELETE FROM embeddings
                    WHERE document_name NOT LIKE 'schema:%'
                      AND document_name NOT LIKE 'api:%'
                """)
            except Exception:
                # Fallback for non-DuckDB backends - this will clear everything
                logger.warning("refresh: falling back to full clear (non-DuckDB backend)")
                self._vector_store.clear()

            # Clear document-sourced entities only
            if hasattr(self._vector_store, 'clear_entities'):
                self._vector_store.clear_entities(source='document')

            # Reload all documents and rebuild index
            stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "mode": "full_rebuild"}
            if self.config.documents:
                for name in self.config.documents:
                    try:
                        self._load_document(name)
                        stats["added"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {name}: {e}")
                self._index_loaded_documents(self._schema_entities)
            return stats

        return self._refresh_incremental()

    def _refresh_incremental(self) -> dict:
        """Incrementally update document index based on file changes.

        Returns:
            Dict with refresh statistics
        """
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

        # Get current document list (expanded for globs/directories)
        current_docs = self._get_all_document_paths()

        # Track which documents to reload
        docs_to_reload: list[str] = []
        docs_to_remove: list[str] = []

        # Check each configured document
        for doc_name, doc_path in current_docs.items():
            if doc_path and doc_path.exists():
                current_mtime = doc_path.stat().st_mtime

                if doc_name in self._loaded_documents:
                    # Check if file has changed
                    loaded_doc = self._loaded_documents[doc_name]
                    if loaded_doc.file_mtime != current_mtime:
                        docs_to_reload.append(doc_name)
                        stats["updated"] += 1
                    else:
                        stats["unchanged"] += 1
                else:
                    # New document
                    docs_to_reload.append(doc_name)
                    stats["added"] += 1
            elif doc_name in self._loaded_documents:
                # Document was removed
                docs_to_remove.append(doc_name)
                stats["removed"] += 1

        # Check for documents that no longer exist in config
        for doc_name in list(self._loaded_documents.keys()):
            if doc_name not in current_docs:
                docs_to_remove.append(doc_name)
                stats["removed"] += 1

        # Remove deleted documents
        for doc_name in docs_to_remove:
            if doc_name in self._loaded_documents:
                del self._loaded_documents[doc_name]

        # Reload changed documents
        for doc_name in docs_to_reload:
            try:
                self._load_document_with_mtime(doc_name)
            except Exception:
                pass  # Skip documents that fail to load

        # Rebuild index if anything changed
        if docs_to_reload or docs_to_remove:
            self._build_index(self._schema_entities)

        return stats

    def _get_all_document_paths(self) -> dict[str, Optional[Path]]:
        """Get all document names mapped to their file paths.

        Returns:
            Dict mapping document names to Path objects (None for non-file docs)
        """
        result = {}

        for doc_name, doc_config in self.config.documents.items():
            if doc_config.type == "file" and doc_config.path:
                expanded = _expand_file_paths(doc_config.path)
                if len(expanded) > 1:
                    # Multiple files - each gets its own entry
                    for filename, filepath in expanded:
                        full_name = f"{doc_name}:{filename}"
                        result[full_name] = filepath
                elif len(expanded) == 1:
                    _, filepath = expanded[0]
                    result[doc_name] = filepath
                else:
                    result[doc_name] = None
            else:
                # Non-file documents (inline, http, etc.)
                result[doc_name] = None

        return result

    def _load_document_with_mtime(self, name: str) -> None:
        """Load a document and record its modification time."""
        # Get the file path
        doc_paths = self._get_all_document_paths()
        filepath = doc_paths.get(name)

        # Load the document
        if ":" in name:
            # Expanded glob/directory entry
            parent_name, filename = name.split(":", 1)
            if parent_name in self.config.documents:
                doc_config = self.config.documents[parent_name]
                if filepath and filepath.exists():
                    self._load_file_directly(name, filepath, doc_config)
        else:
            self._load_document(name)

        # Record mtime and content hash
        if name in self._loaded_documents:
            doc = self._loaded_documents[name]
            if filepath and filepath.exists():
                doc.file_mtime = filepath.stat().st_mtime
            doc.content_hash = hashlib.sha256(doc.content.encode()).hexdigest()[:16]

    def _build_index(self, schema_entities: Optional[list[str]] = None) -> None:
        """Full rebuild of DOCUMENT chunks in the vector store.

        Resource Type: DOCUMENT
        - Clears all document chunks (document_name NOT LIKE 'schema:%' AND NOT LIKE 'api:%')
        - Re-indexes all loaded documents
        - Preserves schema and API chunks from other managers

        The vector store is shared across 3 resource types:
        - DOCUMENT: managed by DocumentDiscoveryTools (this class)
        - SCHEMA: managed by SchemaManager (document_name LIKE 'schema:%')
        - API: managed by APISchemaManager (document_name LIKE 'api:%')

        Each manager is responsible for its own resource type's cache lifecycle.

        Args:
            schema_entities: Optional list of known schema entity names for entity extraction
        """
        # Clear DOCUMENT chunks only (preserve schema:* and api:* from other managers)
        try:
            self._vector_store._conn.execute("""
                DELETE FROM embeddings
                WHERE document_name NOT LIKE 'schema:%'
                  AND document_name NOT LIKE 'api:%'
            """)
        except Exception:
            # Fallback for non-DuckDB backends
            self._vector_store.clear()

        # Clear document-sourced entities only
        if hasattr(self._vector_store, 'clear_entities'):
            self._vector_store.clear_entities(source='document')

        # Index all loaded documents
        self._index_loaded_documents(schema_entities)

    def _extract_and_store_entities(
        self,
        chunks: list[DocumentChunk],
        schema_entities: Optional[list[str]] = None,
    ) -> None:
        """Extract entities from chunks and store them using spaCy NER.

        Args:
            chunks: Document chunks to extract entities from
            schema_entities: Known schema entity names for matching
        """
        # Skip if vector store doesn't support entities
        if not hasattr(self._vector_store, 'add_entities'):
            return

        # Combine API terms
        api_terms = list(set(
            (self._openapi_operations or []) +
            (self._openapi_schemas or []) +
            (self._graphql_types or []) +
            (self._graphql_fields or [])
        ))

        # Create extractor with schema entities and spaCy NER
        # Use "__document__" as session_id for general document extraction
        extractor = EntityExtractor(
            session_id="__document__",
            schema_terms=schema_entities,
            api_terms=api_terms if api_terms else None,
        )

        # Extract entities from each chunk
        all_links: list[ChunkEntity] = []

        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        # Store all unique entities
        entities = extractor.get_all_entities()
        logger.debug(f"Entity extraction: {len(entities)} unique entities, {len(all_links)} links from {len(chunks)} chunks")
        if entities:
            self._vector_store.add_entities(entities, session_id="__document__")

        # Store all chunk-entity links (deduplicated by chunk_id + entity_id)
        if all_links:
            # Deduplicate links - same entity in same chunk should only have one link
            unique_links = {}
            for link in all_links:
                key = (link.chunk_id, link.entity_id)
                if key not in unique_links:
                    unique_links[key] = link
                else:
                    # Merge mention counts if duplicate
                    existing = unique_links[key]
                    unique_links[key] = ChunkEntity(
                        chunk_id=link.chunk_id,
                        entity_id=link.entity_id,
                        mention_count=existing.mention_count + link.mention_count,
                        confidence=max(existing.confidence, link.confidence),
                        mention_text=existing.mention_text or link.mention_text,
                    )
            self._vector_store.link_chunk_entities(list(unique_links.values()))

    def _chunk_document(self, name: str, content: str) -> list[DocumentChunk]:
        """Split a document into chunks for embedding.

        Chunks are split only on paragraph/line boundaries - never mid-paragraph.
        Paragraphs are combined until hitting CHUNK_SIZE, then a new chunk starts.
        A chunk may exceed CHUNK_SIZE if a single paragraph is larger (we don't split it).
        """
        chunks = []
        current_section = None

        # Determine paragraph separator based on content
        # Use double newline if present, otherwise single newline
        if "\n\n" in content:
            paragraphs = content.split("\n\n")
            separator = "\n\n"
        else:
            paragraphs = content.split("\n")
            separator = "\n"

        chunk_index = 0
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Track sections from markdown headers
            if para.startswith("#"):
                current_section = para.lstrip("#").strip()

            # Check if adding this paragraph would exceed chunk size
            potential_chunk = (current_chunk + separator + para).strip() if current_chunk else para

            if len(potential_chunk) <= self.CHUNK_SIZE:
                # Fits in current chunk - add it
                current_chunk = potential_chunk
            else:
                # Would exceed chunk size
                if current_chunk:
                    # Save current chunk first
                    chunks.append(DocumentChunk(
                        document_name=name,
                        content=current_chunk,
                        section=current_section,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                # Start new chunk with this paragraph (even if it exceeds CHUNK_SIZE)
                current_chunk = para

        # Save final chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                document_name=name,
                content=current_chunk,
                section=current_section,
                chunk_index=chunk_index,
            ))

        return chunks

    def _load_document(self, name: str) -> dict | None:
        """Load a document from its configured source.

        Returns:
            None for text documents (stored in _loaded_documents)
            dict for binary files (PDF, Office) to be returned directly
        """
        from datetime import datetime

        # Check base config first
        doc_config = self.config.documents.get(name)

        # Check project documents if not in base
        if not doc_config:
            for project in self.config.projects.values():
                if project.documents and name in project.documents:
                    doc_config = project.documents[name]
                    break

        if not doc_config:
            configured = list(self.config.documents.keys())
            raise ValueError(
                f"Document not configured: {name}. "
                f"Configured documents: {configured}. "
                f"doc_read() is ONLY for configured reference documents. "
                f"If the data is in a datastore table, use store.query() instead."
            )
        content = ""
        doc_format = doc_config.format

        if doc_config.type == "inline":
            content = doc_config.content or ""
            if doc_format == "auto":
                doc_format = "text"

        elif doc_config.type == "file":
            if doc_config.path:
                path = Path(doc_config.path)
                # Resolve relative paths from config directory
                if not path.is_absolute() and self.config.config_dir:
                    path = (Path(self.config.config_dir) / doc_config.path).resolve()
                if path.exists():
                    suffix = path.suffix.lower()

                    # Binary document formats - extract text content
                    if suffix == ".pdf":
                        content = _extract_pdf_text(path)
                        doc_format = "text"
                    elif suffix == ".docx":
                        content = _extract_docx_text(path)
                        doc_format = "text"
                    elif suffix == ".xlsx":
                        content = _extract_xlsx_text(path)
                        doc_format = "text"
                    elif suffix == ".pptx":
                        content = _extract_pptx_text(path)
                        doc_format = "text"
                    else:
                        # Check for structured data files - use schema metadata
                        schema = _infer_structured_schema(path, doc_config.description)
                        if schema:
                            content = schema.to_metadata_doc()
                            doc_format = schema.file_format
                        # Text-based files - return content for rendering
                        else:
                            content = path.read_text()
                            if doc_format == "auto":
                                doc_format = _detect_format(suffix)
                else:
                    raise FileNotFoundError(f"Document file not found: {doc_config.path}")

        elif doc_config.type == "http":
            if doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()

                # Check content type and URL extension for binary formats
                content_type = response.headers.get("content-type", "")
                url_lower = doc_config.url.lower() if doc_config.url else ""

                if "pdf" in content_type or url_lower.endswith(".pdf"):
                    content = _extract_pdf_text_from_bytes(response.content)
                    doc_format = "text"
                elif "wordprocessingml" in content_type or url_lower.endswith(".docx"):
                    content = _extract_docx_text_from_bytes(response.content)
                    doc_format = "text"
                elif "spreadsheetml" in content_type or url_lower.endswith(".xlsx"):
                    content = _extract_xlsx_text_from_bytes(response.content)
                    doc_format = "text"
                elif "presentationml" in content_type or url_lower.endswith(".pptx"):
                    content = _extract_pptx_text_from_bytes(response.content)
                    doc_format = "text"
                else:
                    content = response.text
                    if doc_format == "auto":
                        doc_format = _detect_format_from_content_type(content_type)

        elif doc_config.type == "pdf":
            # Direct PDF type - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = _extract_pdf_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"PDF file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = _extract_pdf_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "docx":
            # Word document - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = _extract_docx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"Word document not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = _extract_docx_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "xlsx":
            # Excel spreadsheet - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = _extract_xlsx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"Excel file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = _extract_xlsx_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "pptx":
            # PowerPoint presentation - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = _extract_pptx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"PowerPoint file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = _extract_pptx_text_from_bytes(response.content)
                doc_format = "text"

        # TODO: Implement confluence, notion loaders
        else:
            raise NotImplementedError(f"Document type not yet implemented: {doc_config.type}")

        # Extract sections for markdown
        sections = []
        if doc_format in ("markdown", "md"):
            for line in content.split("\n"):
                if line.startswith("#"):
                    sections.append(line.lstrip("#").strip())

        self._loaded_documents[name] = LoadedDocument(
            name=name,
            config=doc_config,
            content=content,
            format=doc_format,
            sections=sections,
            loaded_at=datetime.now().isoformat(),
        )

    def _load_file_directly(self, name: str, filepath: Path, doc_config: DocumentConfig) -> None:
        """Load a file directly from a path (for expanded glob/directory entries)."""
        from datetime import datetime

        suffix = filepath.suffix.lower()
        doc_format = doc_config.format

        # Check if it's a structured data file - use schema metadata instead of raw content
        schema = _infer_structured_schema(filepath, doc_config.description)
        if schema:
            # For structured files, index the metadata, not the raw data
            content = schema.to_metadata_doc()
            doc_format = schema.file_format
            sections = ["Schema", "Columns"]

            # Store schema for later reference
            file_config = DocumentConfig(
                type="file",
                path=str(filepath),
                description=doc_config.description,
                format=doc_format,
                tags=doc_config.tags,
            )

            self._loaded_documents[name] = LoadedDocument(
                name=name,
                config=file_config,
                content=content,
                format=doc_format,
                sections=sections,
                loaded_at=datetime.now().isoformat(),
            )
            return

        # Handle other file types
        if suffix == ".pdf":
            content = _extract_pdf_text(filepath)
            doc_format = "text"
        elif suffix == ".docx":
            content = _extract_docx_text(filepath)
            doc_format = "text"
        elif suffix == ".xlsx":
            content = _extract_xlsx_text(filepath)
            doc_format = "text"
        elif suffix == ".pptx":
            content = _extract_pptx_text(filepath)
            doc_format = "text"
        else:
            content = filepath.read_text()
            if doc_format == "auto" or not doc_format:
                doc_format = _detect_format(suffix)

        # Extract sections for markdown
        sections = []
        if doc_format in ("markdown", "md"):
            for line in content.split("\n"):
                if line.startswith("#"):
                    sections.append(line.lstrip("#").strip())

        # Create a modified config with the actual path
        file_config = DocumentConfig(
            type="file",
            path=str(filepath),
            description=doc_config.description,
            format=doc_format,
            tags=doc_config.tags,
        )

        self._loaded_documents[name] = LoadedDocument(
            name=name,
            config=file_config,
            content=content,
            format=doc_format,
            sections=sections,
            loaded_at=datetime.now().isoformat(),
        )
