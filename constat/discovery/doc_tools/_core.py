# Copyright (c) 2025 Kenneth Stott
# Canary: 08a2c2f4-c8e0-40d4-a898-d89a477f355f
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from constat.core.config import Config, DocumentConfig
from constat.discovery.entity_extractor import EntityExtractor
from constat.discovery.models import (
    DocumentChunk,
    LoadedDocument,
)
from ._entities import _deduplicate_chunk_links, _extract_links_from_chunks
from ._chunking import (  # noqa: F401 — re-exported for backward compatibility
    _is_table_line,
    _is_list_line,
    _merge_blocks,
    _extract_markdown_sections,
    chunk_document as _chunk_document_impl,
)
from ._loaders import (
    _extract_content as _extract_content_impl,
    _load_document as _load_document_impl,
    _load_documents_parallel as _load_documents_parallel_impl,
    _load_file_directly as _load_file_directly_impl,
    add_document_from_config as _add_document_from_config_impl,
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
    _convert_html_to_markdown,
)
from ._mime import normalize_type, detect_type_from_source, is_binary_type
from ._transport import fetch_document, infer_transport, FetchResult
from ._schema_inference import _expand_file_paths, _infer_structured_schema

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
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
        router: Optional[object] = None,
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
        self._router = router
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

        # NER stop list (terms to filter during extraction)
        self._stop_list: set[str] = set()

        # Image labels collected during document loading (fed to NER as business terms)
        self._image_labels: list[str] = []

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

        # Active domain IDs for automatic search filtering
        # Set this when domains are loaded so searches automatically include domain docs
        self._active_domain_ids: list[str] = []

        # Note: No cleanup needed - data is scoped by domain_id/session_id

        # Index documents that aren't already indexed (incremental)
        # Skip if caller will handle indexing (e.g., warmup with hash-based invalidation)
        if not self._skip_auto_index and self.config.documents:
            unindexed_docs = self._get_unindexed_documents()
            if unindexed_docs:
                logger.info(f"[DOC_INIT] Indexing {len(unindexed_docs)} documents...")
                self._load_documents_parallel(unindexed_docs)
                # Incrementally add new documents (don't clear existing chunks)
                self._index_loaded_documents(self._schema_entities)
                logger.info(f"[DOC_INIT] Indexing complete, count={self._vector_store.count()}")

        # Index glossary and relationship chunks from config if present
        self._index_relationship_chunks()

    def _index_relationship_chunks(self) -> None:
        """Index relationship chunks from config into vector store."""
        from constat.catalog.glossary_builder import build_relationship_chunks

        if not self.config.relationships:
            return

        chunks: list[DocumentChunk] = build_relationship_chunks(self.config.relationships)
        if not chunks:
            return

        with self._model_lock:
            texts = [c.content for c in chunks]
            embeddings = self._model.encode(texts, normalize_embeddings=True)

        self._vector_store.add_chunks(chunks, embeddings, source="document")
        logger.info(f"Indexed {len(chunks)} relationship chunks")

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
            indexed_docs = set(self._vector_store.get_indexed_document_names(source='document'))
        except Exception:
            indexed_docs = set()

        # Return documents in config that are not yet indexed
        return [name for name in self.config.documents if name not in indexed_docs]

    def _load_documents_parallel(self, doc_names: list[str]) -> None:
        """Load multiple documents in parallel using threads (I/O bound)."""
        _load_documents_parallel_impl(self, doc_names)

    def _index_loaded_documents(self, schema_entities: Optional[list[str]] = None) -> None:
        """Add document chunks to the vector store (incremental).

        This is the core indexing method for the DOCUMENT resource type.
        It adds chunks for all loaded documents without clearing existing data.
        Chunking is parallelized across documents; embedding is batched.

        Vector store chunk naming:
        - Documents: document_name = "<doc_name>" (e.g., "business_rules")
        - Schema (from SchemaManager): document_name = "schema:<db>.<table>"
        - APIs (from APISchemaManager): document_name = "api:<api_name>.<endpoint>"

        Args:
            schema_entities: Optional list of known schema entity names for entity extraction
        """
        docs = list(self._loaded_documents.items())
        if not docs:
            return

        # Parallel chunking — CPU bound, per-doc independent
        logger.info(f"[DOC_INDEX] Chunking {len(docs)} documents")
        if len(docs) <= 2:
            chunk_results = [
                (name, self._chunk_document(name, doc.content)) for name, doc in docs
            ]
        else:
            max_workers = min(len(docs), 8)
            logger.info(f"[DOC_INDEX] Parallel chunking with {max_workers} threads")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._chunk_document, name, doc.content): name
                    for name, doc in docs
                }
                chunk_results = []
                for future in as_completed(futures):
                    name = futures[future]
                    chunk_results.append((name, future.result()))

        chunks = []
        for name, doc_chunks in chunk_results:
            chunks.extend(doc_chunks)
            doc = self._loaded_documents.get(name)
            if doc and getattr(doc, 'source_url', None) and hasattr(self._vector_store, 'store_document_url'):
                self._vector_store.store_document_url(name, doc.source_url)

        if not chunks:
            return

        # Generate embeddings (batched — model handles internal vectorization)
        logger.info(f"[DOC_INDEX] Embedding {len(chunks)} chunks from {len(docs)} documents")
        texts = [chunk.content for chunk in chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Add to vector store
        self._vector_store.add_chunks(chunks, embeddings, source="document")

        # Extract and store entities
        self._extract_and_store_entities(chunks, schema_entities)

    def embed_entity_values(
        self,
        entity_terms: dict[str, list[str]],
        entity_configs: list,
        session_id: str | None,
        domain_id: str | None = None,
        entity_details: dict[str, list[str]] | None = None,
        api_configs: dict | None = None,
    ) -> None:
        """Embed entity resolution values into the vector store.

        Creates one summary chunk per entity_type+source and individual
        value chunks for semantic matching.  When ``entity_details`` is
        provided, the detail text (which includes all source fields) is
        used as chunk content so that related terms are collocated in
        vector space.

        Args:
            entity_terms: {entity_type: [name, ...]} — names for NER
            entity_configs: List of EntityResolutionConfig objects
            session_id: Session ID (None for warmup)
            domain_id: Domain ID for the entity values
            entity_details: {entity_type: [structured_text, ...]} — rich text for embedding
            api_configs: {name: APIConfig} to detect API sources for naming
        """
        from constat.discovery.models import ChunkType

        if not entity_terms:
            return

        # Delete previous entity_resolution chunks for this scope
        if domain_id:
            self._vector_store.delete_by_source("entity_resolution", domain_id=domain_id)
        else:
            # Base config — delete only base entity_resolution chunks (domain_id IS NULL)
            # Use the backend directly to avoid delete_by_source's "delete all" behavior
            self._vector_store._vector._conn.execute(
                "DELETE FROM embeddings WHERE source = 'entity_resolution' AND domain_id IS NULL"
            )

        # Build a map of entity_type → list of configs (for document_name)
        type_configs: dict[str, list] = {}
        for cfg in entity_configs:
            type_configs.setdefault(cfg.entity_type.upper(), []).append(cfg)

        all_chunks: list[DocumentChunk] = []
        for entity_type, values in entity_terms.items():
            configs = type_configs.get(entity_type.upper(), [])
            detail_texts = (entity_details or {}).get(entity_type, [])

            # Determine which sources are APIs
            api_source_names = set(api_configs.keys()) if api_configs else set()

            # Group values (and details) by source for summary chunks
            source_groups: dict[str, list[tuple[str, str]]] = {}
            source_meta: dict[str, dict] = {}
            for cfg in configs:
                is_api = cfg.source in api_source_names or bool(cfg.endpoint) or bool(cfg.items_path)
                # Derive table name from explicit field or SQL query
                table = getattr(cfg, 'table', None)
                if not table and cfg.query and cfg.source and not is_api:
                    # Parse table from SQL: "SELECT ... FROM tablename ..."
                    import re
                    m = re.search(r'\bFROM\s+(\w+)', cfg.query, re.IGNORECASE)
                    if m:
                        table = m.group(1)

                if is_api:
                    query_name = getattr(cfg, 'items_path', None) or getattr(cfg, 'endpoint', None) or ''
                    key = f"api:{cfg.source}.{query_name}" if query_name else f"api:{cfg.source}"
                elif cfg.source and table:
                    key = f"schema:{cfg.source}.{table}"
                elif cfg.source:
                    key = f"schema:{cfg.source}"
                elif cfg.values:
                    key = f"entity_resolution:{entity_type.lower()}"
                else:
                    key = f"entity_resolution:{entity_type.lower()}"

                source_meta[key] = {
                    "is_api": is_api, "query": cfg.query,
                    "items_path": getattr(cfg, 'items_path', None),
                    "endpoint": getattr(cfg, 'endpoint', None),
                    "table": table,
                    "source": cfg.source,
                }
                if cfg.values:
                    pairs = [(v, v) for v in cfg.values]
                else:
                    pairs = list(zip(values, detail_texts)) if detail_texts else [(v, v) for v in values]
                source_groups.setdefault(key, []).extend(pairs)

            if not source_groups:
                pairs = list(zip(values, detail_texts)) if detail_texts else [(v, v) for v in values]
                source_groups[f"entity_resolution:{entity_type.lower()}"] = pairs

            for source_key, pairs in source_groups.items():
                doc_name = source_key

                # Summary chunk (index 0) — include query for API/DB sources
                meta = source_meta.get(source_key, {})
                preview = ", ".join(name for name, _ in pairs[:10])
                if len(pairs) > 10:
                    preview += f", ... ({len(pairs)} values)"
                query_info = ""
                if meta.get("query"):
                    query_info = f"\nQuery: {meta['query']}"
                summary_content = f"{entity_type} entity values from {source_key}: {preview}{query_info}"
                all_chunks.append(DocumentChunk(
                    document_name=doc_name,
                    content=summary_content,
                    chunk_index=0,
                    source="entity_resolution",
                    chunk_type=ChunkType.ENTITY_VALUE,
                    domain_id=domain_id,
                ))

                # Individual value chunks — use detail text when available
                for idx, (name, detail) in enumerate(pairs, start=1):
                    all_chunks.append(DocumentChunk(
                        document_name=doc_name,
                        content=detail,
                        chunk_index=idx,
                        source="entity_resolution",
                        chunk_type=ChunkType.ENTITY_VALUE,
                        domain_id=domain_id,
                    ))

        if not all_chunks:
            return

        texts = [c.content for c in all_chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        self._vector_store.add_chunks(
            all_chunks, embeddings,
            source="entity_resolution",
            session_id=session_id,
            domain_id=domain_id,
        )
        logger.info(f"Embedded {len(all_chunks)} entity resolution chunks")

    def _create_vector_store(self) -> VectorStoreBackend:
        """Create vector store based on config."""
        storage_config = self.config.storage
        vs_config = storage_config.vector_store if storage_config else None

        backend = vs_config.backend if vs_config else "duckdb"
        db_path = vs_config.db_path if vs_config else None
        reranker_model = vs_config.reranker_model if vs_config else None
        cluster_min_terms = vs_config.cluster_min_terms if vs_config else 2
        cluster_divisor = vs_config.cluster_divisor if vs_config else 5
        cluster_max_k = vs_config.cluster_max_k if vs_config else None
        store_chunk_text = vs_config.store_chunk_text if vs_config else True

        return create_vector_store(
            backend=backend,
            db_path=db_path,
            reranker_model=reranker_model,
            cluster_min_terms=cluster_min_terms,
            cluster_divisor=cluster_divisor,
            cluster_max_k=cluster_max_k,
            store_chunk_text=store_chunk_text,
        )

    def _add_document_internal(
        self,
        name: str,
        content: str,
        doc_format: str = "text",
        description: str = "",
        domain_id: str | None = None,
        session_id: str | None = None,
        skip_entity_extraction: bool = False,
    ) -> bool:
        """Internal method to add a document with full control over flags.

        Args:
            name: Document name
            content: Document content
            doc_format: Format (text, markdown, etc.)
            description: Optional description
            domain_id: Domain this document belongs to (for domain filtering)
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
            type=doc_format if doc_format != "auto" else "text",
            content=content,
            description=description or "",
        )

        # Convert HTML to markdown so heading structure is preserved for chunking
        if doc_format == "html":
            content = _convert_html_to_markdown(content)
            doc_format = "markdown"

        # Store the loaded document
        self._loaded_documents[name] = LoadedDocument(
            name=name,
            config=doc_config,
            content=content,
            format=doc_format,
            sections=_extract_markdown_sections(content, doc_format),
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

        # Add to vector store with domain_id/session_id for filtering
        if hasattr(self._vector_store, 'add_chunks'):
            self._vector_store.add_chunks(
                chunks,
                embeddings,
                source="document",
                session_id=session_id,
                domain_id=domain_id,
            )

        # Extract entities with appropriate scope (unless skipped for later session-level extraction)
        if not skip_entity_extraction:
            if session_id:
                # Session-added documents get session_id for session filtering
                self._extract_and_store_entities_session(chunks, session_id)
            elif domain_id:
                # Domain documents get domain_id for domain filtering
                self._extract_and_store_entities_domain(chunks, domain_id)
            else:
                # Base documents (no domain_id, no session_id) - permanent
                self._extract_and_store_entities(chunks, self._schema_entities)

        return True

    def add_document_from_file(
        self,
        file_path: str,
        name: str | None = None,
        description: str = "",
        domain_id: str | None = None,
        session_id: str | None = None,
        skip_entity_extraction: bool = False,
    ) -> tuple[bool, str]:
        """Add a document from a file path.

        Args:
            file_path: Path to the document file
            name: Optional name (defaults to filename without extension)
            description: Optional description
            domain_id: Optional domain ID for domain-scoped documents
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
            elif suffix in (".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".flac", ".webm", ".mkv", ".aac"):
                from ._audio import transcribe_audio, render_transcript
                doc_config = self._resolve_doc_config(name) if name in self.config.documents else DocumentConfig()
                result = transcribe_audio(path, doc_config)
                content = render_transcript(result, path.stem)
                doc_format = "markdown"
            elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp", ".bmp", ".gif"):
                from ._image import _extract_image, _render_image_result, _describe_image_sync, _ocr_via_vision
                _SUFFIX_TO_MIME = {
                    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".tiff": "image/tiff", ".tif": "image/tiff", ".webp": "image/webp",
                    ".bmp": "image/bmp", ".gif": "image/gif",
                }
                image_result = _extract_image(path=path)
                mime = _SUFFIX_TO_MIME.get(suffix, "image/png")
                # Fallback to LLM vision OCR if tesseract failed
                if not image_result.ocr_text and self._router:
                    logger.warning("Image %s: Tesseract OCR returned no text, falling back to LLM vision OCR", path.name)
                    vision_ocr = _ocr_via_vision(self._router, path.read_bytes(), mime)
                    if vision_ocr.text:
                        image_result.ocr_text = vision_ocr.text
                        image_result.ocr_confidence = vision_ocr.mean_confidence
                        image_result.ocr_word_count = vision_ocr.word_count
                        from ._image import _classify_image
                        image_result.category = _classify_image(vision_ocr)
                logger.info("Image %s: category=%s, ocr_words=%d, ocr_text=%r",
                            path.name, image_result.category, image_result.ocr_word_count,
                            image_result.ocr_text[:100] if image_result.ocr_text else "")
                if self._router and image_result.category == "image-primary":
                    try:
                        desc = _describe_image_sync(self._router, path.read_bytes(), mime)
                        image_result.description = desc.get("description")
                        image_result.subcategory = desc.get("subcategory", image_result.subcategory)
                        image_result.labels = desc.get("labels", image_result.labels)
                        logger.info("Image %s: vision description=%r, labels=%s",
                                    path.name, image_result.description or "", image_result.labels)
                    except Exception as e:
                        logger.warning("Image %s: vision description failed: %s", path.name, e)
                if image_result.labels:
                    self._image_labels.extend(image_result.labels)
                content = _render_image_result(image_result, path.stem)
                doc_format = "markdown"
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
            domain_id=domain_id,
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
                self._vector_store.clear_document_chunks()
            except (AttributeError, Exception):
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
            except (OSError, ValueError):
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
            if doc_config.path:
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
            self._vector_store.clear_document_chunks()
        except (AttributeError, Exception):
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

        # Create extractor with schema entities and spaCy NER
        # Use "__document__" as session_id for general document extraction
        extractor = EntityExtractor(
            session_id="__document__",
            schema_terms=schema_entities,
            api_terms=self._collect_api_terms(),
            stop_list=self._stop_list,
        )

        # Extract entities from each chunk
        all_links = _extract_links_from_chunks(extractor, chunks)

        # Store all unique entities (skip those without domain_id — relational store requires it)
        entities = extractor.get_all_entities()
        storable = [e for e in entities if e.domain_id]
        logger.debug(f"Entity extraction: {len(entities)} unique entities ({len(storable)} with domain_id), {len(all_links)} links from {len(chunks)} chunks")
        if storable:
            self._vector_store.add_entities(storable, session_id="__document__")

        # Store all chunk-entity links (deduplicated by chunk_id + entity_id)
        if all_links:
            self._vector_store.link_chunk_entities(_deduplicate_chunk_links(all_links))

    TABLE_CHUNK_LIMIT = CHUNK_SIZE * 4  # 6000 chars — allow oversized but not unbounded

    def _index_loaded_doc(
        self, doc_name: str, domain_id: str | None, session_id: str | None,
        skip_entity_extraction: bool,
    ) -> int:
        """Chunk, embed, and store a single loaded document. Returns chunk count."""
        doc = self._loaded_documents.get(doc_name)
        if not doc:
            return 0
        chunks = self._chunk_document(doc_name, doc.content)
        if not chunks:
            return 0
        logger.info(f"[DOC_INDEX] {doc_name}: {len(chunks)} chunks, embedding...")
        texts = [c.content for c in chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)
        self._vector_store.add_chunks(
            chunks, embeddings, source="document",
            domain_id=domain_id, session_id=session_id,
        )
        if not skip_entity_extraction:
            if domain_id:
                self._extract_and_store_entities_domain(chunks, domain_id)
            elif session_id:
                self._extract_and_store_entities_session(chunks, session_id)
            else:
                self._extract_and_store_entities(chunks, self._schema_entities)
        return len(chunks)

    def _index_loaded_docs_batch(
        self,
        doc_names: list[str],
        domain_id: str | None,
        session_id: str | None,
        skip_entity_extraction: bool,
    ) -> int:
        """Chunk (parallel) then batch-embed multiple loaded documents. Returns total chunk count.

        Use instead of calling _index_loaded_doc in a loop when indexing many docs
        (e.g. crawled URL pages) — a single model.encode() call is far faster than N calls.
        """
        valid_names = [n for n in doc_names if n in self._loaded_documents]
        if not valid_names:
            return 0

        # Parallel chunking — CPU-bound, independent per doc
        max_workers = min(len(valid_names), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._chunk_document, name, self._loaded_documents[name].content): name
                for name in valid_names
            }
            chunks_by_doc: dict[str, list] = {}
            for future in as_completed(futures):
                name = futures[future]
                chunks_by_doc[name] = future.result()

        # Flat list preserving doc order
        all_chunks = []
        for name in valid_names:
            all_chunks.extend(chunks_by_doc.get(name, []))

        if not all_chunks:
            return 0

        logger.info(
            f"[DOC_INDEX] Batch embedding {len(all_chunks)} chunks from {len(valid_names)} documents"
        )
        texts = [c.content for c in all_chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        self._vector_store.add_chunks(
            all_chunks, embeddings, source="document",
            domain_id=domain_id, session_id=session_id,
        )

        if not skip_entity_extraction:
            if domain_id:
                self._extract_and_store_entities_domain(all_chunks, domain_id)
            elif session_id:
                self._extract_and_store_entities_session(all_chunks, session_id)
            else:
                self._extract_and_store_entities(all_chunks, self._schema_entities)

        return len(all_chunks)

    def _chunk_document(self, name: str, content: str) -> list[DocumentChunk]:
        """Split a document into chunks for embedding.

        Chunks are split only on paragraph/line boundaries - never mid-paragraph.
        Paragraphs are combined until hitting CHUNK_SIZE, then a new chunk starts.
        A chunk may exceed CHUNK_SIZE if a single paragraph is larger (we don't split it).
        Table blocks and list blocks are merged into atomic units before chunking.
        """
        return _chunk_document_impl(name, content, self.CHUNK_SIZE, self.TABLE_CHUNK_LIMIT)

    def _resolve_doc_config(self, name: str) -> DocumentConfig:
        """Resolve a DocumentConfig by name from base config or domains."""
        doc_config = self.config.documents.get(name)
        if not doc_config:
            for domain in self.config.domains.values():
                if domain.documents and name in domain.documents:
                    doc_config = domain.documents[name]
                    break
        if not doc_config:
            configured = list(self.config.documents.keys())
            raise ValueError(
                f"Document not configured: {name}. "
                f"Configured documents: {configured}. "
                f"doc_read() is ONLY for configured reference documents. "
                f"If the data is in a datastore table, use store.query() instead."
            )
        return doc_config

    def _extract_content(self, result: FetchResult, doc_type: str) -> tuple[str, str]:
        """Extract text content from FetchResult bytes based on doc_type.

        Returns (content, doc_format) tuple.
        """
        return _extract_content_impl(self, result, doc_type)

    def _load_document(self, name: str) -> dict | None:
        """Load a document from its configured source.

        Returns:
            None for text documents (stored in _loaded_documents)
            dict for IMAP (with chunk counts)
        """
        return _load_document_impl(self, name)

    def add_document_from_config(
        self,
        name: str,
        doc_config: DocumentConfig,
        domain_id: str | None = None,
        session_id: str | None = None,
        skip_entity_extraction: bool = False,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[bool, str]:
        """Add a document from a DocumentConfig (supports URL, inline, etc.).

        Temporarily registers the config so _resolve_doc_config can find it,
        loads via _load_document (handles URL fetch, crawl, HTML->markdown),
        then indexes all loaded docs with domain/session scoping.

        Args:
            name: Document name
            doc_config: DocumentConfig with url, content, etc.
            domain_id: Domain this document belongs to
            session_id: Session this document was added in
            skip_entity_extraction: If True, skip NER

        Returns:
            Tuple of (success, message)
        """
        return _add_document_from_config_impl(
            self, name, doc_config, domain_id, session_id,
            skip_entity_extraction, progress_callback,
        )

    def _load_file_directly(self, name: str, filepath: Path, doc_config: DocumentConfig) -> None:
        """Load a file directly from a path (for expanded glob/directory entries)."""
        _load_file_directly_impl(self, name, filepath, doc_config)
