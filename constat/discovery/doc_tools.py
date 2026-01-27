# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Document discovery tools for reference documents.

These tools allow the LLM to discover and search reference documents
on-demand rather than loading everything into the system prompt.
"""

import glob as glob_module
import json
import logging
from pathlib import Path
from typing import Optional
import hashlib
import threading

import numpy as np

logger = logging.getLogger(__name__)

from constat.core.config import Config, DocumentConfig
from constat.embedding_loader import EmbeddingModelLoader
from constat.discovery.models import (
    DocumentChunk,
    LoadedDocument,
    StructuredFileSchema,
    Entity,
    ChunkEntity,
)
from constat.discovery.vector_store import (
    VectorStoreBackend,
    create_vector_store,
)
from constat.discovery.entity_extractor import EntityExtractor, ExtractionConfig


def _is_glob_pattern(path: str) -> bool:
    """Check if a path contains glob pattern characters."""
    return any(c in path for c in ["*", "?", "[", "]"])


def _expand_file_paths(path: str) -> list[tuple[str, Path]]:
    """
    Expand a path that may be a glob pattern, directory, or single file.

    Args:
        path: File path, glob pattern, or directory path

    Returns:
        List of (display_name, resolved_path) tuples
    """
    p = Path(path)

    # Case 1: Glob pattern
    if _is_glob_pattern(path):
        matches = sorted(glob_module.glob(path, recursive=True))
        return [(Path(m).name, Path(m)) for m in matches if Path(m).is_file()]

    # Case 2: Directory - list all files
    if p.is_dir():
        files = []
        for f in sorted(p.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                files.append((f.name, f))
        return files

    # Case 3: Single file
    if p.exists():
        return [(p.name, p)]

    # Path doesn't exist yet - return as-is for later error handling
    return [(p.name, p)]



def _infer_csv_schema(filepath: Path, sample_rows: int = 100) -> StructuredFileSchema:
    """Infer schema from a CSV file."""
    import csv

    columns = []
    row_count = 0

    with open(filepath, 'r', newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return StructuredFileSchema(
                filename=filepath.name,
                filepath=str(filepath),
                file_format="csv",
                row_count=0,
                columns=[],
            )

        # Initialize column info
        col_data = {h: {'name': h, 'values': []} for h in headers}

        # Sample rows to infer types and collect sample values
        for i, row in enumerate(reader):
            row_count += 1
            if i < sample_rows:
                for j, val in enumerate(row):
                    if j < len(headers):
                        col_data[headers[j]]['values'].append(val)

        # Count remaining rows
        for _ in reader:
            row_count += 1

    # Infer types and get sample values
    for header in headers:
        values = col_data[header]['values']
        col_type = _infer_column_type(values)
        unique_values = list(set(v for v in values if v))[:10]

        columns.append({
            'name': header,
            'type': col_type,
            'sample_values': unique_values,
        })

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="csv",
        row_count=row_count,
        columns=columns,
    )


def _infer_json_schema(filepath: Path, sample_docs: int = 100) -> StructuredFileSchema:
    """Infer schema from a JSON file (array of objects or single object)."""
    import json

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return StructuredFileSchema(
                filename=filepath.name,
                filepath=str(filepath),
                file_format="json",
                row_count=0,
                columns=[],
            )

    # Handle array of objects vs single object
    if isinstance(data, list):
        docs = data[:sample_docs]
        row_count = len(data)
    elif isinstance(data, dict):
        docs = [data]
        row_count = 1
    else:
        return StructuredFileSchema(
            filename=filepath.name,
            filepath=str(filepath),
            file_format="json",
            row_count=1,
            columns=[],
        )

    # Collect all keys and their values
    key_values: dict[str, list] = {}
    for doc in docs:
        if isinstance(doc, dict):
            for key, val in doc.items():
                if key not in key_values:
                    key_values[key] = []
                key_values[key].append(val)

    # Build column info
    columns = []
    for key, values in key_values.items():
        col_type = _infer_json_value_type(values)
        # Get sample values (only for simple types)
        sample_values = []
        for v in values[:10]:
            if isinstance(v, (str, int, float, bool)) and v is not None:
                sample_values.append(str(v) if not isinstance(v, str) else v)
        unique_samples = list(set(sample_values))[:10]

        columns.append({
            'name': key,
            'type': col_type,
            'sample_values': unique_samples,
        })

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="json",
        row_count=row_count,
        columns=columns,
    )


def _infer_column_type(values: list[str]) -> str:
    """Infer column type from string values."""
    if not values:
        return "unknown"

    # Check for numeric types
    int_count = 0
    float_count = 0
    date_count = 0

    for v in values:
        if not v:
            continue
        try:
            int(v)
            int_count += 1
            continue
        except ValueError:
            pass
        try:
            float(v)
            float_count += 1
            continue
        except ValueError:
            pass
        # Simple date check
        if len(v) == 10 and v[4:5] == '-' and v[7:8] == '-':
            date_count += 1

    non_empty = len([v for v in values if v])
    if non_empty == 0:
        return "unknown"

    if int_count == non_empty:
        return "integer"
    if int_count + float_count == non_empty:
        return "float"
    if date_count > non_empty * 0.8:
        return "date"
    return "string"


def _infer_json_value_type(values: list) -> str:
    """Infer type from JSON values."""
    if not values:
        return "unknown"

    types = set()
    for v in values:
        if v is None:
            continue
        elif isinstance(v, bool):
            types.add("boolean")
        elif isinstance(v, int):
            types.add("integer")
        elif isinstance(v, float):
            types.add("float")
        elif isinstance(v, str):
            types.add("string")
        elif isinstance(v, list):
            types.add("array")
        elif isinstance(v, dict):
            types.add("object")

    if len(types) == 0:
        return "null"
    if len(types) == 1:
        return types.pop()
    if types == {"integer", "float"}:
        return "float"
    return "mixed"


def _infer_structured_schema(filepath: Path, description: Optional[str] = None) -> Optional[StructuredFileSchema]:
    """Infer schema for a structured file based on its extension."""
    suffix = filepath.suffix.lower()

    if suffix == ".csv":
        schema = _infer_csv_schema(filepath)
    elif suffix == ".json":
        schema = _infer_json_schema(filepath)
    elif suffix == ".jsonl":
        # JSON Lines - read first N lines as separate JSON objects
        schema = _infer_jsonl_schema(filepath)
    else:
        return None

    schema.description = description
    return schema


def _infer_jsonl_schema(filepath: Path, sample_lines: int = 100) -> StructuredFileSchema:
    """Infer schema from a JSON Lines file."""
    import json

    docs = []
    row_count = 0

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            row_count += 1
            if i < sample_lines and line.strip():
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Collect all keys and their values
    key_values: dict[str, list] = {}
    for doc in docs:
        if isinstance(doc, dict):
            for key, val in doc.items():
                if key not in key_values:
                    key_values[key] = []
                key_values[key].append(val)

    # Build column info
    columns = []
    for key, values in key_values.items():
        col_type = _infer_json_value_type(values)
        sample_values = []
        for v in values[:10]:
            if isinstance(v, (str, int, float, bool)) and v is not None:
                sample_values.append(str(v) if not isinstance(v, str) else v)
        unique_samples = list(set(sample_values))[:10]

        columns.append({
            'name': key,
            'type': col_type,
            'sample_values': unique_samples,
        })

    return StructuredFileSchema(
        filename=filepath.name,
        filepath=str(filepath),
        file_format="jsonl",
        row_count=row_count,
        columns=columns,
    )


class DocumentDiscoveryTools:
    """Tools for discovering and searching reference documents on-demand.

    Supports incremental updates - only reloads documents that have changed
    based on file modification times and content hashes.
    """

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
    ):
        """Initialize document discovery tools.

        Args:
            config: Config with document definitions
            cache_dir: Directory for caching document metadata
            vector_store: Vector store backend for semantic search
            schema_entities: Schema entities for entity extraction
            allowed_documents: Set of allowed document names. If None, all documents
                are visible. If empty set, no documents are visible.
        """
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

        # Clean up ephemeral data from previous sessions
        if hasattr(self._vector_store, 'clear_ephemeral'):
            logger.debug("DocumentDiscoveryTools.__init__: calling clear_ephemeral()")
            self._vector_store.clear_ephemeral()
        else:
            logger.debug("DocumentDiscoveryTools.__init__: vector store has no clear_ephemeral method")

        # Index documents if vector store is empty (first-time setup)
        # This MUST complete synchronously before __init__ returns
        if self._vector_store.count() == 0 and self.config.documents:
            logger.info(f"[DOC_INIT] Vector store empty, indexing {len(self.config.documents)} documents...")
            for name in self.config.documents:
                try:
                    self._load_document(name)
                except Exception as e:
                    logger.warning(f"[DOC_INIT] Failed to load {name}: {e}")
            self._build_index(self._schema_entities)
            logger.info(f"[DOC_INIT] Indexing complete, count={self._vector_store.count()}")

    def _is_document_allowed(self, doc_name: str) -> bool:
        """Check if a document is allowed based on permissions."""
        if self.allowed_documents is None:
            return True  # No filtering
        return doc_name in self.allowed_documents

    def _create_vector_store(self) -> VectorStoreBackend:
        """Create vector store based on config."""
        storage_config = self.config.storage
        vs_config = storage_config.vector_store if storage_config else None

        backend = vs_config.backend if vs_config else "duckdb"
        db_path = vs_config.db_path if vs_config else None

        return create_vector_store(backend=backend, db_path=db_path)

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
            return

        self._schema_entities = new_entities

        # Re-extract entities from existing documents if we have indexed chunks
        if self._vector_store.count() > 0 and hasattr(self._vector_store, 'add_entities'):
            # Clear existing entity links (but keep entities from other sources)
            if hasattr(self._vector_store, 'clear_chunk_entity_links'):
                self._vector_store.clear_chunk_entity_links()

            # Get all chunks and re-extract entities
            chunks = self._vector_store.get_chunks()
            if chunks:
                self._extract_and_store_entities(chunks, self._schema_entities)

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

    def add_ephemeral_document(
        self,
        name: str,
        content: str,
        doc_format: str = "text",
        description: str = "",
    ) -> bool:
        """Add a session-only document that will be cleaned up on restart.

        Use this for documents added via /file during a session.

        Args:
            name: Document name
            content: Document content
            doc_format: Format (text, markdown, etc.)
            description: Optional description

        Returns:
            True if indexed successfully
        """
        from datetime import datetime

        # Remove existing document with same name to avoid duplicate key errors
        if name in self._loaded_documents:
            self.remove_document(name)
        # Always try to delete from vector store in case chunks exist from previous session
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

        # Add to vector store with ephemeral=True
        if hasattr(self._vector_store, 'add_chunks'):
            # Check if the method accepts ephemeral parameter
            import inspect
            sig = inspect.signature(self._vector_store.add_chunks)
            if 'ephemeral' in sig.parameters:
                self._vector_store.add_chunks(chunks, embeddings, ephemeral=True)
            else:
                self._vector_store.add_chunks(chunks, embeddings)

        # Extract entities with ephemeral=True
        self._extract_and_store_entities_ephemeral(chunks)

        return True

    def add_ephemeral_document_from_file(
        self,
        file_path: str,
        name: str | None = None,
        description: str = "",
    ) -> tuple[bool, str]:
        """Add a session-only document from a file path.

        Args:
            file_path: Path to the document file
            name: Optional name (defaults to filename without extension)
            description: Optional description

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
                content = self._extract_pdf_text(path)
                doc_format = "text"
            elif suffix == ".docx":
                content = self._extract_docx_text(path)
                doc_format = "text"
            elif suffix == ".xlsx":
                content = self._extract_xlsx_text(path)
                doc_format = "text"
            elif suffix == ".pptx":
                content = self._extract_pptx_text(path)
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

        # Add as ephemeral document
        success = self.add_ephemeral_document(
            name=name,
            content=content,
            doc_format=doc_format,
            description=description or f"Session document from {path.name}",
        )

        if success:
            return True, f"Added document '{name}' ({len(content):,} chars)"
        return False, "Failed to index document"

    def _extract_and_store_entities_ephemeral(
        self,
        chunks: list[DocumentChunk],
    ) -> None:
        """Extract entities from chunks and store them as ephemeral.

        Uses spaCy NER for named entity extraction plus pattern matching
        for database, OpenAPI, and GraphQL schemas.

        Args:
            chunks: Document chunks to extract entities from
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return

        logger.debug(f"Extracting entities from {len(chunks)} chunks")

        # Create extractor with all known schema entities
        config = ExtractionConfig(
            extract_schema=bool(self._schema_entities),
            extract_ner=True,
            schema_entities=self._schema_entities or [],
            openapi_operations=self._openapi_operations,
            openapi_schemas=self._openapi_schemas,
            graphql_types=self._graphql_types,
            graphql_fields=self._graphql_fields,
        )
        extractor = EntityExtractor(config)

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
            import inspect
            sig = inspect.signature(self._vector_store.add_entities)
            if 'ephemeral' in sig.parameters:
                self._vector_store.add_entities(entities, ephemeral=True)
            else:
                self._vector_store.add_entities(entities)

        if all_links:
            sig = inspect.signature(self._vector_store.link_chunk_entities)
            if 'ephemeral' in sig.parameters:
                self._vector_store.link_chunk_entities(all_links, ephemeral=True)
            else:
                self._vector_store.link_chunk_entities(all_links)

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

    def get_ephemeral_documents(self) -> dict[str, dict]:
        """Get documents added during this session (not in config).

        Returns:
            Dict of {name: {format, char_count, loaded_at}} for ephemeral docs
        """
        # Get names of documents from config (documents is a dict keyed by name)
        config_doc_names = set(self.config.documents.keys()) if self.config.documents else set()

        # Find documents that were added but not in config
        ephemeral = {}
        for name, doc in self._loaded_documents.items():
            if name not in config_doc_names:
                ephemeral[name] = {
                    "format": doc.format,
                    "char_count": len(doc.content),
                    "loaded_at": doc.loaded_at,
                }

        return ephemeral

    def refresh(self, force_full: bool = False) -> dict:
        """Refresh documents, using incremental update by default.

        Args:
            force_full: If True, force full rebuild (clear all caches)

        Returns:
            Dict with refresh statistics: {added, updated, removed, unchanged}
        """
        if force_full:
            self._loaded_documents.clear()
            self._vector_store.clear()
            if hasattr(self._vector_store, 'clear_entities'):
                self._vector_store.clear_entities()

            # Reload all documents and rebuild index
            stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "mode": "full_rebuild"}
            if self.config.documents:
                for name in self.config.documents:
                    try:
                        self._load_document(name)
                        stats["added"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {name}: {e}")
                self._build_index(self._schema_entities)
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

    def list_documents(self) -> list[dict]:
        """
        List all configured reference documents with descriptions.

        For file-type documents with glob patterns or directory paths,
        expands to show each individual file.

        Returns:
            List of document info dicts with name, type, description, tags, path
        """
        results = []

        for doc_name, doc_config in self.config.documents.items():
            # Skip documents not allowed by permissions
            if not self._is_document_allowed(doc_name):
                continue
            # For file types, check if it's a glob/directory that needs expansion
            if doc_config.type == "file" and doc_config.path:
                expanded = _expand_file_paths(doc_config.path)

                if len(expanded) > 1:
                    # Multiple files - list each one with collection context
                    collection_desc = doc_config.description or f"Collection: {doc_name}"
                    for filename, filepath in expanded:
                        full_name = f"{doc_name}:{filename}"
                        results.append({
                            "name": full_name,
                            "type": doc_config.type,
                            "description": f"File in collection. Collection description: {collection_desc}",
                            "format": doc_config.format or self._detect_format(filepath.suffix),
                            "tags": doc_config.tags,
                            "path": str(filepath),
                            "collection": doc_name,
                            "collection_description": collection_desc,
                            "loaded": full_name in self._loaded_documents,
                        })
                elif len(expanded) == 1:
                    # Single file
                    filename, filepath = expanded[0]
                    results.append({
                        "name": doc_name,
                        "type": doc_config.type,
                        "description": doc_config.description or f"File: {filename}",
                        "format": doc_config.format or self._detect_format(filepath.suffix),
                        "tags": doc_config.tags,
                        "path": str(filepath),
                        "loaded": doc_name in self._loaded_documents,
                    })
                else:
                    # No files matched - still list it (will error on load)
                    results.append({
                        "name": doc_name,
                        "type": doc_config.type,
                        "description": doc_config.description or f"Document: {doc_name}",
                        "format": doc_config.format,
                        "tags": doc_config.tags,
                        "path": doc_config.path,
                        "loaded": doc_name in self._loaded_documents,
                    })
            else:
                # Non-file types (inline, http, etc.)
                entry = {
                    "name": doc_name,
                    "type": doc_config.type,
                    "description": doc_config.description or f"Document: {doc_name}",
                    "format": doc_config.format,
                    "tags": doc_config.tags,
                    "loaded": doc_name in self._loaded_documents,
                }
                # Include source location where applicable
                if doc_config.type == "http" and doc_config.url:
                    entry["url"] = doc_config.url
                elif doc_config.type == "inline":
                    entry["source"] = "inline content"
                results.append(entry)

        return results

    def get_document(self, name: str) -> dict:
        """
        Get the full content of a document.

        Args:
            name: Document name. For expanded glob/directory entries,
                  use format "config_name:filename" (e.g., "data_files:sales.csv")

        Returns:
            Dict with document content and metadata
        """
        # Handle expanded names from glob/directory (format: "parent:filename")
        if ":" in name:
            parent_name, filename = name.split(":", 1)
            # Check permissions on parent name
            if not self._is_document_allowed(parent_name):
                return {"error": f"Access denied to document: {parent_name}"}
            if parent_name not in self.config.documents:
                return {"error": f"Document config not found: {parent_name}"}

            doc_config = self.config.documents[parent_name]
            if doc_config.type != "file" or not doc_config.path:
                return {"error": f"Document {parent_name} is not a file type"}

            # Find the specific file in the expanded paths
            expanded = _expand_file_paths(doc_config.path)
            matching = [(fn, fp) for fn, fp in expanded if fn == filename]

            if not matching:
                return {"error": f"File '{filename}' not found in {parent_name}"}

            _, filepath = matching[0]

            # Load this specific file if not cached
            if name not in self._loaded_documents:
                try:
                    self._load_file_directly(name, filepath, doc_config)
                except Exception as e:
                    return {"error": f"Failed to load file: {str(e)}"}
        else:
            # Standard document name
            # Check permissions
            if not self._is_document_allowed(name):
                return {"error": f"Access denied to document: {name}"}
            if name not in self.config.documents:
                return {"error": f"Document not found: {name}"}

            # Load if not cached
            if name not in self._loaded_documents:
                try:
                    self._load_document(name)
                except Exception as e:
                    return {"error": f"Failed to load document: {str(e)}"}

        doc = self._loaded_documents.get(name)
        if not doc:
            return {"error": f"Document not loaded: {name}"}

        result = {
            "name": doc.name,
            "content": doc.content,
            "format": doc.format,
            "sections": doc.sections,
            "loaded_at": doc.loaded_at,
        }

        # Include path for file-based documents
        if hasattr(doc.config, 'path') and doc.config.path:
            result["path"] = doc.config.path

        return result

    def search_documents(self, query: str, limit: int = 5) -> list[dict]:
        """
        Search across all documents for relevant content using semantic search.

        Args:
            query: Natural language query
            limit: Maximum results to return

        Returns:
            List of relevant document excerpts with relevance scores
        """
        if self._model is None:
            logger.debug(f"[SEARCH] No embedding model")
            return []

        # Use lock for thread-safe access - both embedding model AND DuckDB connection
        # are not thread-safe for concurrent operations
        with self._model_lock:
            # Embed the query
            query_embedding = self._model.encode([query], convert_to_numpy=True)

            # Search using vector store (also protected by lock for consistency)
            search_results = self._vector_store.search(query_embedding, limit=limit)

        logger.debug(f"[SEARCH] Found {len(search_results)} results for '{query}'")

        results = []
        for chunk_id, similarity, chunk in search_results:
            logger.debug(f"[SEARCH] Result: doc={chunk.document_name}, section={chunk.section}, score={similarity:.3f}")
            results.append({
                "document": chunk.document_name,
                # Return full chunk content (already sized at CHUNK_SIZE=800)
                # Truncating at 500 chars can cut content mid-sentence
                "excerpt": chunk.content,
                "relevance": round(similarity, 3),
                "section": chunk.section,
            })

        return results

    def search_documents_enriched(self, query: str, limit: int = 5) -> list[dict]:
        """Search documents with entity enrichment.

        Like search_documents, but includes entities mentioned in each chunk.
        Useful for understanding what concepts are discussed in relevant chunks.

        Args:
            query: Natural language query
            limit: Maximum results to return

        Returns:
            List of dicts with document, excerpt, relevance, section, and entities
        """
        if self._model is None:
            return []

        # Use enriched search if available
        if hasattr(self._vector_store, 'search_enriched'):
            # Use lock for thread-safe access - both embedding model AND DuckDB connection
            with self._model_lock:
                query_embedding = self._model.encode([query], convert_to_numpy=True)
                enriched_results = self._vector_store.search_enriched(query_embedding, limit=limit)

            results = []
            for enriched in enriched_results:
                results.append({
                    "document": enriched.chunk.document_name,
                    "excerpt": enriched.chunk.content,
                    "relevance": round(enriched.score, 3),
                    "section": enriched.chunk.section,
                    "entities": [
                        {"name": e.name, "type": e.type}
                        for e in enriched.entities
                    ],
                })
            return results

        # Fall back to regular search
        return self.search_documents(query, limit)

    def explore_entity(self, entity_name: str, limit: int = 5) -> list[dict]:
        """Find chunks mentioning the given entity.

        Use when the LLM notices a relevant entity and wants more context.
        Returns chunks ordered by relevance (mention count, recency).

        This is designed to be exposed as an LLM tool:
        {
            "name": "explore_entity",
            "description": "Find additional context about an entity (table, concept, term)
                           mentioned in the retrieved chunks. Use when you need more
                           information about something specific.",
            "parameters": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to explore"
                }
            }
        }

        Args:
            entity_name: Name of the entity to explore
            limit: Maximum number of chunks to return

        Returns:
            List of dicts with document, excerpt, section, mention_count, confidence
            Empty list if entity not found
        """
        if not hasattr(self._vector_store, 'find_entity_by_name'):
            return []

        entity = self._vector_store.find_entity_by_name(entity_name)
        if not entity:
            return []

        chunks = self._vector_store.get_chunks_for_entity(entity.id, limit=limit)

        results = []
        for chunk_id, chunk, mention_count, confidence in chunks:
            results.append({
                "document": chunk.document_name,
                "excerpt": chunk.content,
                "section": chunk.section,
                "entity": {
                    "name": entity.name,
                    "type": entity.type,
                },
                "mention_count": mention_count,
                "confidence": round(confidence, 3),
            })

        return results

    def get_document_section(self, name: str, section: str) -> dict:
        """
        Get a specific section from a document.

        Args:
            name: Document name
            section: Section header/title

        Returns:
            Dict with section content
        """
        doc = self.get_document(name)
        if "error" in doc:
            return doc

        content = doc["content"]

        # Try to find the section
        section_lower = section.lower()
        lines = content.split("\n")

        in_section = False
        section_lines = []
        section_level = 0

        for line in lines:
            # Check for markdown headers
            if line.startswith("#"):
                header_level = len(line) - len(line.lstrip("#"))
                header_text = line.lstrip("#").strip().lower()

                if section_lower in header_text:
                    in_section = True
                    section_level = header_level
                    section_lines.append(line)
                elif in_section and header_level <= section_level:
                    # End of section
                    break
                elif in_section:
                    section_lines.append(line)
            elif in_section:
                section_lines.append(line)

        if section_lines:
            return {
                "document": name,
                "section": section,
                "content": "\n".join(section_lines),
            }
        else:
            return {"error": f"Section '{section}' not found in document '{name}'"}

    def _load_document(self, name: str) -> None:
        """Load a document from its configured source."""
        from datetime import datetime

        if name not in self.config.documents:
            raise ValueError(f"Document not configured: {name}")

        doc_config = self.config.documents[name]
        content = ""
        doc_format = doc_config.format

        if doc_config.type == "inline":
            content = doc_config.content or ""
            if doc_format == "auto":
                doc_format = "text"

        elif doc_config.type == "file":
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    suffix = path.suffix.lower()

                    # Check for structured data files - use schema metadata
                    schema = _infer_structured_schema(path, doc_config.description)
                    if schema:
                        content = schema.to_metadata_doc()
                        doc_format = schema.file_format
                    # Handle binary document formats
                    elif suffix == ".pdf":
                        content = self._extract_pdf_text(path)
                        doc_format = "text"
                    elif suffix == ".docx":
                        content = self._extract_docx_text(path)
                        doc_format = "text"
                    elif suffix == ".xlsx":
                        content = self._extract_xlsx_text(path)
                        doc_format = "text"
                    elif suffix == ".pptx":
                        content = self._extract_pptx_text(path)
                        doc_format = "text"
                    else:
                        content = path.read_text()
                        if doc_format == "auto":
                            doc_format = self._detect_format(suffix)
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
                    content = self._extract_pdf_text_from_bytes(response.content)
                    doc_format = "text"
                elif "wordprocessingml" in content_type or url_lower.endswith(".docx"):
                    content = self._extract_docx_text_from_bytes(response.content)
                    doc_format = "text"
                elif "spreadsheetml" in content_type or url_lower.endswith(".xlsx"):
                    content = self._extract_xlsx_text_from_bytes(response.content)
                    doc_format = "text"
                elif "presentationml" in content_type or url_lower.endswith(".pptx"):
                    content = self._extract_pptx_text_from_bytes(response.content)
                    doc_format = "text"
                else:
                    content = response.text
                    if doc_format == "auto":
                        doc_format = self._detect_format_from_content_type(content_type)

        elif doc_config.type == "pdf":
            # Direct PDF type - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = self._extract_pdf_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"PDF file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = self._extract_pdf_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "docx":
            # Word document - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = self._extract_docx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"Word document not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = self._extract_docx_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "xlsx":
            # Excel spreadsheet - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = self._extract_xlsx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"Excel file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = self._extract_xlsx_text_from_bytes(response.content)
                doc_format = "text"

        elif doc_config.type == "pptx":
            # PowerPoint presentation - load from path or url
            if doc_config.path:
                path = Path(doc_config.path)
                if path.exists():
                    content = self._extract_pptx_text(path)
                    doc_format = "text"
                else:
                    raise FileNotFoundError(f"PowerPoint file not found: {doc_config.path}")
            elif doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = self._extract_pptx_text_from_bytes(response.content)
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
            content = self._extract_pdf_text(filepath)
            doc_format = "text"
        elif suffix == ".docx":
            content = self._extract_docx_text(filepath)
            doc_format = "text"
        elif suffix == ".xlsx":
            content = self._extract_xlsx_text(filepath)
            doc_format = "text"
        elif suffix == ".pptx":
            content = self._extract_pptx_text(filepath)
            doc_format = "text"
        else:
            content = filepath.read_text()
            if doc_format == "auto" or not doc_format:
                doc_format = self._detect_format(suffix)

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

    def _is_structured_data_format(self, doc_format: str) -> bool:
        """Check if format is structured data that shouldn't be semantically indexed."""
        # These formats are data to be queried, not text to be searched
        structured_formats = {"csv", "json", "jsonl", "parquet", "xml", "yaml", "yml"}
        return doc_format.lower() in structured_formats

    def _extract_pdf_text(self, path) -> str:
        """Extract text content from a PDF file.

        Args:
            path: Path to the PDF file

        Returns:
            Extracted text content with page markers
        """
        from pypdf import PdfReader

        reader = PdfReader(path)
        pages = []

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")

        return "\n\n".join(pages)

    def _extract_pdf_text_from_bytes(self, pdf_bytes: bytes) -> str:
        """Extract text content from PDF bytes.

        Args:
            pdf_bytes: Raw PDF file content

        Returns:
            Extracted text content with page markers
        """
        from io import BytesIO
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []

        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i}]\n{text.strip()}")

        return "\n\n".join(pages)

    def _extract_docx_text(self, path) -> str:
        """Extract text content from a Word document.

        Args:
            path: Path to the .docx file

        Returns:
            Extracted text content with paragraph separation
        """
        from docx import Document

        doc = Document(path)
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                # Check if it's a heading
                if para.style and para.style.name.startswith("Heading"):
                    level = para.style.name.replace("Heading ", "")
                    try:
                        level_num = int(level)
                        paragraphs.append(f"{'#' * level_num} {text}")
                    except ValueError:
                        paragraphs.append(text)
                else:
                    paragraphs.append(text)

        # Also extract text from tables
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(" | ".join(cells))
            if table_rows:
                paragraphs.append("\n".join(table_rows))

        return "\n\n".join(paragraphs)

    def _extract_docx_text_from_bytes(self, docx_bytes: bytes) -> str:
        """Extract text content from Word document bytes.

        Args:
            docx_bytes: Raw .docx file content

        Returns:
            Extracted text content
        """
        from io import BytesIO
        from docx import Document

        doc = Document(BytesIO(docx_bytes))
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                if para.style and para.style.name.startswith("Heading"):
                    level = para.style.name.replace("Heading ", "")
                    try:
                        level_num = int(level)
                        paragraphs.append(f"{'#' * level_num} {text}")
                    except ValueError:
                        paragraphs.append(text)
                else:
                    paragraphs.append(text)

        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(" | ".join(cells))
            if table_rows:
                paragraphs.append("\n".join(table_rows))

        return "\n\n".join(paragraphs)

    def _extract_xlsx_text(self, path) -> str:
        """Extract text content from an Excel spreadsheet.

        Args:
            path: Path to the .xlsx file

        Returns:
            Extracted text content with sheet and cell markers
        """
        from openpyxl import load_workbook

        wb = load_workbook(path, data_only=True)
        sheets = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            rows = []

            for row in sheet.iter_rows():
                cells = []
                for cell in row:
                    if cell.value is not None:
                        cells.append(str(cell.value))
                    else:
                        cells.append("")
                # Only include rows that have some content
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))

            if rows:
                sheets.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))

        return "\n\n".join(sheets)

    def _extract_xlsx_text_from_bytes(self, xlsx_bytes: bytes) -> str:
        """Extract text content from Excel spreadsheet bytes.

        Args:
            xlsx_bytes: Raw .xlsx file content

        Returns:
            Extracted text content
        """
        from io import BytesIO
        from openpyxl import load_workbook

        wb = load_workbook(BytesIO(xlsx_bytes), data_only=True)
        sheets = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            rows = []

            for row in sheet.iter_rows():
                cells = []
                for cell in row:
                    if cell.value is not None:
                        cells.append(str(cell.value))
                    else:
                        cells.append("")
                if any(c.strip() for c in cells):
                    rows.append(" | ".join(cells))

            if rows:
                sheets.append(f"[Sheet: {sheet_name}]\n" + "\n".join(rows))

        return "\n\n".join(sheets)

    def _extract_pptx_text(self, path) -> str:
        """Extract text content from a PowerPoint presentation.

        Args:
            path: Path to the .pptx file

        Returns:
            Extracted text content with slide markers
        """
        from pptx import Presentation

        prs = Presentation(path)
        slides = []

        for i, slide in enumerate(prs.slides, 1):
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())

                # Handle tables in slides
                if shape.has_table:
                    table_rows = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        table_rows.append(" | ".join(cells))
                    if table_rows:
                        slide_text.append("\n".join(table_rows))

            if slide_text:
                slides.append(f"[Slide {i}]\n" + "\n".join(slide_text))

        return "\n\n".join(slides)

    def _extract_pptx_text_from_bytes(self, pptx_bytes: bytes) -> str:
        """Extract text content from PowerPoint presentation bytes.

        Args:
            pptx_bytes: Raw .pptx file content

        Returns:
            Extracted text content
        """
        from io import BytesIO
        from pptx import Presentation

        prs = Presentation(BytesIO(pptx_bytes))
        slides = []

        for i, slide in enumerate(prs.slides, 1):
            slide_text = []

            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())

                if shape.has_table:
                    table_rows = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        table_rows.append(" | ".join(cells))
                    if table_rows:
                        slide_text.append("\n".join(table_rows))

            if slide_text:
                slides.append(f"[Slide {i}]\n" + "\n".join(slide_text))

        return "\n\n".join(slides)

    def _detect_format(self, suffix: str) -> str:
        """Detect document format from file extension."""
        format_map = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".txt": "text",
            ".html": "html",
            ".htm": "html",
            ".pdf": "pdf",
            ".docx": "docx",
            ".xlsx": "xlsx",
            ".pptx": "pptx",
        }
        return format_map.get(suffix.lower(), "text")

    def _detect_format_from_content_type(self, content_type: str) -> str:
        """Detect document format from HTTP content-type."""
        if "markdown" in content_type:
            return "markdown"
        if "html" in content_type:
            return "html"
        if "pdf" in content_type:
            return "pdf"
        return "text"

    def _build_index(self, schema_entities: Optional[list[str]] = None) -> None:
        """Build vector embeddings for document chunks.

        Note: Structured data files (CSV, JSON) are indexed via their
        schema metadata, not raw data rows.

        Args:
            schema_entities: Optional list of known schema entity names
                            (tables, columns) for entity extraction
        """
        # Clear existing index
        self._vector_store.clear()

        # Clear entities if supported
        if hasattr(self._vector_store, 'clear_entities'):
            self._vector_store.clear_entities()

        chunks = []
        for name, doc in self._loaded_documents.items():
            # Chunk the document (structured files have metadata content)
            doc_chunks = self._chunk_document(name, doc.content)
            chunks.extend(doc_chunks)

        if not chunks:
            return

        # Generate embeddings (model is loaded eagerly in __init__)
        # Use model lock for thread safety
        texts = [chunk.content for chunk in chunks]
        with self._model_lock:
            embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Add to vector store
        self._vector_store.add_chunks(chunks, embeddings)

        # Extract and store entities
        self._extract_and_store_entities(chunks, schema_entities)

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
        config = ExtractionConfig(
            extract_schema=bool(schema_entities),
            extract_ner=True,
            schema_entities=schema_entities or [],
            openapi_operations=self._openapi_operations,
            openapi_schemas=self._openapi_schemas,
            graphql_types=self._graphql_types,
            graphql_fields=self._graphql_fields,
        )
        extractor = EntityExtractor(config)

        # Extract entities from each chunk
        all_links: list[ChunkEntity] = []

        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        # Store all unique entities
        entities = extractor.get_all_entities()
        if entities:
            self._vector_store.add_entities(entities)

        # Store all chunk-entity links
        if all_links:
            self._vector_store.link_chunk_entities(all_links)

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


# Tool schemas for LLM
DOC_TOOL_SCHEMAS = [
    {
        "name": "list_documents",
        "description": "List all available reference documents with descriptions and tags. Use this to see what documentation is available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_document",
        "description": "Get the full content of a reference document. Use this to read business rules, data dictionaries, or other documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the document to retrieve",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_documents",
        "description": "Search across all documents for relevant content using semantic search. Use this to find specific information without reading entire documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what information you're looking for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document_section",
        "description": "Get a specific section from a document by header/title. Use this when you know which section contains the information you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the document",
                },
                "section": {
                    "type": "string",
                    "description": "Section header/title to retrieve",
                },
            },
            "required": ["name", "section"],
        },
    },
]
