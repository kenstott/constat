"""Document discovery tools for reference documents.

These tools allow the LLM to discover and search reference documents
on-demand rather than loading everything into the system prompt.
"""

import glob as glob_module
from pathlib import Path
from typing import Optional
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from constat.core.config import Config, DocumentConfig
from constat.discovery.models import (
    DocumentChunk,
    LoadedDocument,
    StructuredFileSchema,
)
from constat.discovery.vector_store import (
    VectorStoreBackend,
    create_vector_store,
)


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

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    # all-MiniLM-L6-v2 has max_seq_length=256 tokens (~1024 chars)
    # Use 800 chars to stay within limit while maximizing context
    CHUNK_SIZE = 800
    CACHE_FILENAME = "doc_index_cache.json"

    def __init__(
        self,
        config: Config,
        cache_dir: Optional[Path] = None,
        vector_store: Optional[VectorStoreBackend] = None,
    ):
        self.config = config
        self._loaded_documents: dict[str, LoadedDocument] = {}
        self._model: Optional[SentenceTransformer] = None
        self._index_built = False

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

    def _create_vector_store(self) -> VectorStoreBackend:
        """Create vector store based on config."""
        storage_config = self.config.storage
        vs_config = storage_config.vector_store if storage_config else None

        backend = vs_config.backend if vs_config else "duckdb"
        db_path = vs_config.db_path if vs_config else None

        return create_vector_store(backend=backend, db_path=db_path)

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
            self._index_built = False
            return {"added": 0, "updated": 0, "removed": 0, "unchanged": 0, "mode": "full_rebuild"}

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
            self._index_built = False  # Force index rebuild

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
        # Ensure all documents are loaded and indexed
        self._ensure_indexed()

        if self._model is None or self._vector_store.count() == 0:
            return []

        # Embed the query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Search using vector store
        search_results = self._vector_store.search(query_embedding, limit=limit)

        results = []
        for chunk_id, similarity, chunk in search_results:
            results.append({
                "document": chunk.document_name,
                # Return full chunk content (already sized at CHUNK_SIZE=800)
                # Truncating at 500 chars can cut content mid-sentence
                "excerpt": chunk.content,
                "relevance": round(similarity, 3),
                "section": chunk.section,
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

        # Invalidate index (unless it's a structured data file)
        if not self._is_structured_data_format(doc_format):
            self._index_built = False

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
            # Structured files DO get indexed (via their metadata)
            self._index_built = False
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

        # Invalidate index
        self._index_built = False

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

    def _ensure_indexed(self) -> None:
        """Ensure all documents are loaded and indexed."""
        # Load any unloaded documents
        for name in self.config.documents:
            if name not in self._loaded_documents:
                try:
                    self._load_document(name)
                except Exception:
                    # Skip documents that fail to load
                    pass

        # Build index if needed
        if not self._index_built:
            self._build_index()

    def _build_index(self) -> None:
        """Build vector embeddings for document chunks.

        Note: Structured data files (CSV, JSON) are indexed via their
        schema metadata, not raw data rows.
        """
        # Clear existing index
        self._vector_store.clear()

        chunks = []
        for name, doc in self._loaded_documents.items():
            # Chunk the document (structured files have metadata content)
            doc_chunks = self._chunk_document(name, doc.content)
            chunks.extend(doc_chunks)

        if not chunks:
            self._index_built = True
            return

        # Load embedding model
        if self._model is None:
            self._model = SentenceTransformer(self.EMBEDDING_MODEL)

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Add to vector store
        self._vector_store.add_chunks(chunks, embeddings)
        self._index_built = True

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
