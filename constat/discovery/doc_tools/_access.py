# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Access mixin for DocumentDiscoveryTools — list, get, search, load."""

import logging

from ._file_extractors import (
    _detect_format,
)
from ._schema_inference import _expand_file_paths

logger = logging.getLogger(__name__)


def _loaded_doc_to_result(doc) -> dict:
    """Build a result dict from a LoadedDocument."""
    result = {
        "name": doc.name,
        "content": doc.content,
        "format": doc.format,
        "sections": doc.sections,
        "loaded_at": doc.loaded_at,
    }
    if hasattr(doc.config, 'path') and doc.config.path:
        result["path"] = doc.config.path
    return result


# noinspection PyUnresolvedReferences
class _AccessMixin:
    """Access, search, and load methods for DocumentDiscoveryTools."""

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
                            "format": doc_config.format or _detect_format(filepath.suffix),
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
                        "format": doc_config.format or _detect_format(filepath.suffix),
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
        # Check if already loaded
        if name in self._loaded_documents:
            doc = self._loaded_documents[name]
            result = {
                "name": doc.name,
                "content": doc.content,
                "format": doc.format,
                "sections": doc.sections,
                "loaded_at": doc.loaded_at,
            }
            if hasattr(doc.config, 'path') and doc.config.path:
                result["path"] = doc.config.path
            return result

        # Handle expanded names from glob/directory (format: "parent:filename")
        # But skip API-style document names (api:*, Database:*, Table:*, etc.)
        api_prefixes = ("api:", "Database:", "Table:", "__")
        if ":" in name and not any(name.startswith(p) for p in api_prefixes):
            parent_name, filename = name.split(":", 1)
            # Check permissions on parent name
            if not self._is_document_allowed(parent_name):
                return {"error": f"Access denied to document: {parent_name}"}

            # Try base config documents first
            doc_config = self.config.documents.get(parent_name)

            # Try project documents if not in base
            if not doc_config:
                for project in self.config.projects.values():
                    if project.documents and parent_name in project.documents:
                        doc_config = project.documents[parent_name]
                        break

            if not doc_config:
                return {"error": f"Document config not found: {parent_name}"}

            if doc_config.type != "file" or not doc_config.path:
                return {"error": f"Document {parent_name} is not a file type"}

            # Find the specific file in the expanded paths
            expanded = _expand_file_paths(doc_config.path)
            matching = [(fn, fp) for fn, fp in expanded if fn == filename]

            if not matching:
                return {"error": f"File '{filename}' not found in {parent_name}"}

            _, filepath = matching[0]

            # Load this specific file
            try:
                self._load_file_directly(name, filepath, doc_config)
            except Exception as e:
                return {"error": f"Failed to load file: {str(e)}"}
        else:
            # System-generated documents (API, Database, Table) skip permission checks
            # and go straight to vector store reconstruction
            system_prefixes = ("api:", "Database:", "Table:", "__")
            if any(name.startswith(p) for p in system_prefixes):
                return self._reconstruct_from_chunks(name)

            # Standard document name - check permissions
            if not self._is_document_allowed(name):
                return {"error": f"Access denied to document: {name}"}

            # Try base config documents
            doc_config = self.config.documents.get(name)

            # Try project documents if not in base
            if not doc_config:
                for proj_name, project in self.config.projects.items():
                    if project.documents and name in project.documents:
                        doc_config = project.documents[name]
                        break

            # Fallback: reconstruct from vector store chunks
            if not doc_config:
                return self._reconstruct_from_chunks(name)

            # Load from config
            try:
                result = self._load_document(name)
                # Binary files (PDF, Office) return dict directly instead of storing
                if isinstance(result, dict):
                    return result
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

    def _reconstruct_from_chunks(self, name: str) -> dict:
        """Reconstruct document content from vector store chunks.

        This is a fallback for documents that were indexed but aren't in config
        (e.g., session-uploaded documents, API metadata).

        Args:
            name: Document name to reconstruct

        Returns:
            Dict with document content or error
        """
        try:
            # Query chunks for this document
            chunks = self._vector_store._conn.execute(
                """
                SELECT content, section, chunk_index
                FROM embeddings
                WHERE document_name = ?
                ORDER BY chunk_index
                """,
                [name],
            ).fetchall()

            if not chunks:
                logger.warning(f"Document '{name}' not found in embeddings table")
                return {"error": f"Document not found: {name}"}

            # Reconstruct content from chunks
            sections = []
            content_parts = []
            for content, section, _ in chunks:
                content_parts.append(content)
                if section and section not in sections:
                    sections.append(section)

            return {
                "name": name,
                "content": "\n\n".join(content_parts),
                "format": "text",  # Assume plain text for reconstructed docs
                "sections": sections,
                "loaded_at": None,
                "reconstructed": True,
            }
        except Exception as e:
            logger.warning(f"Failed to reconstruct document {name} from chunks: {e}")
            return {"error": f"Document not found: {name}"}

    def search_documents(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[dict]:
        """
        Search across all documents for relevant content using semantic search.

        Uses this instance's active_project_ids (set when projects are loaded into the session).

        Args:
            query: Natural language query
            limit: Maximum results to return
            session_id: Session ID to include (for session-specific documents)

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

            # Search using vector store with filtering (also protected by lock for consistency)
            search_results = self._vector_store.search(
                query_embedding,
                limit=limit,
                project_ids=self._active_project_ids,
                session_id=session_id,
            )

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

    def search_documents_enriched(
        self,
        query: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[dict]:
        """Search documents with entity enrichment.

        Like search_documents, but includes entities mentioned in each chunk.
        Useful for understanding what concepts are discussed in relevant chunks.

        Uses this instance's active_project_ids (set when projects are loaded into the session).

        Args:
            query: Natural language query
            limit: Maximum results to return
            session_id: Session ID to include (for session-specific documents)

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
                enriched_results = self._vector_store.search_enriched(
                    query_embedding,
                    limit=limit,
                    project_ids=self._active_project_ids,
                    session_id=session_id,
                )

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
        return self.search_documents(query, limit, session_id)

    def explore_entity(
        self,
        entity_name: str,
        limit: int = 5,
        session_id: str | None = None,
    ) -> list[dict]:
        """Find chunks mentioning the given entity.

        Use when the LLM notices a relevant entity and wants more context.
        Returns chunks ordered by relevance (mention count, recency).

        Uses this instance's active_project_ids (set when projects are loaded into the session).

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
            session_id: Session ID to include (for session-specific documents)

        Returns:
            List of dicts with document, excerpt, section, mention_count, confidence
            Empty list if entity not found
        """
        if not hasattr(self._vector_store, 'find_entity_by_name'):
            return []

        entity = self._vector_store.find_entity_by_name(entity_name, project_ids=self._active_project_ids, session_id=session_id)
        if not entity:
            return []

        chunks = self._vector_store.get_chunks_for_entity(
            entity.id,
            limit=limit,
            project_ids=self._active_project_ids,
            session_id=session_id,
        )

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
