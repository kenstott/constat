"""Document discovery tools for reference documents.

These tools allow the LLM to discover and search reference documents
on-demand rather than loading everything into the system prompt.
"""

from dataclasses import dataclass, field
from typing import Optional
import hashlib

import numpy as np
from sentence_transformers import SentenceTransformer

from constat.core.config import Config, DocumentConfig


@dataclass
class DocumentChunk:
    """A chunk of a document for embedding and search."""
    document_name: str
    content: str
    section: Optional[str] = None
    chunk_index: int = 0


@dataclass
class LoadedDocument:
    """A loaded document with content and metadata."""
    name: str
    config: DocumentConfig
    content: str
    format: str
    sections: list[str] = field(default_factory=list)
    loaded_at: Optional[str] = None


class DocumentDiscoveryTools:
    """Tools for discovering and searching reference documents on-demand."""

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 500  # Characters per chunk for embedding

    def __init__(self, config: Config):
        self.config = config
        self._loaded_documents: dict[str, LoadedDocument] = {}
        self._chunks: list[DocumentChunk] = []
        self._embeddings: Optional[np.ndarray] = None
        self._model: Optional[SentenceTransformer] = None

    def list_documents(self) -> list[dict]:
        """
        List all configured reference documents with descriptions.

        Returns:
            List of document info dicts with name, type, description, tags
        """
        results = []

        for doc_name, doc_config in self.config.documents.items():
            results.append({
                "name": doc_name,
                "type": doc_config.type,
                "description": doc_config.description or f"Document: {doc_name}",
                "format": doc_config.format,
                "tags": doc_config.tags,
                "loaded": doc_name in self._loaded_documents,
            })

        return results

    def get_document(self, name: str) -> dict:
        """
        Get the full content of a document.

        Args:
            name: Document name

        Returns:
            Dict with document content and metadata
        """
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

        return {
            "name": doc.name,
            "content": doc.content,
            "format": doc.format,
            "sections": doc.sections,
            "loaded_at": doc.loaded_at,
        }

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

        if self._model is None or self._embeddings is None or len(self._chunks) == 0:
            return []

        # Embed the query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:limit]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            relevance = float(similarities[idx])

            results.append({
                "document": chunk.document_name,
                "excerpt": chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""),
                "relevance": round(relevance, 3),
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
                from pathlib import Path
                path = Path(doc_config.path)
                if path.exists():
                    content = path.read_text()
                    if doc_format == "auto":
                        doc_format = self._detect_format(path.suffix)
                else:
                    raise FileNotFoundError(f"Document file not found: {doc_config.path}")

        elif doc_config.type == "http":
            if doc_config.url:
                import requests
                headers = doc_config.headers or {}
                response = requests.get(doc_config.url, headers=headers, timeout=30)
                response.raise_for_status()
                content = response.text
                if doc_format == "auto":
                    content_type = response.headers.get("content-type", "")
                    doc_format = self._detect_format_from_content_type(content_type)

        # TODO: Implement confluence, notion, pdf, office loaders
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

        # Invalidate index
        self._embeddings = None

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
        if self._embeddings is None:
            self._build_index()

    def _build_index(self) -> None:
        """Build vector embeddings for document chunks."""
        self._chunks = []

        for name, doc in self._loaded_documents.items():
            # Chunk the document
            chunks = self._chunk_document(name, doc.content)
            self._chunks.extend(chunks)

        if not self._chunks:
            return

        # Load embedding model
        if self._model is None:
            self._model = SentenceTransformer(self.EMBEDDING_MODEL)

        # Generate embeddings
        texts = [chunk.content for chunk in self._chunks]
        self._embeddings = self._model.encode(texts, convert_to_numpy=True)

    def _chunk_document(self, name: str, content: str) -> list[DocumentChunk]:
        """Split a document into chunks for embedding."""
        chunks = []
        current_section = None

        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        current_chunk = ""
        chunk_index = 0

        for para in paragraphs:
            # Track sections
            if para.startswith("#"):
                current_section = para.lstrip("#").strip()

            # Add to current chunk or start new one
            if len(current_chunk) + len(para) < self.CHUNK_SIZE:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk
                if current_chunk.strip():
                    chunks.append(DocumentChunk(
                        document_name=name,
                        content=current_chunk.strip(),
                        section=current_section,
                        chunk_index=chunk_index,
                    ))
                    chunk_index += 1
                current_chunk = para + "\n\n"

        # Save final chunk
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                document_name=name,
                content=current_chunk.strip(),
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
