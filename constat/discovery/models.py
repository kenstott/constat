# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data models for document discovery."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any

from constat.core.config import DocumentConfig


def normalize_entity_name(name: str) -> str:
    """Normalize an entity name for display.

    Converts underscores and hyphens to spaces for readability.

    Args:
        name: Raw entity name (e.g., "employee_salary", "user-profile")

    Returns:
        Normalized name (e.g., "employee salary", "user profile")
    """
    # Replace underscores and hyphens with spaces
    normalized = re.sub(r'[_-]+', ' ', name)
    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def extract_resource_from_path(path: str) -> str:
    """Extract the primary resource name from an API path.

    Extracts the last meaningful path segment, ignoring path parameters.

    Args:
        path: API path (e.g., "/breeds", "/users/{id}/orders", "/api/v1/products")

    Returns:
        Resource name (e.g., "breeds", "orders", "products")
    """
    # Split path into segments
    segments = [s for s in path.split('/') if s]

    # Filter out path parameters ({id}, {userId}, etc.) and common prefixes
    ignore_patterns = {'api', 'v1', 'v2', 'v3'}
    meaningful_segments = []

    for segment in segments:
        # Skip path parameters
        if segment.startswith('{') and segment.endswith('}'):
            continue
        # Skip common API prefixes
        if segment.lower() in ignore_patterns:
            continue
        meaningful_segments.append(segment)

    if not meaningful_segments:
        # Fallback to last segment even if it's a parameter
        return segments[-1] if segments else path

    # Return the last meaningful segment (the primary resource)
    return meaningful_segments[-1]


class EntityType:
    """Entity type constants."""
    # Database schema
    TABLE = "table"
    COLUMN = "column"
    # API schemas
    API_ENDPOINT = "api_endpoint"
    API_SCHEMA = "api_schema"
    GRAPHQL_TYPE = "graphql_type"
    GRAPHQL_FIELD = "graphql_field"
    # Business/domain
    CONCEPT = "concept"
    BUSINESS_TERM = "business_term"
    # spaCy NER types
    ORGANIZATION = "organization"
    PRODUCT = "product"
    LOCATION = "location"
    PERSON = "person"
    EVENT = "event"
    METRIC = "metric"


@dataclass
class Entity:
    """An extracted entity from documents or schema."""
    id: str
    name: str
    type: str  # EntityType constant
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ChunkEntity:
    """Link between a chunk and an entity with exact mention text."""
    chunk_id: str
    entity_id: str
    mention_count: int = 1
    confidence: float = 1.0
    mention_text: Optional[str] = None  # Exact text as it appeared in source


@dataclass
class EnrichedChunk:
    """A chunk with its similarity score and associated entities."""
    chunk: "DocumentChunk"
    score: float
    entities: list[Entity] = field(default_factory=list)


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
    file_mtime: Optional[float] = None  # File modification time for change detection
    content_hash: Optional[str] = None  # Hash of content for change detection


@dataclass
class StructuredFileSchema:
    """Inferred schema for a structured data file."""
    filename: str
    filepath: str
    file_format: str  # csv, json, jsonl, parquet
    row_count: Optional[int] = None
    columns: list[dict] = field(default_factory=list)  # [{name, type, sample_values}]
    description: Optional[str] = None

    def to_metadata_doc(self) -> str:
        """Generate a metadata document for indexing."""
        lines = [
            f"Structured Data File: {self.filename}",
            f"Path: {self.filepath}",
            f"Format: {self.file_format}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.row_count is not None:
            lines.append(f"Row count: {self.row_count}")

        if self.columns:
            lines.append("\nColumns:")
            for col in self.columns:
                col_line = f"  - {col['name']} ({col.get('type', 'unknown')})"
                if col.get('sample_values'):
                    samples = col['sample_values'][:5]  # Limit samples
                    col_line += f": {samples}"
                lines.append(col_line)

        return "\n".join(lines)
