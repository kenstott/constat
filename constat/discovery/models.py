"""Data models for document discovery."""

from dataclasses import dataclass, field
from typing import Optional

from constat.core.config import DocumentConfig


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
