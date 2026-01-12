"""File-based data source connectors.

Supports CSV, JSON, Parquet, Arrow/Feather files as data sources
with schema introspection similar to SQL/NoSQL databases.
"""

from .connector import FileConnector, FileType

__all__ = ["FileConnector", "FileType"]
