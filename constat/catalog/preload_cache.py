"""Metadata preload cache for faster session startup.

This module provides caching of relevant table metadata based on seed patterns.
The cache is built once and loaded into context at session start, eliminating
the need for discovery tool calls for common query patterns.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from constat.core.config import Config, ContextPreloadConfig
from constat.catalog.schema_manager import SchemaManager, TableMetadata


@dataclass
class PreloadedTable:
    """Cached table metadata for context preloading."""
    database: str
    name: str
    comment: Optional[str]
    columns: list[dict]  # [{name, type, nullable, primary_key, comment?}]
    primary_keys: list[str]
    foreign_keys: list[dict]  # [{from, to}]
    row_count: int
    relevance_score: float  # How well it matched seed patterns


@dataclass
class PreloadCache:
    """Cached metadata for context preloading."""
    config_hash: str  # Hash of config to detect changes
    tables: list[PreloadedTable]
    seed_patterns: list[str]  # The patterns used to build this cache

    def to_context_string(self, max_columns_per_table: int = 30) -> str:
        """Generate a context string for inclusion in system prompt.

        Returns a compact schema representation suitable for LLM context.
        """
        if not self.tables:
            return ""

        lines = ["Preloaded schema (frequently accessed tables):"]

        for table in self.tables:
            # Table header with row count
            lines.append(f"\n{table.database}.{table.name} (~{table.row_count:,} rows)")

            if table.comment:
                lines.append(f"  Description: {table.comment}")

            # Columns (truncated if needed)
            cols = table.columns[:max_columns_per_table]
            col_strs = []
            for col in cols:
                pk_marker = "*" if col.get("primary_key") else ""
                col_strs.append(f"{pk_marker}{col['name']}:{col['type']}")

            if len(table.columns) > max_columns_per_table:
                col_strs.append(f"... +{len(table.columns) - max_columns_per_table} more")

            lines.append(f"  Columns: {', '.join(col_strs)}")

            # Foreign keys
            if table.foreign_keys:
                fk_strs = [f"{fk['from']}→{fk['to']}" for fk in table.foreign_keys]
                lines.append(f"  References: {', '.join(fk_strs)}")

        return "\n".join(lines)


class MetadataPreloadCache:
    """Manages the metadata preload cache.

    The cache is stored as JSON and loaded at session start to provide
    immediate schema context without discovery tool calls.
    """

    CACHE_FILENAME = "metadata_preload.json"

    def __init__(self, config: Config, cache_dir: Optional[Path] = None):
        """Initialize the preload cache manager.

        Args:
            config: The application config containing context_preload settings
            cache_dir: Directory to store cache file. Defaults to .constat/
        """
        self.config = config
        self.preload_config = config.context_preload

        # Determine cache directory
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Default to .constat in current directory
            self.cache_dir = Path.cwd() / ".constat"

        self.cache_file = self.cache_dir / self.CACHE_FILENAME
        self._cache: Optional[PreloadCache] = None

    def _compute_config_hash(self) -> str:
        """Compute a hash of config elements that affect the cache.

        Changes to databases or seed patterns should invalidate the cache.
        """
        hash_input = {
            "databases": {
                name: {"uri": db.uri, "type": db.type, "path": db.path}
                for name, db in self.config.databases.items()
            },
            "seed_patterns": self.preload_config.seed_patterns,
            "similarity_threshold": self.preload_config.similarity_threshold,
            "max_tables": self.preload_config.max_tables,
        }
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def load(self) -> Optional[PreloadCache]:
        """Load the cache from disk if it exists and is valid.

        Returns None if cache doesn't exist, is invalid, or config has changed.
        """
        if not self.cache_file.exists():
            return None

        try:
            with open(self.cache_file) as f:
                data = json.load(f)

            # Check if config has changed
            current_hash = self._compute_config_hash()
            if data.get("config_hash") != current_hash:
                return None  # Cache invalidated by config change

            # Parse tables
            tables = [
                PreloadedTable(**table_data)
                for table_data in data.get("tables", [])
            ]

            self._cache = PreloadCache(
                config_hash=data["config_hash"],
                tables=tables,
                seed_patterns=data.get("seed_patterns", []),
            )
            return self._cache

        except (json.JSONDecodeError, KeyError, TypeError):
            return None  # Invalid cache file

    def save(self, cache: PreloadCache) -> None:
        """Save the cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "config_hash": cache.config_hash,
            "seed_patterns": cache.seed_patterns,
            "tables": [asdict(table) for table in cache.tables],
        }

        with open(self.cache_file, "w") as f:
            json.dump(data, f, indent=2)

        self._cache = cache

    def build(self, schema_manager: SchemaManager) -> PreloadCache:
        """Build the cache from current database metadata using seed patterns.

        This should be called once during setup or when /refresh is invoked.

        Args:
            schema_manager: Initialized SchemaManager with metadata loaded

        Returns:
            The newly built PreloadCache
        """
        if not self.preload_config.seed_patterns:
            # No seed patterns configured - empty cache
            cache = PreloadCache(
                config_hash=self._compute_config_hash(),
                tables=[],
                seed_patterns=[],
            )
            self.save(cache)
            return cache

        # Use vector search to find tables matching each seed pattern
        matched_tables: dict[str, tuple[TableMetadata, float]] = {}  # full_name -> (meta, score)

        for pattern in self.preload_config.seed_patterns:
            results = schema_manager.find_relevant_tables(
                pattern,
                top_k=self.preload_config.max_tables,
            )

            for result in results:
                if result["relevance"] >= self.preload_config.similarity_threshold:
                    full_name = result["full_name"]
                    table_meta = schema_manager.metadata_cache.get(full_name)
                    if table_meta:
                        # Keep the highest relevance score if seen multiple times
                        existing_score = matched_tables.get(full_name, (None, 0))[1]
                        if result["relevance"] > existing_score:
                            matched_tables[full_name] = (table_meta, result["relevance"])

        # Sort by relevance and limit to max_tables
        sorted_matches = sorted(
            matched_tables.items(),
            key=lambda x: x[1][1],
            reverse=True,
        )[:self.preload_config.max_tables]

        # Convert to PreloadedTable
        preloaded_tables = []
        for full_name, (table_meta, score) in sorted_matches:
            # Limit columns per table
            columns = [
                {
                    "name": col.name,
                    "type": col.type,
                    "nullable": col.nullable,
                    "primary_key": col.primary_key,
                    **({"comment": col.comment} if col.comment else {}),
                }
                for col in table_meta.columns[:self.preload_config.max_columns_per_table]
            ]

            preloaded_tables.append(PreloadedTable(
                database=table_meta.database,
                name=table_meta.name,
                comment=table_meta.comment,
                columns=columns,
                primary_keys=table_meta.primary_keys,
                foreign_keys=[
                    {"from": fk.from_column, "to": f"{fk.to_table}.{fk.to_column}"}
                    for fk in table_meta.foreign_keys
                ],
                row_count=table_meta.row_count,
                relevance_score=score,
            ))

        cache = PreloadCache(
            config_hash=self._compute_config_hash(),
            tables=preloaded_tables,
            seed_patterns=self.preload_config.seed_patterns,
        )

        self.save(cache)
        return cache

    def get_context_string(self) -> str:
        """Get the preloaded schema as a context string.

        Loads from cache if available, otherwise returns empty string.
        """
        if self._cache is None:
            self._cache = self.load()

        if self._cache is None:
            return ""

        return self._cache.to_context_string(
            max_columns_per_table=self.preload_config.max_columns_per_table
        )

    def get_cached_tables(self) -> list[PreloadedTable]:
        """Get the list of cached tables."""
        if self._cache is None:
            self._cache = self.load()

        return self._cache.tables if self._cache else []

    def is_cache_valid(self) -> bool:
        """Check if a valid cache exists."""
        return self.load() is not None

    def invalidate(self) -> None:
        """Invalidate the cache by deleting the cache file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
        self._cache = None
