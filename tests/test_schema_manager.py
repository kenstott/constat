# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for schema introspection, caching, and vector search."""

import pytest
from pathlib import Path

from constat.core.config import Config
from constat.catalog.schema_manager import SchemaManager


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent
CONFIG_PATH = FIXTURES_DIR / "config.yaml"
CHINOOK_DB = FIXTURES_DIR / "data" / "chinook.db"


@pytest.fixture(scope="module")
def config() -> Config:
    """Load test config - skip env var substitution for API key."""
    # Create a test config without requiring ANTHROPIC_API_KEY
    return Config(
        databases={
            "chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}
        },
        system_prompt="Test system prompt",
    )


@pytest.fixture(scope="module")
def schema_manager(config: Config) -> SchemaManager:
    """Initialize schema manager with Chinook database."""
    manager = SchemaManager(config)
    manager.initialize()
    return manager


class TestConfig:
    """Test configuration loading."""

    def test_config_from_dict(self):
        """Config can be created from dict."""
        config = Config(
            databases={"test": {"uri": "sqlite:///test.db"}},
            system_prompt="Test prompt",
        )
        assert "test" in config.databases
        assert config.system_prompt == "Test prompt"

    def test_config_defaults(self):
        """Config has sensible defaults."""
        config = Config()
        assert config.llm.provider == "anthropic"
        assert config.execution.timeout_seconds == 60
        assert config.databases == {}


class TestSchemaManagerIntrospection:
    """Test database introspection and metadata caching."""

    def test_connects_to_database(self, schema_manager: SchemaManager):
        """Schema manager connects to configured database."""
        assert "chinook" in schema_manager.connections

    def test_introspects_all_tables(self, schema_manager: SchemaManager):
        """All Chinook tables are introspected."""
        tables = schema_manager.list_tables()
        expected_tables = [
            "chinook.Album",
            "chinook.Artist",
            "chinook.Customer",
            "chinook.Employee",
            "chinook.Genre",
            "chinook.Invoice",
            "chinook.InvoiceLine",
            "chinook.MediaType",
            "chinook.Playlist",
            "chinook.PlaylistTrack",
            "chinook.Track",
        ]
        assert sorted(tables) == sorted(expected_tables)

    def test_introspects_columns(self, schema_manager: SchemaManager):
        """Tables have correct columns."""
        track_schema = schema_manager.get_table_schema("chinook.Track")

        column_names = [c["name"] for c in track_schema["columns"]]
        assert "TrackId" in column_names
        assert "Name" in column_names
        assert "AlbumId" in column_names
        assert "GenreId" in column_names
        assert "UnitPrice" in column_names

    def test_identifies_primary_keys(self, schema_manager: SchemaManager):
        """Primary keys are correctly identified."""
        artist_schema = schema_manager.get_table_schema("chinook.Artist")
        assert artist_schema["primary_keys"] == ["ArtistId"]

        # Check column is marked as PK
        pk_cols = [c for c in artist_schema["columns"] if c.get("primary_key")]
        assert len(pk_cols) == 1
        assert pk_cols[0]["name"] == "ArtistId"

    def test_identifies_foreign_keys(self, schema_manager: SchemaManager):
        """Foreign keys are correctly identified from DB constraints."""
        track_schema = schema_manager.get_table_schema("chinook.Track")

        fk_targets = {fk["to"] for fk in track_schema["foreign_keys"]}
        # Track has FKs to Album, Genre, MediaType
        assert "Album.AlbumId" in fk_targets
        assert "Genre.GenreId" in fk_targets
        assert "MediaType.MediaTypeId" in fk_targets

    def test_identifies_reverse_references(self, schema_manager: SchemaManager):
        """Tables know which other tables reference them."""
        album_schema = schema_manager.get_table_schema("chinook.Album")

        # Track.AlbumId references Album
        assert "Track.AlbumId" in album_schema["referenced_by"]

    def test_counts_rows(self, schema_manager: SchemaManager):
        """Row counts are populated."""
        track_schema = schema_manager.get_table_schema("chinook.Track")
        assert track_schema["row_count"] == 3503

        artist_schema = schema_manager.get_table_schema("chinook.Artist")
        assert artist_schema["row_count"] == 275

    def test_simplifies_types(self, schema_manager: SchemaManager):
        """SQL types are simplified for token efficiency."""
        track_schema = schema_manager.get_table_schema("chinook.Track")

        # Find columns by name and check types
        cols_by_name = {c["name"]: c for c in track_schema["columns"]}

        # INTEGER -> int
        assert cols_by_name["TrackId"]["type"] == "int"
        # NVARCHAR -> str
        assert cols_by_name["Name"]["type"] == "str"

    def test_get_table_by_short_name(self, schema_manager: SchemaManager):
        """Can look up table by short name when unambiguous."""
        # "Track" should resolve to "chinook.Track"
        schema = schema_manager.get_table_schema("Track")
        assert schema["table"] == "Track"
        assert schema["database"] == "chinook"


class TestSchemaManagerOverview:
    """Test token-optimized overview generation."""

    def test_overview_lists_databases(self, schema_manager: SchemaManager):
        """Overview includes database names and table counts."""
        overview = schema_manager.get_overview()

        assert "chinook" in overview
        assert "db_chinook" in overview

    def test_overview_lists_tables(self, schema_manager: SchemaManager):
        """Overview includes table names."""
        overview = schema_manager.get_overview()

        assert "Track" in overview
        assert "Album" in overview
        assert "Artist" in overview
        assert "Invoice" in overview

    def test_overview_shows_relationships(self, schema_manager: SchemaManager):
        """Overview includes key relationships."""
        overview = schema_manager.get_overview()

        assert "Key relationships:" in overview
        # Should show FK relationships
        assert "→" in overview

    def test_overview_is_compact(self, schema_manager: SchemaManager):
        """Overview stays under token budget."""
        overview = schema_manager.get_overview()

        # Rough estimate: 4 chars per token
        estimated_tokens = len(overview) / 4
        assert estimated_tokens < 500, f"Overview too long: ~{estimated_tokens} tokens"


@pytest.fixture(scope="module")
def schema_manager_with_vectors(schema_manager: SchemaManager) -> SchemaManager:
    """Schema manager with build_chunks() called for vector search."""
    schema_manager.build_chunks()
    return schema_manager


class TestSchemaManagerVectorSearch:
    """Test semantic search over table schemas."""

    def test_finds_relevant_tables_for_revenue_query(self, schema_manager_with_vectors: SchemaManager):
        """Vector search finds invoice-related tables for revenue queries."""
        results = schema_manager_with_vectors.find_relevant_tables("customer revenue and sales totals")

        table_names = [r["table"] for r in results]
        # Should find Invoice, InvoiceLine, Customer
        assert "Invoice" in table_names or "InvoiceLine" in table_names

    def test_finds_relevant_tables_for_music_query(self, schema_manager_with_vectors: SchemaManager):
        """Vector search finds music-related tables for track queries."""
        results = schema_manager_with_vectors.find_relevant_tables("song tracks albums and artists")

        table_names = [r["table"] for r in results]
        # Should find Track, Album, Artist
        assert "Track" in table_names
        assert "Album" in table_names or "Artist" in table_names

    def test_finds_relevant_tables_for_genre_query(self, schema_manager_with_vectors: SchemaManager):
        """Vector search finds Genre table for genre queries."""
        results = schema_manager_with_vectors.find_relevant_tables("music genres like rock and jazz")

        table_names = [r["table"] for r in results]
        assert "Genre" in table_names

    def test_returns_relevance_scores(self, schema_manager_with_vectors: SchemaManager):
        """Results include relevance scores between 0 and 1."""
        results = schema_manager_with_vectors.find_relevant_tables("customer purchases")

        for result in results:
            assert "relevance" in result
            assert 0 <= result["relevance"] <= 1

    def test_returns_requested_number_of_results(self, schema_manager_with_vectors: SchemaManager):
        """top_k parameter limits results."""
        results_3 = schema_manager_with_vectors.find_relevant_tables("data", top_k=3)
        results_5 = schema_manager_with_vectors.find_relevant_tables("data", top_k=5)

        assert len(results_3) == 3
        assert len(results_5) == 5

    def test_results_ordered_by_relevance(self, schema_manager_with_vectors: SchemaManager):
        """Results are ordered by descending relevance."""
        results = schema_manager_with_vectors.find_relevant_tables("customer invoices")

        relevances = [r["relevance"] for r in results]
        assert relevances == sorted(relevances, reverse=True)

    def test_results_include_summary(self, schema_manager_with_vectors: SchemaManager):
        """Results include a brief summary with column names."""
        results = schema_manager_with_vectors.find_relevant_tables("track names")

        for result in results:
            assert "summary" in result
            assert result["table"] in result["summary"]
            assert "rows" in result["summary"]


class TestSchemaManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_get_nonexistent_table_raises(self, schema_manager: SchemaManager):
        """Getting nonexistent table raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            schema_manager.get_table_schema("NonexistentTable")

    def test_get_connection(self, schema_manager: SchemaManager):
        """Can get SQLAlchemy engine for queries."""
        engine = schema_manager.get_connection("chinook")
        assert engine is not None

        # Can execute queries
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM Track"))
            assert result.scalar() == 3503

    def test_get_nonexistent_connection_raises(self, schema_manager: SchemaManager):
        """Getting nonexistent connection raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            schema_manager.get_connection("nonexistent_db")


@pytest.fixture
def fresh_vector_store(tmp_path):
    """Provide a fresh vector store for tests that need complete initialization.

    This overrides the session-scoped vector store with a function-scoped one,
    ensuring each test gets a clean database without cached embeddings.
    Also clears the schema cache to force re-introspection.
    """
    import os
    from pathlib import Path

    # Clear schema cache to force re-introspection
    schema_cache_path = Path.cwd() / ".constat" / "schema_cache.json"
    cache_existed = schema_cache_path.exists()
    cache_backup = None
    if cache_existed:
        cache_backup = schema_cache_path.read_text()
        schema_cache_path.unlink()

    # Set fresh vector store path
    vector_store_path = tmp_path / "vectors.duckdb"
    old_value = os.environ.get("CONSTAT_VECTOR_STORE_PATH")
    os.environ["CONSTAT_VECTOR_STORE_PATH"] = str(vector_store_path)

    yield vector_store_path

    # Restore environment
    if old_value is not None:
        os.environ["CONSTAT_VECTOR_STORE_PATH"] = old_value
    else:
        os.environ.pop("CONSTAT_VECTOR_STORE_PATH", None)

    # Restore schema cache
    if cache_backup is not None:
        schema_cache_path.write_text(cache_backup)


class TestProgressCallback:
    """Test initialization progress callback."""

    def test_progress_callback_called(self, config: Config, fresh_vector_store):
        """Progress callback is called during initialization."""
        progress_events = []

        def callback(stage: str, current: int, total: int, detail: str):
            progress_events.append((stage, current, total, detail))

        manager = SchemaManager(config)
        manager.initialize(progress_callback=callback)

        # Should have received progress events
        assert len(progress_events) > 0

        # Should have connecting stage
        connecting_events = [e for e in progress_events if e[0] == "connecting"]
        assert len(connecting_events) > 0

        # Should have introspecting stage
        introspecting_events = [e for e in progress_events if e[0] == "introspecting"]
        assert len(introspecting_events) > 0

    def test_progress_callback_stages_in_order(self, config: Config, fresh_vector_store):
        """Progress stages occur in correct order."""
        stages = []

        def callback(stage: str, current: int, total: int, detail: str):
            if stage not in stages:
                stages.append(stage)

        manager = SchemaManager(config)
        manager.initialize(progress_callback=callback)

        # initialize() has connecting -> introspecting stages
        assert stages.index("connecting") < stages.index("introspecting")

    def test_progress_callback_current_increments(self, config: Config, fresh_vector_store):
        """Current value increments within each stage."""
        introspect_currents = []

        def callback(stage: str, current: int, total: int, detail: str):
            if stage == "introspecting":
                introspect_currents.append(current)

        manager = SchemaManager(config)
        manager.initialize(progress_callback=callback)

        # Currents should increment
        if len(introspect_currents) > 1:
            for i in range(1, len(introspect_currents)):
                assert introspect_currents[i] >= introspect_currents[i - 1]

    def test_no_callback_works(self, config: Config, fresh_vector_store):
        """Initialize works without callback."""
        manager = SchemaManager(config)
        manager.initialize()  # Should not raise

        # Manager should be functional
        tables = manager.list_tables()
        assert len(tables) > 0

    def test_callback_receives_detail_info(self, config: Config, fresh_vector_store):
        """Callback receives meaningful detail strings."""
        details = []

        def callback(stage: str, current: int, total: int, detail: str):
            details.append(detail)

        manager = SchemaManager(config)
        manager.initialize(progress_callback=callback)

        # Should have non-empty details
        non_empty_details = [d for d in details if d]
        assert len(non_empty_details) > 0

        # Should mention database/table names or stages
        all_details = " ".join(details)
        assert "chinook" in all_details.lower() or "table" in all_details.lower() or "model" in all_details.lower()
