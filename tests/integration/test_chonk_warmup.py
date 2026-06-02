# Copyright (c) 2025 Kenneth Stott
# Canary: 70df3457-fb06-47eb-981f-0ad7acfef0f2
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Integration tests for chonk warmup index quality.

Verifies that warmup_chonk_index correctly indexes all configured sources,
propagates source_id/domain_id to every chunk, and registers namespaces/domains.
These tests catch regressions where sources are silently skipped or chunk linkage
is broken (source_id NULL).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import duckdb
import pytest

DEMO_ROOT = Path(__file__).parents[2] / "demo"
DEMO_DATA = DEMO_ROOT / "data"

EXPECTED_SOURCES = {"sales-analytics-db", "hr-reporting-db", "inventory-db"}
EXPECTED_DOMAINS = {"global:sales-analytics", "global:hr-reporting"}
EXPECTED_NAMESPACE = "global"


@pytest.fixture(scope="module")
def chonk_store_path(tmp_path_factory):
    """Build a fresh chonk store from the demo config and return the path."""
    store_dir = tmp_path_factory.mktemp("chonk_idx")
    store_path = store_dir / "test_global.duckdb"

    # Write a minimal chonk.toml pointing at demo data — NER/SVO disabled for speed.
    toml_content = textwrap.dedent(f"""\
        [namespaces.global]
        description = "Shared company knowledge"

        [[sources]]
        name      = "sales-analytics-db"
        type      = "db_schema"
        namespace = "global"
        domain    = "sales-analytics"
        uri       = "sqlite:///{DEMO_DATA}/sales.db"

        [[sources]]
        name      = "hr-reporting-db"
        type      = "db_schema"
        namespace = "global"
        domain    = "hr-reporting"
        uri       = "sqlite:///{DEMO_DATA}/hr.db"

        [[sources]]
        name      = "inventory-db"
        type      = "db_schema"
        namespace = "global"
        domain    = "sales-analytics"
        uri       = "sqlite:///{DEMO_DATA}/inventory.db"

        [index]
        spacy_model = "en_core_web_sm"

        [index.features]
        ner          = false
        schema_vocab = false
        community    = false
        svo          = false
    """)
    toml_path = store_dir / "chonk.toml"
    toml_path.write_text(toml_content)

    # Build a minimal Config object pointing at the demo domains directory.
    from constat.core.config import Config
    from constat.server._chonk_warmup import warmup_chonk_index

    config = Config.model_validate({
        "config_dir": str(DEMO_ROOT),
        "chonk": {"toml_path": str(toml_path)},
        "projects": {},
    })

    warmup_chonk_index(config, store_path=store_path)
    return store_path


def test_all_sources_registered(chonk_store_path):
    """Every source from chonk.toml must appear in the sources table."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute("SELECT source_id FROM sources").fetchall()
    conn.close()
    registered = {r[0] for r in rows}
    missing = EXPECTED_SOURCES - registered
    assert not missing, f"Sources not registered: {missing}"


def test_source_id_not_null(chonk_store_path):
    """No chunk may have a NULL source_id — every chunk must be traceable to a source."""
    conn = duckdb.connect(str(chonk_store_path))
    null_count = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE source_id IS NULL"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    assert total > 0, "No chunks were indexed at all"
    assert null_count == 0, f"{null_count}/{total} chunks have NULL source_id"


def test_all_sources_have_chunks(chonk_store_path):
    """Every registered source must contribute at least one chunk."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute(
        "SELECT source_id, COUNT(*) AS cnt FROM embeddings GROUP BY source_id"
    ).fetchall()
    conn.close()
    indexed = {r[0]: r[1] for r in rows}
    for src in EXPECTED_SOURCES:
        assert src in indexed, f"Source '{src}' has no chunks in embeddings"
        assert indexed[src] > 0, f"Source '{src}' has 0 chunks"


def test_inventory_tables_represented(chonk_store_path):
    """inventory.db tables (inventory, shipments, warehouses) must appear in chunk content."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute(
        "SELECT content FROM embeddings WHERE source_id = 'inventory-db'"
    ).fetchall()
    conn.close()
    assert rows, "No chunks found for inventory-db"
    all_content = " ".join(r[0] for r in rows).lower()
    for table_name in ("inventory", "shipments", "warehouses"):
        assert table_name in all_content, (
            f"Table '{table_name}' from inventory.db not found in any chunk content"
        )


def test_domain_id_matches_source(chonk_store_path):
    """domain_id on each chunk must match the domain registered for that source."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute("""
        SELECT e.source_id, e.domain_id, s.domain_id AS expected_domain
        FROM embeddings e
        JOIN sources s ON e.source_id = s.source_id
        WHERE e.domain_id != s.domain_id
    """).fetchall()
    conn.close()
    assert not rows, f"Chunk domain_id mismatch for {len(rows)} chunks: {rows[:3]}"


def test_namespace_registered(chonk_store_path):
    """The 'global' namespace must be present in the namespaces table."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute("SELECT namespace_id FROM namespaces").fetchall()
    conn.close()
    ns_ids = {r[0] for r in rows}
    assert EXPECTED_NAMESPACE in ns_ids, f"Namespace '{EXPECTED_NAMESPACE}' not registered; found: {ns_ids}"


def test_domains_registered(chonk_store_path):
    """Both demo domains must be registered under the global namespace."""
    conn = duckdb.connect(str(chonk_store_path))
    rows = conn.execute("SELECT domain_id FROM domains").fetchall()
    conn.close()
    registered = {r[0] for r in rows}
    missing = EXPECTED_DOMAINS - registered
    assert not missing, f"Domains not registered: {missing}"
