# Copyright (c) 2025 Kenneth Stott
# Canary: 139f2285-9f4e-4378-bc88-2ddcef6e213c
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Integration tests for live SVO triple extraction via real LLM.

Calls the Anthropic API (Haiku) through the constat LLM stack:
  SVOExtractor → _ConstatLLMClient → AnthropicProvider → API

Tests both the free-form extract() and entity-anchored extract_entity_anchored()
code paths against a realistic database schema chunk.
"""

from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.integration

_SCHEMA_CHUNK = """
Table: orders
  - order_id (INTEGER, PRIMARY KEY)
  - customer_id (INTEGER, FOREIGN KEY → customers.customer_id)
  - product_id (INTEGER, FOREIGN KEY → products.product_id)
  - order_date (DATE)
  - quantity (INTEGER)
  - total_amount (DECIMAL)

Table: customers
  - customer_id (INTEGER, PRIMARY KEY)
  - name (VARCHAR)
  - email (VARCHAR)
  - region (VARCHAR)

Table: products
  - product_id (INTEGER, PRIMARY KEY)
  - name (VARCHAR)
  - category (VARCHAR)
  - unit_price (DECIMAL)
"""


@pytest.fixture(scope="module")
def svo_extractor():
    """Build an SVOExtractor backed by the real Anthropic Haiku model."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.fail("ANTHROPIC_API_KEY not set — required for SVO extraction integration test")

    from constat.core.config import LLMConfig
    from constat.core.source_config import ChonkModelSpec
    from constat.storage._chonk_llm import build_chonk_llm
    from chonk.graph._extractor import SVOExtractor

    spec = ChonkModelSpec(provider="anthropic", model="claude-haiku-4-5-20251001")
    llm_config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001", api_key=api_key)
    llm_client = build_chonk_llm(spec, llm_config)
    if llm_client is None:
        pytest.fail("build_chonk_llm returned None — check provider configuration")

    from constat.storage._chonk_llm import _ConstatLLMClient

    class _LoggingClient(_ConstatLLMClient):
        def complete(self, prompt: str) -> str:
            print(f"\n{'='*60}\nRAW INPUT\n{'='*60}\n{prompt}")
            result = super().complete(prompt)
            print(f"\n{'='*60}\nRAW OUTPUT\n{'='*60}\n{result}")
            return result

    llm_client.__class__ = _LoggingClient
    return SVOExtractor(llm=llm_client)


class TestSVOExtractFreeForm:
    """Tests for SVOExtractor.extract() — free-form triple extraction."""

    def test_returns_list(self, svo_extractor):
        for attempt in range(3):
            result = svo_extractor.extract(_SCHEMA_CHUNK, chunk_id="schema_chunk_1")
            if result:
                break
        print("\n--- free-form triples ---")
        for t in result:
            print(f"  {t.subject_id} --[{t.verb}]--> {t.object_id}  conf={t.confidence}")
        assert isinstance(result, list), "extract() must return a list"
        assert len(result) > 0, "expected at least one triple from schema text"

    def test_triples_have_valid_verbs(self, svo_extractor):
        from chonk.graph._svo import VERB_SET
        for attempt in range(3):
            result = svo_extractor.extract(_SCHEMA_CHUNK, chunk_id="schema_chunk_1")
            if result:
                break
        for triple in result:
            assert triple.verb in VERB_SET, f"verb '{triple.verb}' not in VERB_SET"

    def test_triples_have_confidence(self, svo_extractor):
        for attempt in range(3):
            result = svo_extractor.extract(_SCHEMA_CHUNK, chunk_id="schema_chunk_1")
            if result:
                break
        for triple in result:
            assert 0.0 <= triple.confidence <= 1.0, f"confidence {triple.confidence} out of range"

    def test_triples_reference_chunk_id(self, svo_extractor):
        for attempt in range(3):
            result = svo_extractor.extract(_SCHEMA_CHUNK, chunk_id="test_chunk_99")
            if result:
                break
        assert any(t.source_chunk_id == "test_chunk_99" for t in result), \
            "at least one triple should carry the chunk_id"

    def test_schema_relationships_extracted(self, svo_extractor):
        """orders references customers and products — expect references triples."""
        for attempt in range(3):
            result = svo_extractor.extract(_SCHEMA_CHUNK, chunk_id="schema_chunk_1")
            if result:
                break
        subjects = {t.subject_id.lower() for t in result}
        objects = {t.object_id.lower() for t in result}
        all_ids = subjects | objects
        # At least some of the three tables should appear
        known = {"orders", "customers", "products",
                 "order_id", "customer_id", "product_id"}
        assert any(k in all_ids for k in known), \
            f"expected schema entity names in triples, got: {all_ids}"


class TestSVOExtractEntityAnchored:
    """Tests for SVOExtractor.extract_entity_anchored()."""

    _ENTITIES = [
        {"id": "orders", "type": "table", "description": None},
        {"id": "customers", "type": "table", "description": None},
        {"id": "products", "type": "table", "description": None},
    ]

    def test_returns_four_tuple(self, svo_extractor):
        for attempt in range(3):
            result = svo_extractor.extract_entity_anchored(
                _SCHEMA_CHUNK, "chunk_ea_1", self._ENTITIES
            )
            triples, descriptions, aliases, rel_descs = result
            if triples and descriptions and aliases:
                break
        print("\n--- entity-anchored triples ---")
        for t in triples:
            print(f"  {t.subject_id} --[{t.verb}]--> {t.object_id}  conf={t.confidence}")
            if t.description:
                print(f"    → {t.description}")
        print("--- descriptions ---")
        for eid, desc in descriptions.items():
            print(f"  {eid}: {desc}")
        print("--- aliases ---")
        for eid, a in aliases.items():
            print(f"  {eid}: {a}")
        assert isinstance(triples, list)
        assert isinstance(descriptions, dict)
        assert isinstance(aliases, dict)
        assert isinstance(rel_descs, dict)

    def test_entity_anchored_triples_use_valid_entity_ids(self, svo_extractor):
        valid_ids = {e["id"] for e in self._ENTITIES}
        for attempt in range(3):
            triples, _, _, _ = svo_extractor.extract_entity_anchored(
                _SCHEMA_CHUNK, "chunk_ea_1", self._ENTITIES
            )
            if triples:
                break
        for t in triples:
            assert t.subject_id in valid_ids, \
                f"subject '{t.subject_id}' not in entity list"
            assert t.object_id in valid_ids, \
                f"object '{t.object_id}' not in entity list"

    def test_descriptions_generated_for_entities(self, svo_extractor):
        for attempt in range(3):
            _, descriptions, _, _ = svo_extractor.extract_entity_anchored(
                _SCHEMA_CHUNK, "chunk_ea_1", self._ENTITIES
            )
            if descriptions:
                break
        assert len(descriptions) > 0, "expected at least one entity description"
        for eid, desc in descriptions.items():
            assert isinstance(desc, str) and len(desc) > 5, \
                f"description for '{eid}' too short: {desc!r}"

    def test_aliases_generated(self, svo_extractor):
        for attempt in range(3):
            _, _, aliases, _ = svo_extractor.extract_entity_anchored(
                _SCHEMA_CHUNK, "chunk_ea_1", self._ENTITIES
            )
            if aliases:
                break
        # aliases is optional — just check types if present
        for eid, alias_list in aliases.items():
            assert isinstance(alias_list, list), f"aliases for '{eid}' must be a list"
            for a in alias_list:
                assert isinstance(a, str) and a.strip(), \
                    f"alias for '{eid}' must be a non-empty string"

    def test_rel_descriptions_match_triples(self, svo_extractor):
        for attempt in range(3):
            triples, _, _, rel_descs = svo_extractor.extract_entity_anchored(
                _SCHEMA_CHUNK, "chunk_ea_1", self._ENTITIES
            )
            if triples:
                break
        triple_keys = {f"{t.subject_id}|{t.verb}|{t.object_id}" for t in triples}
        for key in rel_descs:
            assert key in triple_keys, \
                f"rel_description key '{key}' has no matching triple"


# ---------------------------------------------------------------------------
# Full pipeline: warmup_chonk_index(Config) → svo_triples
# ---------------------------------------------------------------------------

def _create_sqlite_ecommerce(db_path: str) -> None:
    """Create a minimal ecommerce SQLite DB for warmup introspection."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            email       TEXT NOT NULL,
            region      TEXT
        );
        CREATE TABLE products (
            product_id  INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            category    TEXT,
            unit_price  REAL
        );
        CREATE TABLE orders (
            order_id     INTEGER PRIMARY KEY,
            customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
            product_id   INTEGER NOT NULL REFERENCES products(product_id),
            order_date   TEXT,
            quantity     INTEGER,
            total_amount REAL
        );
    """)
    conn.close()


def _write_chonk_toml(path: str, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
    content = f"""
[index.features]
ner          = true
schema_vocab = true
svo          = true

[llm.svo]
provider = "anthropic"
model    = "{model}"
api_key  = "{api_key}"
"""
    with open(path, "w") as f:
        f.write(content.strip())


class TestFullNERSVOPipeline:
    """End-to-end: warmup_chonk_index(Config) → NER → SVO → svo_triples."""

    def test_warmup_produces_triples(self, tmp_path):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.fail("ANTHROPIC_API_KEY not set")

        from constat.core.config import Config, LLMConfig, DomainConfig
        from constat.core.source_config import DatabaseConfig, ChonkConfig
        from constat.server._chonk_warmup import warmup_chonk_index
        from constat.embedding_loader import EmbeddingModelLoader
        from sentence_transformers import SentenceTransformer

        # Pre-load the embedding model singleton so warmup doesn't have to wait
        model = SentenceTransformer("all-MiniLM-L6-v2")
        loader = EmbeddingModelLoader.get_instance()
        loader._model = model
        loader._loaded.set()

        db_path = str(tmp_path / "ecommerce.db")
        _create_sqlite_ecommerce(db_path)

        toml_path = str(tmp_path / "chonk.toml")
        _write_chonk_toml(toml_path, api_key)

        config = Config(
            config_dir=str(tmp_path),
            llm=LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001", api_key=api_key),
            chonk=ChonkConfig(toml_path=toml_path),
            domains={
                "ecommerce": DomainConfig(
                    name="ecommerce",
                    databases={"main": DatabaseConfig(uri=f"sqlite:///{db_path}")},
                )
            },
        )

        from unittest.mock import patch
        import constat.server._chonk_warmup as _warmup_mod
        from constat.storage._chonk_llm import _ConstatLLMClient

        _real_build = _warmup_mod.__dict__.get(
            "build_chonk_llm",
            __import__("constat.storage._chonk_llm", fromlist=["build_chonk_llm"]).build_chonk_llm,
        )

        def _logging_build(spec, llm_cfg):
            client = _real_build(spec, llm_cfg)
            if client is None:
                return None

            class _Log(_ConstatLLMClient):
                def complete(self, prompt: str) -> str:
                    print(f"\n{'='*60}\nRAW INPUT\n{'='*60}\n{prompt}")
                    result = super().complete(prompt)
                    print(f"\n{'='*60}\nRAW OUTPUT\n{'='*60}\n{result}")
                    return result

            client.__class__ = _Log
            return client

        store_path = tmp_path / "chonk_global.duckdb"
        with patch("constat.storage._chonk_llm.build_chonk_llm", side_effect=_logging_build):
            warmup_chonk_index(config, store_path=store_path)

        from chonk.storage import Store
        store = Store(str(store_path))
        conn = store.vector._conn

        rows = conn.execute(
            "SELECT subject_id, verb, object_id FROM svo_triples"
        ).fetchall()
        print(f"\n--- svo_triples ({len(rows)}) ---")
        for r in rows:
            print(f"  {r[0]} --[{r[1]}]--> {r[2]}")

        assert len(rows) > 0, "warmup_chonk_index produced no SVO triples"
        all_ids = {r[0] for r in rows} | {r[2] for r in rows}
        schema_terms = {
            "orders", "customers", "products",
            "customer_id", "order_id", "product_id",
        }
        assert schema_terms & all_ids, (
            f"expected schema entity IDs in triples, got: {all_ids}"
        )
