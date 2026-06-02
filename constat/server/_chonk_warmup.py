# Copyright (c) 2025 Kenneth Stott
# Canary: 8db8a807-c71c-45fe-9856-27a2eca0063b
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Chonk native Store warmup — builds global index from domain DB schemas."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.core.config import Config

logger = logging.getLogger(__name__)


def _introspect_db_schema(uri: str, source_db: str) -> list:
    """Return list[TableMeta] by introspecting a SQLAlchemy-compatible DB URI."""
    from chonk import ColumnMeta, TableMeta
    from sqlalchemy import create_engine
    from sqlalchemy import inspect as sa_inspect

    engine = create_engine(uri)
    inspector = sa_inspect(engine)
    tables = []
    for table_name in inspector.get_table_names():
        pk_constraint = inspector.get_pk_constraint(table_name)
        pk_cols = set(pk_constraint.get("constrained_columns", []))
        fk_map: dict[str, str] = {}
        for fk in inspector.get_foreign_keys(table_name):
            for local_col, ref_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                fk_map[local_col] = f"{fk['referred_table']}.{ref_col}"
        columns = []
        for col in inspector.get_columns(table_name):
            col_name = col["name"]
            col_type = str(col["type"])
            columns.append(ColumnMeta(
                col_name,
                col_type,
                is_primary_key=col_name in pk_cols,
                is_foreign_key=col_name in fk_map,
                foreign_key_ref=fk_map.get(col_name),
            ))
        tables.append(TableMeta(table_name, source_db=source_db, columns=columns))
    engine.dispose()
    return tables


def warmup_chonk_index(config: Config, store_path: str | Path | None = None) -> None:
    """Build chonk native Store from chonk.toml sources.

    Indexes each domain's databases via SQLAlchemy introspection so
    EnhancedSearch can use chonk's native namespace/domain isolation.
    Skips silently if chonk.toml_path is not configured.
    """
    chonk_cfg = getattr(config, "chonk", None)
    if not chonk_cfg or not chonk_cfg.toml_path:
        return

    toml_path = Path(chonk_cfg.toml_path)
    if not toml_path.is_absolute() and config.config_dir:
        toml_path = (Path(config.config_dir) / chonk_cfg.toml_path).resolve()

    if not toml_path.exists():
        logger.warning(f"chonk.toml not found: {toml_path}")
        return

    from chonk.loader import DocumentLoader
    from chonk.storage import Store

    from constat.core.source_config import ChonkTomlConfig
    from constat.embedding_loader import EmbeddingModelLoader
    from constat.storage._chonk_registry import (
        set_domain_descriptions,
        set_global_store_path,
    )

    chonk_toml = ChonkTomlConfig.from_toml(toml_path)

    if store_path is None:
        resolved_store_path = Path(".constat/chonk_global.duckdb")
    else:
        resolved_store_path = Path(store_path)
    resolved_store_path.parent.mkdir(parents=True, exist_ok=True)

    embed_model = EmbeddingModelLoader.get_instance().get_model()
    assert embed_model is not None, "EmbeddingModelLoader returned None"
    embed_dim = embed_model.get_embedding_dimension() or 384

    store = Store(str(resolved_store_path), embedding_dim=embed_dim)

    # enrich_context=True adds breadcrumb prefix to each chunk — improves cross-domain retrieval.
    # min/max chunk sizes match the ECDR-Bench leader configuration.
    loader = DocumentLoader(min_chunk_size=1100, max_chunk_size=2200)
    config_base = Path(config.config_dir).resolve() if config.config_dir else Path.cwd()

    # Register global namespace and derive domain hierarchy from config.domains (filesystem authority).
    # parent_id is required for chonk's recursive CTE descent in resolve_domain_ids.
    store.register_namespace("global", description="Shared company knowledge")
    descriptions: dict[str, str] = {}

    def _register_domain_tree(domains_dir: Path, ns_id: str, prefix: str = "") -> None:
        if not domains_dir.is_dir():
            return
        for entry in sorted(domains_dir.iterdir()):
            if entry.is_dir() and (entry / "config.yaml").exists():
                import yaml as _yaml
                try:
                    cfg_raw = entry.read_text() if False else (entry / "config.yaml").read_text()
                    cfg_data = _yaml.safe_load(cfg_raw) or {}
                except Exception:
                    cfg_data = {}
                domain_path = f"{prefix}/{entry.name}" if prefix else entry.name
                domain_id = f"{ns_id}:{domain_path}"
                parent_path = prefix if prefix else None
                parent_id = f"{ns_id}:{parent_path}" if parent_path else None
                description = cfg_data.get("description", "")
                store.register_domain(domain_id, ns_id, domain_path, description=description, parent_id=parent_id)
                descriptions[domain_id] = description
                sub_dir = entry / "domains"
                _register_domain_tree(sub_dir, ns_id, prefix=domain_path)

    if config.config_dir:
        _register_domain_tree(Path(config.config_dir) / "domains", "global")

    # Index [[sources]] entries — domain field must match a registered domain path.
    for src in chonk_toml.sources:
        if src.type != "db_schema" or not src.uri:
            continue
        ns_id = src.namespace or "global"
        domain_path = src.domain or src.name
        domain_id = f"{ns_id}:{domain_path}"
        uri = src.uri
        if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
            uri = f"sqlite:///{config_base / uri[len('sqlite:///'):]}"
        try:
            store.register_source(src.name, domain_id, "db_schema", uri)
            tables = _introspect_db_schema(uri, source_db=src.name)
            chunks = loader.load_schema(tables)
            if not chunks:
                logger.warning(f"  chonk: no chunks from {src.name}")
                continue
            texts = [c.embedding_content or c.content for c in chunks]
            emb = embed_model.encode(texts, normalize_embeddings=True).astype("float32")
            store.add_document(chunks, emb, namespace=ns_id, source_id=src.name, domain_id=domain_id)
            logger.info(f"  chonk: indexed {src.name} ({len(chunks)} chunks)")
        except Exception as exc:
            logger.warning(f"  chonk: failed to index {src.name}: {exc}")

    set_domain_descriptions(descriptions)
    set_global_store_path(resolved_store_path)
    logger.info(f"  chonk: global store ready at {resolved_store_path}")

    force_rebuild = os.environ.get("CONSTAT_CHONK_FORCE_REBUILD", "").lower() in ("1", "true")

    if chonk_toml.index.features.ner:
        from chonk.ner import build_ner

        raw_entries = [e.model_dump() for e in chonk_toml.vocab_entries()]
        vocab_entries = []
        for entry in raw_entries:
            conn = entry.get("connection", "")
            if conn.startswith("sqlite:///") and not conn.startswith("sqlite:////"):
                rel = conn[len("sqlite:///"):]
                abs_path = config_base / rel
                entry = {**entry, "connection": f"sqlite:///{abs_path}"}
            vocab_entries.append(entry)
        vocab_entries = vocab_entries or None
        # Always enable schema_vocab when indexing DB schema sources — DatabaseSchemaCrawler
        # produces db_table/db_column chunks that SchemaVocabBuilder.add_chunks() needs.
        has_db_sources = any(
            db_cfg.uri for domain in config.projects.values()
            for db_cfg in (domain.databases or {}).values()
        )
        use_schema_vocab = chonk_toml.index.features.schema_vocab or has_db_sources
        n = build_ner(
            store,
            spacy_model=chonk_toml.index.spacy_model,
            use_schema_vocab=use_schema_vocab,
            vocab_entities=vocab_entries,
            force=force_rebuild,
        )
        logger.info(f"  chonk NER: {n} entity associations")

    if chonk_toml.index.features.svo:
        conn = store.vector._conn
        try:
            existing = conn.execute("SELECT COUNT(*) FROM svo_triples").fetchone()
            triple_count = existing[0] if existing else 0
        except Exception:
            triple_count = 0
        if force_rebuild or triple_count == 0:
            _run_svo_phase(store, config, chonk_toml=chonk_toml)
        else:
            logger.info(f"  chonk SVO: {triple_count} triples already indexed, skipping")

    if chonk_toml.index.features.ner:
        try:
            from chonk.cluster import ClusterMap

            from constat.storage._chonk_registry import set_cluster_map

            class _StoreEntityIndex:
                def __init__(self, conn):
                    self._conn = conn

                def chunk_ids(self):
                    return [r[0] for r in self._conn.execute(
                        "SELECT DISTINCT chunk_id FROM chunk_entities"
                    ).fetchall()]

                def total_chunks(self):
                    row = self._conn.execute(
                        "SELECT COUNT(DISTINCT chunk_id) FROM embeddings"
                    ).fetchone()
                    return row[0] if row else 0

                def entity_ids(self):
                    return [r[0] for r in self._conn.execute(
                        "SELECT DISTINCT entity_id FROM chunk_entities"
                    ).fetchall()]

                def get_entities_for_chunk(self, chunk_id):
                    rows = self._conn.execute(
                        "SELECT entity_id, score FROM chunk_entities WHERE chunk_id = ?",
                        [chunk_id],
                    ).fetchall()
                    return [(r[0], r[1]) for r in rows]

            conn = store.vector._conn
            ei = _StoreEntityIndex(conn)
            if ei.entity_ids():
                cluster_map = ClusterMap.build(ei)
                set_cluster_map(cluster_map)
                logger.info(f"  chonk ClusterMap: built with {len(ei.entity_ids())} entities")
        except Exception as exc:
            logger.warning(f"  chonk ClusterMap: build failed: {exc}")

    try:
        import numpy as np
        from chonk.community import CommunityIndex

        from constat.storage._chonk_registry import set_community_index

        conn = store.vector._conn
        rows = conn.execute(
            "SELECT chunk_id, embedding FROM embeddings WHERE embedding IS NOT NULL"
        ).fetchall()
        if rows:
            chunk_ids_ci = [r[0] for r in rows]
            content_vecs = np.array([r[1] for r in rows], dtype=np.float32)
            community_index = CommunityIndex.build(chunk_ids_ci, content_vecs)
            set_community_index(community_index)
            logger.info(f"  chonk CommunityIndex: {len(chunk_ids_ci)} chunks")
    except Exception as exc:
        logger.warning(f"  chonk CommunityIndex: build failed: {exc}")

    try:
        stats = store.build_context_graph(namespace="global", force=force_rebuild)
        edge_count = stats.edge_count if hasattr(stats, "edge_count") else "?"
        logger.info(f"  chonk context graph: {edge_count} edges built")
    except Exception as exc:
        logger.warning(f"  chonk context graph: build failed: {exc}")


def register_user_namespace(user_id: str, base_dir: Path | None = None) -> None:
    """Register user namespace and domains in the chonk global Store.

    Walks `.constat/{user_id}/domains/` recursively and calls register_namespace /
    register_domain so that chonk's CTE descent works for user-scoped searches.
    Safe to call multiple times — register_* calls are idempotent on the Store.
    """
    from constat.storage._chonk_registry import get_global_store_readonly

    store = get_global_store_readonly()
    if store is None:
        return

    ns_id = f"user:{user_id}"
    try:
        store.register_namespace(ns_id, description=f"User domains for {user_id}")
    except Exception:
        pass  # already registered

    if base_dir is None:
        base_dir = Path(".constat")
    from constat.core.paths import user_vault_dir
    domains_dir = user_vault_dir(base_dir, user_id) / "domains"
    if not domains_dir.is_dir():
        return

    def _walk(directory: Path, prefix: str = "") -> None:
        for entry in sorted(directory.iterdir()):
            if not entry.is_dir() or not (entry / "config.yaml").exists():
                continue
            import yaml as _yaml
            try:
                cfg_data = _yaml.safe_load((entry / "config.yaml").read_text()) or {}
            except Exception:
                cfg_data = {}
            domain_path = f"{prefix}/{entry.name}" if prefix else entry.name
            domain_id = f"{ns_id}:{domain_path}"
            parent_path = prefix if prefix else None
            parent_id = f"{ns_id}:{parent_path}" if parent_path else None
            description = cfg_data.get("description", "")
            try:
                store.register_domain(domain_id, ns_id, domain_path, description=description, parent_id=parent_id)
            except Exception:
                pass  # already registered
            sub_dir = entry / "domains"
            if sub_dir.is_dir():
                _walk(sub_dir, prefix=domain_path)

    _walk(domains_dir)
    logger.debug(f"chonk: registered user namespace {ns_id}")


def _run_svo_phase(
    store,
    config: Config,
    on_progress: Callable[[int], None] | None = None,
    chonk_toml=None,
) -> None:
    """Extract SVO triples + entity descriptions + aliases for all indexed chunks."""
    from chonk.graph._extractor import SVOExtractor
    from chonk.graph._index import RelationshipIndex

    from constat.storage._chonk_llm import build_chonk_llm

    toml_svo = chonk_toml.llm.svo if chonk_toml is not None else None
    svo_spec = toml_svo or getattr(config.chonk, "svo_llm", None)
    llm_client = build_chonk_llm(svo_spec, config.llm)
    if not llm_client:
        logger.warning("chonk SVO: no LLM configured, skipping SVO extraction")
        return

    extractor = SVOExtractor(llm=llm_client)
    conn = store.vector._conn

    chunks_rows = conn.execute("""
        SELECT e.chunk_id, e.content, array_agg(ce.entity_id) AS entity_ids
        FROM embeddings e
        JOIN chunk_entities ce ON e.chunk_id = ce.chunk_id
        GROUP BY e.chunk_id, e.content
        HAVING count(ce.entity_id) >= 2
    """).fetchall()

    all_triples = []
    all_descriptions: dict[str, str] = {}
    all_aliases: dict[str, list[str]] = {}

    for i, (chunk_id, content, raw_ids) in enumerate(chunks_rows):
        entity_ids: list[str] = list(raw_ids)
        ph = ", ".join(["?" for _ in entity_ids])
        entity_rows = conn.execute(
            f"SELECT id, entity_type FROM entities WHERE id IN ({ph})", entity_ids
        ).fetchall()
        existing_descs = store.get_entity_descriptions(entity_ids)
        entities = [
            {"id": r[0], "type": r[1], "description": existing_descs.get(r[0])}
            for r in entity_rows
        ]
        if len(entities) < 2:
            continue
        triples, new_descs, new_aliases, _ = extractor.extract_entity_anchored(content, chunk_id, entities)
        all_triples.extend(triples)
        all_descriptions.update(new_descs)
        all_aliases.update(new_aliases)
        if on_progress:
            on_progress(i + 1)

    rel_idx = RelationshipIndex()
    for t in all_triples:
        rel_idx.add(t)
    written_triples = rel_idx.save_to_db(conn)
    written_descs = store.set_entity_descriptions_batch(all_descriptions)
    flat_aliases = {alias: eid for eid, aliases in all_aliases.items() for alias in aliases}
    written_aliases = store.add_entity_aliases_batch(flat_aliases, source="llm") if flat_aliases else 0
    logger.info(f"  chonk SVO: {written_triples} triples, {written_descs} descriptions, {written_aliases} aliases")
