# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Chonk native Store warmup — builds global index from domain DB schemas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from constat.core.config import Config

logger = logging.getLogger(__name__)


def warmup_chonk_index(config: "Config") -> None:
    """Build chonk native Store from chonk.toml sources.

    Indexes each domain's databases via Indexer so EnhancedSearch can use
    chonk's native namespace/domain isolation instead of the adapter shim.
    Skips silently if chonk.toml_path is not configured.
    """
    from pathlib import Path

    chonk_cfg = getattr(config, "chonk", None)
    if not chonk_cfg or not chonk_cfg.toml_path:
        return

    toml_path = Path(chonk_cfg.toml_path)
    if not toml_path.is_absolute() and config.config_dir:
        toml_path = (Path(config.config_dir) / chonk_cfg.toml_path).resolve()

    if not toml_path.exists():
        logger.warning(f"chonk.toml not found: {toml_path}")
        return

    from chonk.storage import Store
    from chonk.indexer import get_indexer
    from constat.core.source_config import ChonkTomlConfig
    from constat.embedding_loader import EmbeddingModelLoader
    from constat.storage._chonk_registry import set_global_store_path

    chonk_toml = ChonkTomlConfig.from_toml(toml_path)

    store_path = Path(".constat/chonk_global.duckdb")
    store_path.parent.mkdir(parents=True, exist_ok=True)

    store = Store(str(store_path))
    store.register_namespace("global", description="System-wide schema and document index")

    embed_model = EmbeddingModelLoader.get_instance().get_model()
    indexer = get_indexer(
        namespace_id="global",
        store=store,
        embed_model=embed_model,
    )

    for domain_name, domain in list(config.projects.items()):
        domain_id = f"global:{domain_name}"
        store.register_domain(domain_id, "global", domain_name)

        if not domain.databases:
            continue

        for db_name, db_cfg in domain.databases.items():
            uri = db_cfg.uri or ""
            if not uri:
                continue
            source_id = f"{domain_name}:{db_name}"
            store.register_source(source_id, domain_id, "db_schema", uri)
            try:
                chunks = indexer.index_source({
                    "type": "db_schema",
                    "uri": uri,
                    "source_id": source_id,
                    "domain_id": domain_id,
                    "namespace": "global",
                })
                logger.info(f"  chonk: indexed {source_id} ({chunks} chunks)")
            except Exception as exc:
                logger.warning(f"  chonk: failed to index {source_id}: {exc}")

    set_global_store_path(store_path)
    logger.info(f"  chonk: global store ready at {store_path}")

    if chonk_toml.index.features.svo:
        _run_svo_phase(store, config, chonk_toml=chonk_toml)


def _run_svo_phase(
    store,
    config: "Config",
    on_progress: "Callable[[int], None] | None" = None,
    chonk_toml=None,
) -> None:
    """Extract SVO triples + entity descriptions + aliases for all indexed chunks.

    Args:
        store: Open chonk Store (write-path or read-write).
        config: constat Config (used to build LLM client).
        on_progress: Optional callback receiving count of chunks processed so far.
    """
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
        triples, new_descs, new_aliases = extractor.extract_entity_anchored(content, chunk_id, entities)
        all_triples.extend(triples)
        all_descriptions.update(new_descs)
        all_aliases.update(new_aliases)
        if on_progress:
            on_progress(i + 1)

    rel_idx = RelationshipIndex()
    for t in all_triples:
        rel_idx.add(t)
    written_triples = rel_idx.save_to_db(conn)
    written_descs = store.upsert_entity_descriptions_batch(all_descriptions, source="llm")
    flat_aliases = {alias: eid for eid, aliases in all_aliases.items() for alias in aliases}
    written_aliases = store.add_entity_aliases_batch(flat_aliases, source="llm") if flat_aliases else 0
    logger.info(f"  chonk SVO: {written_triples} triples, {written_descs} descriptions, {written_aliases} aliases")
