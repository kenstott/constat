# Copyright (c) 2025 Kenneth Stott
# Canary: 29eacf30-27ef-412f-9780-7c3de09263a1
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for tables, artifacts, facts, and entities."""

from __future__ import annotations

import logging
from typing import Optional

import strawberry
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    AddEntityToGlossaryResultType,
    ArtifactContentType,
    ArtifactInfoType,
    ArtifactListType,
    ArtifactVersionInfoType,
    ArtifactVersionsType,
    DeleteResultType,
    EntityInfoType,
    EntityListType,
    EntityReferenceInfoType,
    FactInfoType,
    FactListType,
    FactMutationResultType,
    MoveFactResultType,
    TableDataType,
    TableInfoType,
    TableListType,
    TableVersionInfoType,
    TableVersionsType,
    ToggleStarResultType,
)

logger = logging.getLogger(__name__)


def _get_managed(info: Info, session_id: str):
    sm = info.context.session_manager
    managed = sm.get_session_or_none(session_id)
    if not managed:
        raise ValueError(f"Session {session_id} not found")
    return managed


@strawberry.type
class Query:
    @strawberry.field
    async def tables(self, info: Info, session_id: str) -> TableListType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            return TableListType(tables=[], total=0)

        tables = managed.session.datastore.list_tables()
        starred_tables = set(managed.session.datastore.get_starred_tables())
        unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])

        internal_tables = {"execution_history", "_facts", "_metadata"}

        result_tables = []
        for t in tables:
            table_name = t["name"]
            if table_name.startswith("_") or table_name in internal_tables:
                continue

            is_published = t.get("is_published", False)
            is_final_step = t.get("is_final_step", False)
            has_data = t.get("row_count", 0) > 0

            if table_name in starred_tables:
                is_starred = True
            elif table_name in unstarred_tables:
                is_starred = False
            else:
                is_starred = is_published or (is_final_step and has_data)

            result_tables.append(
                TableInfoType(
                    name=table_name,
                    row_count=t.get("row_count", 0),
                    step_number=t.get("step_number", 0),
                    columns=t.get("columns", []),
                    is_starred=is_starred,
                    is_view=t.get("is_view", False),
                    version=t.get("version", 1),
                    version_count=t.get("version_count", 1),
                )
            )

        return TableListType(tables=result_tables, total=len(result_tables))

    @strawberry.field
    async def table_data(
        self, info: Info, session_id: str, table_name: str,
        page: int = 1, page_size: int = 100,
    ) -> TableDataType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore for this session")

        from constat.server.routes.data import _sanitize_df_for_json

        df = managed.session.datastore.load_dataframe(table_name)
        if df is None:
            raise ValueError(f"Table not found: {table_name}")

        total_rows = len(df)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]

        return TableDataType(
            name=table_name,
            columns=list(df.columns),
            data=_sanitize_df_for_json(page_df),
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            has_more=end_idx < total_rows,
        )

    @strawberry.field
    async def table_versions(
        self, info: Info, session_id: str, table_name: str,
    ) -> TableVersionsType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore for this session")

        versions = managed.session.datastore.get_table_versions(table_name)
        if not versions:
            raise ValueError(f"Table not found: {table_name}")

        return TableVersionsType(
            name=table_name,
            current_version=versions[0]["version"] if versions else 1,
            versions=[
                TableVersionInfoType(
                    version=v["version"],
                    step_number=v.get("step_number"),
                    row_count=v.get("row_count", 0),
                    created_at=v.get("created_at"),
                )
                for v in versions
            ],
        )

    @strawberry.field
    async def table_version_data(
        self, info: Info, session_id: str, table_name: str, version: int,
        page: int = 1, page_size: int = 100,
    ) -> TableDataType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore for this session")

        from constat.server.routes.data import _sanitize_df_for_json

        df = managed.session.datastore.load_table_version(table_name, version)
        if df is None:
            raise ValueError(f"Table version not found: {table_name} v{version}")

        total_rows = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        return TableDataType(
            name=table_name,
            columns=list(df.columns),
            data=_sanitize_df_for_json(page_df),
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            has_more=end < total_rows,
        )

    @strawberry.field
    async def artifacts(self, info: Info, session_id: str) -> ArtifactListType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            return ArtifactListType(artifacts=[], total=0)

        artifacts = managed.session.datastore.list_artifacts()
        tables = managed.session.datastore.list_tables()

        visualization_types = {'chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega', 'markdown', 'md'}
        code_types = {'code', 'python', 'sql', 'script', 'text', 'output', 'error'}

        artifact_list = []
        for artifact_item in artifacts:
            full_check = managed.session.datastore.get_artifact_by_id(artifact_item["id"])
            if full_check and full_check.metadata and full_check.metadata.get("internal"):
                continue

            # Determine starred status
            artifact_obj = managed.session.datastore.get_artifact_by_id(artifact_item["id"])
            metadata = artifact_obj.metadata if artifact_obj else {}
            artifact_type = artifact_item.get("type", "").lower()

            if "is_starred" in metadata:
                is_starred = metadata["is_starred"]
            elif artifact_type in code_types:
                is_starred = False
            elif artifact_type in visualization_types:
                is_starred = True
            else:
                is_starred = False

            full_artifact = managed.session.datastore.get_artifact_by_id(artifact_item["id"])
            artifact_metadata = full_artifact.metadata if full_artifact else None

            artifact_list.append(
                ArtifactInfoType(
                    id=artifact_item["id"],
                    name=artifact_item["name"],
                    artifact_type=artifact_item["type"],
                    step_number=artifact_item.get("step_number", 0),
                    title=artifact_item.get("title"),
                    description=artifact_item.get("description"),
                    mime_type=artifact_item.get("content_type") or "application/octet-stream",
                    created_at=artifact_item.get("created_at"),
                    is_starred=is_starred,
                    metadata=artifact_metadata,
                    version=artifact_item.get("version", 1),
                    version_count=artifact_item.get("version_count", 1),
                )
            )

        # Add consequential tables as virtual artifacts
        if tables:
            starred_tables = set(managed.session.datastore.get_starred_tables())
            unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])
            for t in tables:
                table_name = t["name"]
                if table_name.startswith("_"):
                    continue
                is_published = t.get("is_published", False)
                is_final_step = t.get("is_final_step", False)
                has_data = t.get("row_count", 0) > 0

                if table_name in starred_tables:
                    table_starred = True
                elif table_name in unstarred_tables:
                    table_starred = False
                else:
                    table_starred = is_published or (is_final_step and has_data)

                should_include = table_starred or table_name in starred_tables
                if should_include:
                    virtual_id = -hash(table_name) % 1000000
                    artifact_list.append(
                        ArtifactInfoType(
                            id=virtual_id,
                            name=table_name,
                            artifact_type="table",
                            step_number=t.get("step_number", 0),
                            title=f"Table: {table_name}",
                            description=f"{t.get('row_count', 0)} rows",
                            mime_type="application/x-dataframe",
                            is_starred=table_starred,
                        )
                    )

        return ArtifactListType(artifacts=artifact_list, total=len(artifact_list))

    @strawberry.field
    async def artifact(self, info: Info, session_id: str, artifact_id: int) -> ArtifactContentType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore for this session")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if artifact:
            return ArtifactContentType(
                id=artifact.id,
                name=artifact.name,
                artifact_type=artifact.artifact_type.value,
                content=artifact.content,
                mime_type=artifact.mime_type,
                is_binary=artifact.is_binary,
            )

        # Check virtual table IDs
        tables = managed.session.datastore.list_tables()
        for t in tables:
            table_name = t["name"]
            virtual_id = -hash(table_name) % 1000000
            if virtual_id == artifact_id:
                table_data = managed.session.datastore.get_table_data(table_name)
                if table_data is not None:
                    content = table_data.to_json(orient="records", date_format="iso")
                    return ArtifactContentType(
                        id=artifact_id,
                        name=table_name,
                        artifact_type="table",
                        content=content,
                        mime_type="application/json",
                        is_binary=False,
                    )

        raise ValueError(f"Artifact not found: {artifact_id}")

    @strawberry.field
    async def artifact_versions(
        self, info: Info, session_id: str, artifact_id: int,
    ) -> ArtifactVersionsType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore for this session")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")

        versions = managed.session.datastore.get_artifact_versions(artifact.name)
        return ArtifactVersionsType(
            name=artifact.name,
            current_version=versions[0]["version"] if versions else 1,
            versions=[
                ArtifactVersionInfoType(
                    id=v["id"],
                    version=v["version"],
                    step_number=v["step_number"],
                    attempt=v["attempt"],
                    created_at=v.get("created_at"),
                )
                for v in versions
            ],
        )

    @strawberry.field
    async def facts(self, info: Info, session_id: str) -> FactListType:
        managed = _get_managed(info, session_id)
        all_facts = managed.session.fact_resolver.get_all_facts()

        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        persisted_fact_names = set(fact_store.list_facts().keys())
        persisted_facts_data = fact_store.list_all_facts()

        facts_list = []

        # Config facts
        config_facts = managed.session.config.facts or {}
        for name, value in config_facts.items():
            facts_list.append(FactInfoType(
                name=name,
                value=value,
                source="config",
                confidence=1.0,
                is_persisted=False,
            ))

        # Session facts
        for name, fact in all_facts.items():
            if name in config_facts:
                continue
            persisted_data = persisted_facts_data.get(name)
            facts_list.append(FactInfoType(
                name=name,
                value=fact.value,
                source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                reasoning=fact.reasoning,
                confidence=getattr(fact, "confidence", None),
                is_persisted=name in persisted_fact_names,
                role_id=getattr(fact, "role_id", None),
                domain=persisted_data.get("domain", "") if persisted_data else None,
            ))

        return FactListType(facts=facts_list, total=len(facts_list))

    @strawberry.field
    async def entities(
        self, info: Info, session_id: str, entity_type: Optional[str] = None,
    ) -> EntityListType:
        managed = _get_managed(info, session_id)

        from constat.discovery.models import normalize_entity_name, display_entity_name

        TYPE_CONSOLIDATION = {
            "api_endpoint": "api_endpoint",
            "api_schema": "api_schema",
            "api_field": "api_field",
            "rest_field": "api_field",
            "rest": "api_endpoint",
            "openapi/model": "api_schema",
            "graphql_type": "graphql",
            "graphql_field": "graphql",
        }

        TYPE_PRIORITY = {
            "api_field": 95,
            "api_endpoint": 90,
            "api_schema": 85,
            "api": 82,
            "graphql": 80,
            "table": 75,
            "column": 70,
            "action": 50,
            "concept": 40,
            "business_term": 30,
            "organization": 20,
            "product": 20,
            "location": 20,
            "event": 20,
        }

        entity_map: dict[str, dict] = {}

        def consolidate_source(source_str: str) -> str:
            if source_str.startswith("schema:"):
                parts = source_str.split(".")
                if len(parts) >= 3:
                    return ".".join(parts[:2])
            return source_str

        def add_entity(
            entity_name: str,
            local_entity_type: str,
            entity_source: str,
            metadata: dict,
            entity_references: list[dict] | None = None,
            related_entities: list[dict] | None = None,
        ):
            local_entity_type = TYPE_CONSOLIDATION.get(local_entity_type, local_entity_type)

            if local_entity_type in ("table", "column") and entity_references:
                api_refs = [r for r in entity_references if r.get("document", "").startswith("api:")]
                non_api_refs = [r for r in entity_references if not r.get("document", "").startswith("api:")]
                if api_refs and not non_api_refs:
                    sections = [r.get("section", "") for r in api_refs]
                    if any("field" in s.lower() for s in sections):
                        local_entity_type = "api_field"
                    elif any("schema" in s.lower() for s in sections):
                        local_entity_type = "api_schema"
                    else:
                        local_entity_type = "api_endpoint"

            normalized = normalize_entity_name(entity_name)
            display = display_entity_name(entity_name)
            key = normalized.lower()

            original_name = metadata.get("original_name")
            if not original_name and entity_name != display and entity_name != normalized:
                original_name = entity_name
                metadata = {**metadata, "original_name": original_name}

            consolidated_source = consolidate_source(entity_source)

            if key not in entity_map:
                entity_map[key] = {
                    "id": str(hash(f"{display}")),
                    "name": display,
                    "type": local_entity_type,
                    "types": [local_entity_type],
                    "sources": [consolidated_source],
                    "metadata": metadata,
                    "references": entity_references or [],
                    "related_entities": related_entities or [],
                    "mention_count": len(entity_references) if entity_references else 0,
                    "original_name": original_name,
                }
            else:
                existing = entity_map[key]
                if local_entity_type not in existing["types"]:
                    existing["types"].append(local_entity_type)
                    if TYPE_PRIORITY.get(local_entity_type, 0) > TYPE_PRIORITY.get(existing["type"], 0):
                        existing["type"] = local_entity_type
                if consolidated_source not in existing["sources"]:
                    existing["sources"].append(consolidated_source)
                if entity_references:
                    existing_refs = {(r["document"], r["section"]) for r in existing["references"]}
                    for ref in entity_references:
                        ref_key = (ref["document"], ref["section"])
                        if ref_key not in existing_refs:
                            existing["references"].append(ref)
                            existing_refs.add(ref_key)
                    existing["mention_count"] = len(existing["references"])
                if related_entities and len(related_entities) > len(existing.get("related_entities", [])):
                    existing["related_entities"] = related_entities
                existing["metadata"].update(metadata)
                if original_name and not existing.get("original_name"):
                    existing["original_name"] = original_name

        # 1. Vector store entities
        try:
            vs = None
            if hasattr(managed.session, "doc_tools") and managed.session.doc_tools:
                vs = managed.session.doc_tools._vector_store
            if vs:
                active_domains = getattr(managed, "active_domains", []) or []
                where_clause, params = vs.entity_visibility_filter(
                    session_id, active_domains, alias="e",
                )
                result = vs.list_entities_with_refcount(where_clause, params)

                filtered_rows = []
                for row in result:
                    ent_id, name, display_name, semantic_type, ner_type, ref_count = row
                    if entity_type and semantic_type != entity_type:
                        continue
                    if ref_count == 0:
                        continue
                    filtered_rows.append(row)

                entity_ids = [row[0] for row in filtered_rows]
                all_refs = vs.batch_get_entity_references(entity_ids) if entity_ids else {}
                all_related = vs.batch_get_cooccurring_entities(entity_ids, session_id) if entity_ids else {}

                for row in filtered_rows:
                    ent_id, name, display_name, semantic_type, ner_type, ref_count = row
                    references = [
                        {"document": doc_name, "section": section, "confidence": confidence}
                        for doc_name, section, confidence in all_refs.get(ent_id, [])
                    ]
                    source = "ner" if ner_type else "schema"
                    related = all_related.get(ent_id, [])
                    add_entity(
                        name, semantic_type or "concept", source,
                        {"display_name": display_name, "ner_type": ner_type},
                        references, related,
                    )
        except Exception as e:
            logger.warning(f"Could not get entities from vector_store: {e}")

        # 2. Schema entities
        try:
            if managed.session.schema_manager:
                metadata_cache = managed.session.schema_manager.metadata_cache
                for full_name, table_meta in metadata_cache.items():
                    db_name = table_meta.database
                    table_name = table_meta.name
                    if not entity_type or entity_type == "table":
                        add_entity(
                            table_name, "table", "schema",
                            {"database": db_name, "full_name": full_name},
                            [{"document": f"Database: {db_name}", "section": "Schema", "mentions": 1}],
                        )
                    if not entity_type or entity_type == "column":
                        for col in table_meta.columns:
                            add_entity(
                                col.name, "column", "schema",
                                {"table": table_name, "database": db_name, "dtype": col.type if col.type else None},
                                [{"document": f"Table: {table_name}", "section": f"Database: {db_name}", "mentions": 1}],
                            )
        except Exception as e:
            logger.warning(f"Could not get entities from schema_manager: {e}")

        # 3. API entities
        try:
            if managed.session.config and managed.session.config.apis:
                for api_name, api_config in managed.session.config.apis.items():
                    if not entity_type or entity_type in ("api", "api_endpoint"):
                        add_entity(
                            api_name, "api", "api",
                            {"base_url": getattr(api_config, "base_url", None)},
                            [{"document": f"API: {api_name}", "section": "Configuration", "mentions": 1}],
                        )
        except Exception as e:
            logger.warning(f"Could not get API entities: {e}")

        # 4. Document entities
        try:
            if managed.session.config and managed.session.config.documents:
                for doc_name in managed.session.config.documents.keys():
                    if not entity_type or entity_type == "concept":
                        add_entity(
                            doc_name, "concept", "document",
                            {"source": "document_config"},
                            [{"document": doc_name, "section": "Indexed Document", "mentions": 1}],
                        )
        except Exception as e:
            logger.warning(f"Could not get document entities: {e}")

        # 5. Domain entities
        try:
            active_domains = getattr(managed, "active_domains", [])
            if active_domains and managed.session.config:
                for domain_filename in active_domains:
                    domain = managed.session.config.load_domain(domain_filename)
                    if domain:
                        if not entity_type or entity_type in ("api", "api_endpoint"):
                            for api_name, api_config in domain.apis.items():
                                add_entity(
                                    api_name, "api", "api",
                                    {"base_url": getattr(api_config, "base_url", None), "domain": domain_filename},
                                    [{"document": f"API: {api_name}", "section": f"Domain: {domain_filename}", "mentions": 1}],
                                )
                        if not entity_type or entity_type == "concept":
                            for doc_name in domain.documents.keys():
                                add_entity(
                                    doc_name, "concept", "document",
                                    {"source": "domain", "domain": domain_filename},
                                    [{"document": doc_name, "section": f"Domain: {domain_filename}", "mentions": 1}],
                                )
        except Exception as e:
            logger.warning(f"Could not get entities from active domains: {e}")

        entities = []
        for e in entity_map.values():
            refs = [
                EntityReferenceInfoType(
                    document=r.get("document", ""),
                    section=r.get("section"),
                    mentions=r.get("mentions", 0),
                    mention_text=r.get("mention_text"),
                )
                for r in e.get("references", [])
            ]
            entities.append(EntityInfoType(
                id=e["id"],
                name=e["name"],
                type=e["type"],
                types=e["types"],
                sources=e["sources"],
                metadata=e.get("metadata"),
                references=refs,
                mention_count=e.get("mention_count", 0),
                original_name=e.get("original_name"),
                related_entities=e.get("related_entities"),
            ))

        return EntityListType(entities=entities, total=len(entities))


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def delete_table(
        self, info: Info, session_id: str, table_name: str,
    ) -> DeleteResultType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore")

        tables = managed.session.datastore.list_tables()
        if not any(t["name"] == table_name for t in tables):
            raise ValueError(f"Table not found: {table_name}")

        managed.session.datastore.drop_table(table_name)
        return DeleteResultType(status="deleted", name=table_name)

    @strawberry.mutation
    async def toggle_table_star(
        self, info: Info, session_id: str, table_name: str,
    ) -> ToggleStarResultType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore")

        tables = managed.session.datastore.list_tables()
        table_info = next((t for t in tables if t["name"] == table_name), None)
        if not table_info:
            raise ValueError(f"Table not found: {table_name}")

        starred_tables = set(managed.session.datastore.get_starred_tables())
        unstarred_tables = set(managed.session.datastore.get_state("_unstarred_tables") or [])

        is_published = table_info.get("is_published", False)
        is_final_step = table_info.get("is_final_step", False)
        has_data = table_info.get("row_count", 0) > 0

        if table_name in starred_tables:
            current_starred = True
        elif table_name in unstarred_tables:
            current_starred = False
        else:
            current_starred = is_published or (is_final_step and has_data)

        new_starred = not current_starred

        if new_starred:
            starred_tables.add(table_name)
            unstarred_tables.discard(table_name)
        else:
            starred_tables.discard(table_name)
            unstarred_tables.add(table_name)

        managed.session.datastore.set_starred_tables(list(starred_tables))
        managed.session.datastore.set_state("_unstarred_tables", list(unstarred_tables))

        return ToggleStarResultType(name=table_name, is_starred=new_starred)

    @strawberry.mutation
    async def delete_artifact(
        self, info: Info, session_id: str, artifact_id: int,
    ) -> DeleteResultType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if artifact:
            deleted = managed.session.datastore.delete_artifact(artifact_id)
            if not deleted:
                raise ValueError(f"Artifact not found: {artifact_id}")
            return DeleteResultType(status="deleted", name=artifact.name)

        # Check virtual table IDs
        tables = managed.session.datastore.list_tables()
        for t in tables:
            table_name = t["name"]
            virtual_id = -hash(table_name) % 1000000
            if virtual_id == artifact_id:
                managed.session.datastore.drop_table(table_name)
                return DeleteResultType(status="deleted", name=table_name)

        raise ValueError(f"Artifact not found: {artifact_id}")

    @strawberry.mutation
    async def toggle_artifact_star(
        self, info: Info, session_id: str, artifact_id: int,
    ) -> ToggleStarResultType:
        managed = _get_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("No datastore")

        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact not found: {artifact_id}")

        current = artifact.metadata.get("is_starred", False)
        new_status = not current
        managed.session.datastore.update_artifact_metadata(artifact_id, {"is_starred": new_status})

        return ToggleStarResultType(name=artifact.name, is_starred=new_status)

    @strawberry.mutation
    async def add_fact(
        self, info: Info, session_id: str, name: str, value: JSON,
        persist: bool = False,
    ) -> FactMutationResultType:
        managed = _get_managed(info, session_id)

        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=name,
            value=value,
            source=FactSource.USER_PROVIDED,
            reasoning="Added via UI",
        )

        is_persisted = False
        if persist:
            from constat.storage.facts import FactStore
            fact_store = FactStore(user_id=managed.user_id)
            fact_store.save_fact(name=name, value=value, description="Added via UI")
            is_persisted = True

        return FactMutationResultType(
            status="created",
            fact=FactInfoType(
                name=name,
                value=value,
                source=FactSource.USER_PROVIDED.value,
                is_persisted=is_persisted,
            ),
        )

    @strawberry.mutation
    async def edit_fact(
        self, info: Info, session_id: str, fact_name: str, value: JSON,
    ) -> FactMutationResultType:
        managed = _get_managed(info, session_id)

        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise ValueError(f"Fact not found: {fact_name}")

        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=value,
            source=FactSource.USER_PROVIDED,
            reasoning="Edited via UI",
        )

        return FactMutationResultType(
            status="updated",
            fact=FactInfoType(
                name=fact_name,
                value=value,
                source=FactSource.USER_PROVIDED.value,
            ),
        )

    @strawberry.mutation
    async def persist_fact(
        self, info: Info, session_id: str, fact_name: str,
    ) -> FactMutationResultType:
        managed = _get_managed(info, session_id)

        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise ValueError(f"Fact not found: {fact_name}")

        if hasattr(managed.session.fact_resolver, "persist_fact"):
            managed.session.fact_resolver.persist_fact(fact_name)

        return FactMutationResultType(status="persisted")

    @strawberry.mutation
    async def forget_fact(
        self, info: Info, session_id: str, fact_name: str,
    ) -> FactMutationResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise ValueError(f"Fact not found: {fact_name}")

        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        fact_store.delete_fact(fact_name)

        if hasattr(managed.session.fact_resolver, "_cache"):
            managed.session.fact_resolver._cache.pop(fact_name, None)

        sm.write_config_tombstone(session_id, "facts", fact_name)

        return FactMutationResultType(status="forgotten")

    @strawberry.mutation
    async def move_fact(
        self, info: Info, session_id: str, fact_name: str, to_domain: str,
    ) -> MoveFactResultType:
        import yaml as _yaml
        from pathlib import Path

        managed = _get_managed(info, session_id)

        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)

        fact_data = fact_store.get_fact(fact_name)
        if not fact_data:
            raise ValueError(f"Persisted fact not found: {fact_name}")

        fact_store.delete_fact(fact_name)

        config = managed.session.config
        to_cfg = config.load_domain(to_domain) if to_domain else None

        if to_cfg and to_cfg.source_path:
            tgt_file = Path(to_cfg.source_path).parent / "facts.yaml"
        else:
            tgt_file = fact_store.file_path

        tgt_data: dict = {}
        if tgt_file.exists():
            tgt_data = _yaml.safe_load(tgt_file.read_text()) or {}
        if "facts" not in tgt_data:
            tgt_data["facts"] = {}

        fact_data["domain"] = to_domain
        tgt_data["facts"][fact_name] = fact_data

        tgt_file.parent.mkdir(parents=True, exist_ok=True)
        tgt_file.write_text(_yaml.dump(tgt_data, default_flow_style=False, sort_keys=False))

        return MoveFactResultType(status="moved", fact_name=fact_name, to_domain=to_domain)

    @strawberry.mutation
    async def add_entity_to_glossary(
        self, info: Info, session_id: str, entity_id: str,
    ) -> AddEntityToGlossaryResultType:
        managed = _get_managed(info, session_id)

        try:
            if hasattr(managed.session, "add_to_glossary"):
                managed.session.add_to_glossary(entity_id)
                return AddEntityToGlossaryResultType(status="added", entity_id=entity_id)
        except Exception as e:
            logger.warning(f"Could not add to glossary: {e}")

        return AddEntityToGlossaryResultType(
            status="added", entity_id=entity_id, note="Glossary update pending",
        )
