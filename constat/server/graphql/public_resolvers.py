# Copyright (c) 2025 Kenneth Stott
# Canary: cb435dfb-75be-4e06-9214-10cdc081fa0f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for public (unauthenticated) session access."""

from __future__ import annotations

import logging
from typing import Optional

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    ArtifactContentType,
    ArtifactInfoType,
    MessagesType,
    ProofFactsType,
    PublicSessionSummaryType,
    StoredMessageType,
    StoredProofFactType,
    TableDataType,
    TableInfoType,
)
from constat.server.session_manager import ManagedSession

logger = logging.getLogger(__name__)


def _get_public_managed(info: Info, session_id: str) -> ManagedSession:
    """Validate session exists and is public. Raises ValueError otherwise (no info leakage)."""
    sm = info.context.session_manager
    managed = sm.get_session_or_none(session_id)
    if not managed:
        raise ValueError("Not found")
    if not managed.session.datastore or not managed.session.datastore.is_public():
        raise ValueError("Not found")
    return managed


@strawberry.type
class Query:
    @strawberry.field
    async def public_session(self, info: Info, session_id: str) -> PublicSessionSummaryType:
        managed = _get_public_managed(info, session_id)
        summary = None
        try:
            summary = managed.session.datastore.get_session_meta("summary")
        except (KeyError, ValueError, OSError):
            pass
        status = managed.status.value if hasattr(managed.status, "value") else str(managed.status)
        return PublicSessionSummaryType(
            session_id=managed.session_id,
            summary=summary,
            status=status,
        )

    @strawberry.field
    async def public_messages(self, info: Info, session_id: str) -> MessagesType:
        managed = _get_public_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        raw = history.load_messages_by_server_id(session_id)
        msgs = [
            StoredMessageType(
                id=m.get("id", ""),
                type=m.get("type", ""),
                content=m.get("content", ""),
                timestamp=m.get("timestamp", ""),
                step_number=m.get("stepNumber") or m.get("step_number"),
                is_final_insight=m.get("isFinalInsight") or m.get("is_final_insight"),
                step_duration_ms=m.get("stepDurationMs") or m.get("step_duration_ms"),
                role=m.get("role"),
                skills=m.get("skills"),
            )
            for m in raw
        ]
        return MessagesType(messages=msgs)

    @strawberry.field
    async def public_artifacts(self, info: Info, session_id: str) -> list[ArtifactInfoType]:
        managed = _get_public_managed(info, session_id)
        if not managed.session.datastore:
            return []
        artifacts = managed.session.datastore.list_artifacts()
        result = []
        for a in artifacts:
            full = managed.session.datastore.get_artifact_by_id(a["id"])
            metadata = full.metadata if full else None
            result.append(
                ArtifactInfoType(
                    id=a["id"],
                    name=a["name"],
                    artifact_type=a["type"],
                    step_number=a.get("step_number", 0),
                    title=a.get("title"),
                    description=a.get("description"),
                    mime_type=a.get("content_type") or "application/octet-stream",
                    created_at=a.get("created_at"),
                    is_starred=False,
                    metadata=metadata,
                    version=a.get("version", 1),
                    version_count=a.get("version_count", 1),
                )
            )
        return result

    @strawberry.field
    async def public_artifact(
        self, info: Info, session_id: str, artifact_id: int,
    ) -> ArtifactContentType:
        managed = _get_public_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("Not found")
        artifact = managed.session.datastore.get_artifact_by_id(artifact_id)
        if not artifact:
            raise ValueError("Not found")
        return ArtifactContentType(
            id=artifact.id,
            name=artifact.name,
            artifact_type=artifact.artifact_type.value,
            content=artifact.content,
            mime_type=artifact.mime_type,
            is_binary=artifact.is_binary,
        )

    @strawberry.field
    async def public_tables(self, info: Info, session_id: str) -> list[TableInfoType]:
        managed = _get_public_managed(info, session_id)
        if not managed.session.datastore:
            return []
        tables = managed.session.datastore.list_tables()
        result = []
        for t in tables:
            name = t["name"]
            if name.startswith("_"):
                continue
            schema = managed.session.datastore.get_table_schema(name)
            columns = [c["name"] for c in schema] if schema else []
            result.append(
                TableInfoType(
                    name=name,
                    row_count=t.get("row_count", 0),
                    step_number=t.get("step_number", 0),
                    columns=columns,
                    is_starred=False,
                    version=t.get("version", 1),
                    version_count=t.get("version_count", 1),
                )
            )
        return result

    @strawberry.field
    async def public_table_data(
        self, info: Info, session_id: str, table_name: str,
        page: int = 1, page_size: int = 100,
    ) -> TableDataType:
        managed = _get_public_managed(info, session_id)
        if not managed.session.datastore:
            raise ValueError("Not found")

        from constat.server.routes.data import _sanitize_df_for_json

        df = managed.session.datastore.load_dataframe(table_name)
        if df is None:
            raise ValueError("Not found")

        total_rows = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        page_df = df.iloc[start:end]

        return TableDataType(
            name=table_name,
            columns=list(page_df.columns),
            data=_sanitize_df_for_json(page_df),
            total_rows=total_rows,
            page=page,
            page_size=page_size,
            has_more=end < total_rows,
        )

    @strawberry.field
    async def public_proof_facts(self, info: Info, session_id: str) -> ProofFactsType:
        managed = _get_public_managed(info, session_id)
        from constat.storage.history import SessionHistory
        history = SessionHistory(user_id=managed.user_id or "default")
        facts_raw, summary = history.load_proof_facts_by_server_id(session_id)
        facts = [
            StoredProofFactType(
                id=f.get("id", ""),
                name=f.get("name", ""),
                description=f.get("description"),
                status=f.get("status", ""),
                value=f.get("value"),
                source=f.get("source"),
                confidence=f.get("confidence"),
                tier=f.get("tier"),
                strategy=f.get("strategy"),
                formula=f.get("formula"),
                reason=f.get("reason"),
                dependencies=f.get("dependencies", []),
                elapsed_ms=f.get("elapsed_ms"),
            )
            for f in facts_raw
        ]
        return ProofFactsType(facts=facts, summary=summary)
