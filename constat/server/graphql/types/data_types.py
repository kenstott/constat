# Copyright (c) 2025 Kenneth Stott
# Canary: c3322482-e78b-4565-8a55-bf76e4fae3e5
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for tables, artifacts, facts, and entities (Phase 4)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class TableInfoType:
    name: str
    row_count: int
    step_number: int
    columns: list[str]
    is_starred: bool = False
    is_view: bool = False
    role_id: Optional[str] = None
    version: int = 1
    version_count: int = 1


@strawberry.type
class TableListType:
    tables: list[TableInfoType]
    total: int


@strawberry.type
class TableDataType:
    name: str
    columns: list[str]
    data: JSON
    total_rows: int
    page: int
    page_size: int
    has_more: bool


@strawberry.type
class TableVersionInfoType:
    version: int
    step_number: Optional[int] = None
    row_count: int = 0
    created_at: Optional[str] = None


@strawberry.type
class TableVersionsType:
    name: str
    current_version: int
    versions: list[TableVersionInfoType]


@strawberry.type
class ArtifactInfoType:
    id: int
    name: str
    artifact_type: str
    step_number: int
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: str = "application/octet-stream"
    created_at: Optional[str] = None
    is_starred: bool = False
    metadata: Optional[JSON] = None
    role_id: Optional[str] = None
    version: int = 1
    version_count: int = 1


@strawberry.type
class ArtifactListType:
    artifacts: list[ArtifactInfoType]
    total: int


@strawberry.type
class ArtifactContentType:
    id: int
    name: str
    artifact_type: str
    content: str
    mime_type: str
    is_binary: bool


@strawberry.type
class ArtifactVersionInfoType:
    id: int
    version: int
    step_number: int
    attempt: int
    created_at: Optional[str] = None


@strawberry.type
class ArtifactVersionsType:
    name: str
    current_version: int
    versions: list[ArtifactVersionInfoType]


@strawberry.type
class FactInfoType:
    name: str
    value: Optional[JSON] = None
    source: str = ""
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    is_persisted: bool = False
    role_id: Optional[str] = None
    domain: Optional[str] = None


@strawberry.type
class FactListType:
    facts: list[FactInfoType]
    total: int


@strawberry.type
class EntityReferenceInfoType:
    document: str
    section: Optional[str] = None
    mentions: int = 0
    mention_text: Optional[str] = None


@strawberry.type
class EntityInfoType:
    id: str
    name: str
    type: str
    types: list[str] = strawberry.field(default_factory=list)
    sources: list[str] = strawberry.field(default_factory=list)
    metadata: Optional[JSON] = None
    references: list[EntityReferenceInfoType] = strawberry.field(default_factory=list)
    mention_count: int = 0
    original_name: Optional[str] = None
    related_entities: Optional[JSON] = None


@strawberry.type
class EntityListType:
    entities: list[EntityInfoType]
    total: int


@strawberry.type
class DeleteResultType:
    status: str
    name: str


@strawberry.type
class ToggleStarResultType:
    name: str
    is_starred: bool


@strawberry.type
class FactMutationResultType:
    status: str
    fact: Optional[FactInfoType] = None


@strawberry.type
class MoveFactResultType:
    status: str
    fact_name: str
    to_domain: str


@strawberry.type
class AddEntityToGlossaryResultType:
    status: str
    entity_id: str
    note: Optional[str] = None
