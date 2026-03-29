# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL type definitions for the glossary domain."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class GlossaryParentType:
    name: str
    display_name: str


@strawberry.type
class GlossaryChildType:
    name: str
    display_name: str
    parent_verb: Optional[str] = None


@strawberry.type
class EntityRelationshipType:
    id: str
    subject: str
    verb: str
    object: str
    confidence: float
    user_edited: bool = False


@strawberry.type
class ConnectedResourceSource:
    document_name: str
    source: str
    section: Optional[str] = None
    url: Optional[str] = None


@strawberry.type
class ConnectedResourceType:
    entity_name: str
    entity_type: str
    sources: list[ConnectedResourceSource]


@strawberry.type
class GlossaryTermType:
    name: str
    display_name: str
    definition: Optional[str] = None
    domain: Optional[str] = None
    domain_path: Optional[str] = None
    parent_id: Optional[str] = None
    parent_verb: str = "HAS_KIND"
    aliases: list[str] = strawberry.field(default_factory=list)
    semantic_type: Optional[str] = None
    ner_type: Optional[str] = None
    cardinality: str = "many"
    status: Optional[str] = None
    provenance: Optional[str] = None
    glossary_status: str = "self_describing"
    entity_id: Optional[str] = None
    glossary_id: Optional[str] = None
    tags: Optional[JSON] = None
    ignored: bool = False
    canonical_source: Optional[str] = None
    connected_resources: list[ConnectedResourceType] = strawberry.field(default_factory=list)
    spanning_domains: Optional[list[str]] = None

    # Resolver fields — populated by detail resolver, not list
    parent: Optional[GlossaryParentType] = None
    children: Optional[list[GlossaryChildType]] = None
    relationships: Optional[list[EntityRelationshipType]] = None
    cluster_siblings: Optional[list[str]] = None


@strawberry.type
class GlossaryListType:
    terms: list[GlossaryTermType]
    total_defined: int
    total_self_describing: int
    clusters: Optional[JSON] = None


@strawberry.input
class GlossaryTermInput:
    name: str
    definition: str
    domain: Optional[str] = None
    aliases: list[str] = strawberry.field(default_factory=list)
    parent_id: Optional[str] = None
    semantic_type: Optional[str] = None
    is_abstract: Optional[bool] = None


@strawberry.input
class GlossaryTermUpdateInput:
    definition: Optional[str] = None
    aliases: Optional[list[str]] = None
    parent_id: Optional[str] = None
    parent_verb: Optional[str] = None
    status: Optional[str] = None
    domain: Optional[str] = None
    semantic_type: Optional[str] = None
    tags: Optional[JSON] = None
    ignored: Optional[bool] = None
    canonical_source: Optional[str] = None


@strawberry.enum
class GlossaryChangeAction(Enum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    GENERATION_STARTED = "generation_started"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETE = "generation_complete"


@strawberry.type
class GlossaryChangeEvent:
    session_id: str
    action: GlossaryChangeAction
    term_name: str
    term: Optional[GlossaryTermType] = None
    # Generation progress fields
    stage: Optional[str] = None
    percent: Optional[int] = None
    terms_count: Optional[int] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None


@strawberry.type
class GenerateResultType:
    status: str
    message: str


@strawberry.type
class DraftDefinitionType:
    name: str
    draft: str


@strawberry.type
class DraftAliasesType:
    name: str
    aliases: list[str]


@strawberry.type
class DraftTagsType:
    name: str
    tags: list[str]


@strawberry.type
class RefineResultType:
    name: str
    before: Optional[str]
    after: str


@strawberry.type
class TaxonomySuggestionType:
    child: str
    parent: str
    parent_verb: str
    confidence: str
    reason: str


@strawberry.type
class TaxonomySuggestionsType:
    suggestions: list[TaxonomySuggestionType]
    message: Optional[str] = None


@strawberry.type
class RenameResultType:
    old_name: str
    new_name: str
    display_name: str
    relationships_updated: int
