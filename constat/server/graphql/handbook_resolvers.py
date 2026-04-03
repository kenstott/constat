# Copyright (c) 2025 Kenneth Stott
# Canary: b2c3d4e5-f6a7-8901-bcde-f23456789012
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for the Domain Handbook feature."""

from __future__ import annotations

import json
import logging
from typing import Optional

import strawberry
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info

logger = logging.getLogger(__name__)


@strawberry.type
class HandbookEntryType:
    key: str
    display: str
    metadata: Optional[JSON] = None
    editable: bool = False


@strawberry.type
class HandbookSectionType:
    title: str
    content: list[HandbookEntryType]
    last_updated: Optional[str] = None


@strawberry.type
class DomainHandbookType:
    domain: str
    generated_at: str
    summary: str = ""
    sections: JSON = None  # dict[str, HandbookSection] serialized


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


def _get_managed_session(info: Info, session_id: str, user_id: str):
    """Get a managed session, validate ownership."""
    managed = info.context.session_manager.get_session_or_none(session_id)
    if not managed or managed.user_id != user_id:
        raise ValueError("Session not found")
    return managed


def _section_to_type(section) -> HandbookSectionType:
    """Convert a HandbookSection dataclass to the GraphQL type."""
    entries = [
        HandbookEntryType(
            key=e.key,
            display=e.display,
            metadata=e.metadata if e.metadata else None,
            editable=e.editable,
        )
        for e in section.content
    ]
    return HandbookSectionType(
        title=section.title,
        content=entries,
        last_updated=section.last_updated or None,
    )


def _handbook_to_type(handbook) -> DomainHandbookType:
    """Convert a DomainHandbook dataclass to the GraphQL type."""
    sections_json = {}
    for key, section in handbook.sections.items():
        sections_json[key] = {
            "title": section.title,
            "last_updated": section.last_updated,
            "content": [
                {
                    "key": e.key,
                    "display": e.display,
                    "metadata": e.metadata,
                    "editable": e.editable,
                }
                for e in section.content
            ],
        }
    return DomainHandbookType(
        domain=handbook.domain,
        generated_at=handbook.generated_at,
        summary=handbook.summary,
        sections=sections_json,
    )


@strawberry.type
class HandbookUpdateResultType:
    status: str
    section: str
    key: str


@strawberry.type
class Query:
    @strawberry.field
    async def handbook(
        self,
        info: Info,
        session_id: str,
        domain: Optional[str] = None,
    ) -> DomainHandbookType:
        """Generate a complete domain handbook."""
        user_id = _require_auth(info)
        managed = _get_managed_session(info, session_id, user_id)
        handbook = managed.session.generate_handbook(domain=domain)
        return _handbook_to_type(handbook)

    @strawberry.field
    async def handbook_section(
        self,
        info: Info,
        session_id: str,
        section: str,
        domain: Optional[str] = None,
    ) -> HandbookSectionType:
        """Get a single handbook section."""
        user_id = _require_auth(info)
        managed = _get_managed_session(info, session_id, user_id)
        handbook = managed.session.generate_handbook(domain=domain)
        if section not in handbook.sections:
            raise ValueError(f"Section '{section}' not found in handbook")
        return _section_to_type(handbook.sections[section])


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def update_handbook_entry(
        self,
        info: Info,
        session_id: str,
        section: str,
        key: str,
        field_name: str,
        new_value: str,
        reason: Optional[str] = None,
    ) -> HandbookUpdateResultType:
        """Edit a handbook entry -- routes to underlying store."""
        user_id = _require_auth(info)
        managed = _get_managed_session(info, session_id, user_id)
        managed.session.update_handbook_entry(
            section=section,
            key=key,
            field_name=field_name,
            new_value=new_value,
            reason=reason,
        )
        return HandbookUpdateResultType(
            status="ok",
            section=section,
            key=key,
        )
