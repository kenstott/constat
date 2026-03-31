# Copyright (c) 2025 Kenneth Stott
# Canary: 546515f6-d108-4e05-9058-9c938f35e359
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for domain management (Phase 6)."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class DomainInfoType:
    filename: str
    name: str
    description: str = ""
    path: str = ""
    tier: str = "system"
    active: bool = True
    owner: str = ""
    steward: str = ""


@strawberry.type
class DomainTreeNodeType:
    filename: str
    name: str
    path: str = ""
    description: str = ""
    tier: str = "system"
    active: bool = True
    steward: str = ""
    owner: str = ""
    databases: list[str] = strawberry.field(default_factory=list)
    apis: list[str] = strawberry.field(default_factory=list)
    documents: list[str] = strawberry.field(default_factory=list)
    skills: list[str] = strawberry.field(default_factory=list)
    agents: list[str] = strawberry.field(default_factory=list)
    rules: list[str] = strawberry.field(default_factory=list)
    facts: list[str] = strawberry.field(default_factory=list)
    system_prompt: str = ""
    domains: list[str] = strawberry.field(default_factory=list)
    children: list[DomainTreeNodeType] = strawberry.field(default_factory=list)


@strawberry.type
class DomainDetailType:
    filename: str
    name: str
    description: str = ""
    tier: str = "system"
    active: bool = True
    owner: str = ""
    steward: str = ""
    databases: list[str] = strawberry.field(default_factory=list)
    apis: list[str] = strawberry.field(default_factory=list)
    documents: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class DomainListType:
    domains: list[DomainInfoType]


@strawberry.type
class DomainContentType:
    content: str
    path: str
    filename: str


@strawberry.type
class DomainSkillType:
    name: str
    description: str = ""
    domain: str = ""


@strawberry.type
class DomainAgentType:
    name: str
    description: str = ""
    domain: str = ""


@strawberry.type
class DomainRuleType:
    id: str
    summary: str
    category: str
    confidence: float = 0.0
    source_count: int = 0
    tags: list[str] = strawberry.field(default_factory=list)
    domain: str = ""
    source: str = ""
    scope: Optional[JSON] = None


@strawberry.type
class DomainFactType:
    name: str
    value: Optional[JSON] = None
    domain: str = ""
    source: Optional[str] = None
    confidence: Optional[float] = None


# Domain inputs

@strawberry.input
class CreateDomainInput:
    name: str
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    parent_domain: Optional[str] = None
    initial_domains: Optional[list[str]] = None


@strawberry.input
class UpdateDomainInput:
    name: Optional[str] = None
    description: Optional[str] = None
    order: Optional[int] = None
    active: Optional[bool] = None


@strawberry.input
class MoveDomainSourceInput:
    source_type: str
    source_name: str
    from_domain: str
    to_domain: str
    session_id: Optional[str] = None


@strawberry.input
class MoveDomainSkillInput:
    skill_name: str
    from_domain: str
    to_domain: str
    validate_only: Optional[bool] = None


@strawberry.input
class MoveDomainAgentInput:
    agent_name: str
    from_domain: str
    to_domain: str


@strawberry.input
class MoveDomainRuleInput:
    rule_id: str
    to_domain: str


# Domain result types

@strawberry.type
class CreateDomainResultType:
    status: str
    filename: str
    name: str
    description: str = ""


@strawberry.type
class UpdateDomainResultType:
    status: str
    filename: str


@strawberry.type
class DeleteDomainResultType:
    status: str
    filename: str


@strawberry.type
class DomainContentSaveResultType:
    status: str
    filename: str
    path: str


@strawberry.type
class PromoteDomainResultType:
    status: str
    filename: str
    new_tier: str


@strawberry.type
class MoveDomainSkillResultType:
    status: str
    skill: Optional[str] = None
    to_domain: Optional[str] = None
    warnings: Optional[list[str]] = None


@strawberry.type
class MoveDomainAgentResultType:
    status: str
    agent: Optional[str] = None
    to_domain: Optional[str] = None


@strawberry.type
class MoveDomainRuleResultType:
    status: str
    rule_id: Optional[str] = None
    to_domain: Optional[str] = None


@strawberry.type
class MoveDomainSourceResultType:
    status: str
