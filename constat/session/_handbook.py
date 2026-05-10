# Copyright (c) 2025 Kenneth Stott
# Canary: a1b2c3d4-e5f6-7890-abcd-ef1234567890
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Handbook generation mixin for Session.

Generates a readable, auto-assembled handbook per domain from existing stores.
No new data collection — just assembles from config, relational store, learning
store, agent/skill managers, and session history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HandbookEntry:
    key: str
    display: str
    metadata: dict = field(default_factory=dict)
    editable: bool = False


@dataclass
class HandbookSection:
    title: str
    content: list[HandbookEntry] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class DomainHandbook:
    domain: str
    generated_at: str
    summary: str = ""
    sections: dict[str, HandbookSection] = field(default_factory=dict)


# noinspection PyUnresolvedReferences
class HandbookMixin:
    """Mixin for Session to generate domain handbooks.

    Expects the following attributes from the Session class hierarchy:
    - config, resolved_config, session_databases, session_files
    - doc_tools, schema_manager, api_schema_manager
    - learning_store, agent_manager, skill_manager
    - history, session_id
    """

    def generate_handbook(self, domain: str | None = None) -> DomainHandbook:
        """Generate a complete domain handbook.

        Args:
            domain: Domain filename to generate handbook for.
                    If None, uses the first active domain or "default".

        Returns:
            Assembled DomainHandbook with all sections.
        """
        domain = domain or self._resolve_handbook_domain()
        now = datetime.now(timezone.utc).isoformat()

        sections: dict[str, HandbookSection] = {}
        builders = [
            ("overview", self._build_overview_section),
            ("sources", self._build_sources_section),
            ("entities", self._build_entities_section),
            ("glossary", self._build_glossary_section),
            ("rules", self._build_rules_section),
            ("patterns", self._build_patterns_section),
            ("agents_skills", self._build_agents_skills_section),
            ("limitations", self._build_limitations_section),
        ]

        for key, builder in builders:
            sections[key] = builder(domain)

        # Build summary from section titles and entry counts (no LLM in v1)
        non_empty = [
            f"{s.title} ({len(s.content)})"
            for s in sections.values()
            if s.content
        ]
        summary = f"Domain handbook with {len(non_empty)} sections: {', '.join(non_empty)}." if non_empty else ""

        return DomainHandbook(
            domain=domain,
            generated_at=now,
            summary=summary,
            sections=sections,
        )

    def _resolve_handbook_domain(self) -> str:
        """Resolve the domain to use for handbook generation."""
        # Check for active domains on the managed session (set by session_manager)
        if hasattr(self, "active_domains") and self.active_domains:
            return self.active_domains[0]
        return "default"

    def _build_overview_section(self, domain: str) -> HandbookSection:
        """Domain name, description, parent domains from config."""
        entries: list[HandbookEntry] = []

        # Get domain info from config
        domain_config = self._get_domain_config(domain)
        if domain_config:
            entries.append(HandbookEntry(
                key="name",
                display=f"Domain: {domain_config.get('name', domain)}",
                metadata={"name": domain_config.get("name", domain)},
            ))
            if domain_config.get("description"):
                entries.append(HandbookEntry(
                    key="description",
                    display=domain_config["description"],
                    metadata={"description": domain_config["description"]},
                    editable=True,
                ))
            if domain_config.get("domains"):
                parent_names = domain_config["domains"]
                entries.append(HandbookEntry(
                    key="parent_domains",
                    display=f"Parent domains: {', '.join(parent_names)}",
                    metadata={"parents": parent_names},
                ))

        return HandbookSection(title="Overview", content=entries)

    def _build_sources_section(self, domain: str) -> HandbookSection:
        """Data sources with schema summaries."""
        entries: list[HandbookEntry] = []

        # Config databases
        if hasattr(self, "config") and self.config.databases:
            for db_cfg in self.config.databases:
                meta: dict[str, Any] = {
                    "type": db_cfg.type,
                    "source": "config",
                }
                # Get table count from schema manager
                if hasattr(self, "schema_manager"):
                    tables = self.schema_manager.get_tables(db_cfg.name)
                    meta["table_count"] = len(tables)
                entries.append(HandbookEntry(
                    key=f"db:{db_cfg.name}",
                    display=f"Database: {db_cfg.name} ({db_cfg.type})",
                    metadata=meta,
                ))

        # Session databases
        for name, db_info in getattr(self, "session_databases", {}).items():
            entries.append(HandbookEntry(
                key=f"session_db:{name}",
                display=f"Session DB: {name} ({db_info.get('type', 'unknown')})",
                metadata={
                    "type": db_info.get("type", ""),
                    "description": db_info.get("description", ""),
                    "source": "session",
                },
            ))

        # Config APIs
        if hasattr(self, "config") and self.config.apis:
            for api_cfg in self.config.apis:
                entries.append(HandbookEntry(
                    key=f"api:{api_cfg.name}",
                    display=f"API: {api_cfg.name}",
                    metadata={"type": "api", "source": "config"},
                ))

        # Session files / documents
        for name, file_info in getattr(self, "session_files", {}).items():
            entries.append(HandbookEntry(
                key=f"file:{name}",
                display=f"File: {name}",
                metadata={
                    "uri": file_info.get("uri", ""),
                    "source": "session",
                },
            ))

        return HandbookSection(title="Data Sources", content=entries)

    def _build_entities_section(self, domain: str) -> HandbookSection:
        """Key entities with relationships from RelationalStore."""
        entries: list[HandbookEntry] = []

        relational = self._get_relational_store()
        if relational is None:
            return HandbookSection(title="Key Entities", content=entries)

        vis_filter, vis_params = relational.entity_visibility_filter(
            self.session_id,
            active_domains=[domain] if domain != "default" else None,
            alias="e",
            cross_session=True,
        )
        rows = relational.list_entities_with_refcount(vis_filter, vis_params)

        for row in rows:
            entity_id, name, display_name, semantic_type, ner_type, ref_count = row
            entries.append(HandbookEntry(
                key=f"entity:{entity_id}",
                display=f"{display_name or name} ({semantic_type or ner_type or 'unknown'})",
                metadata={
                    "id": entity_id,
                    "name": name,
                    "display_name": display_name,
                    "semantic_type": semantic_type,
                    "ner_type": ner_type,
                    "ref_count": ref_count,
                },
                editable=True,
            ))

        return HandbookSection(title="Key Entities", content=entries)

    def _build_glossary_section(self, domain: str) -> HandbookSection:
        """Glossary terms with grounding status from RelationalStore."""
        entries: list[HandbookEntry] = []

        relational = self._get_relational_store()
        if relational is None:
            return HandbookSection(title="Glossary", content=entries)

        terms = relational.list_glossary_terms(
            session_id=self.session_id,
            domain=domain if domain != "default" else None,
            user_id=getattr(self, "user_id", None),
        )

        for term in terms:
            definition = term.definition or ""
            aliases = term.aliases or []
            entries.append(HandbookEntry(
                key=f"glossary:{term.id}",
                display=f"{term.display_name or term.name}: {definition}",
                metadata={
                    "id": term.id,
                    "name": term.name,
                    "definition": definition,
                    "aliases": aliases,
                    "semantic_type": term.semantic_type or "",
                    "domain": term.domain or "",
                    "status": term.status or "draft",
                    "provenance": term.provenance or "llm",
                },
                editable=True,
            ))

        return HandbookSection(title="Glossary", content=entries)

    def _build_rules_section(self, domain: str) -> HandbookSection:
        """Learned rules with confidence from LearningStore."""
        entries: list[HandbookEntry] = []

        if not hasattr(self, "learning_store"):
            return HandbookSection(title="Learned Rules", content=entries)

        rules = self.learning_store.list_rules(
            domain=domain if domain != "default" else None,
        )

        for rule in rules:
            confidence = rule.get("confidence", 0.0)
            entries.append(HandbookEntry(
                key=f"rule:{rule['id']}",
                display=f"[{confidence:.0%}] {rule['summary']}",
                metadata={
                    "id": rule["id"],
                    "category": rule.get("category", ""),
                    "confidence": confidence,
                    "applied_count": rule.get("applied_count", 0),
                    "tags": rule.get("tags", []),
                    "source_count": len(rule.get("source_learnings", [])),
                },
                editable=True,
            ))

        return HandbookSection(title="Learned Rules", content=entries)

    def _build_patterns_section(self, domain: str) -> HandbookSection:
        """Common query patterns from session history."""
        entries: list[HandbookEntry] = []

        if not hasattr(self, "history"):
            return HandbookSection(title="Common Patterns", content=entries)

        sessions = self.history.list_sessions(limit=20)
        query_counts: dict[str, int] = {}
        for sess in sessions:
            queries = sess.get("queries", [])
            for q in queries:
                text = q if isinstance(q, str) else q.get("text", "")
                if text:
                    # Normalize to lower case for grouping
                    normalized = text.strip().lower()
                    query_counts[normalized] = query_counts.get(normalized, 0) + 1

        # Sort by frequency, take top 10
        sorted_patterns = sorted(query_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for pattern_text, count in sorted_patterns:
            entries.append(HandbookEntry(
                key=f"pattern:{hash(pattern_text) & 0xFFFFFFFF:08x}",
                display=f"({count}x) {pattern_text}",
                metadata={"query": pattern_text, "count": count},
            ))

        return HandbookSection(title="Common Patterns", content=entries)

    def _build_agents_skills_section(self, domain: str) -> HandbookSection:
        """Available agents and skills."""
        entries: list[HandbookEntry] = []

        # Agents
        if hasattr(self, "agent_manager"):
            agent_names = self.agent_manager.list_agents(
                domain=domain if domain != "default" else None,
            )
            for name in agent_names:
                agent = self.agent_manager.get_agent(name)
                if agent:
                    entries.append(HandbookEntry(
                        key=f"agent:{name}",
                        display=f"Agent: {name} - {agent.description}",
                        metadata={
                            "name": name,
                            "description": agent.description,
                            "domain": agent.domain,
                            "type": "agent",
                        },
                    ))

        # Skills
        if hasattr(self, "skill_manager"):
            skill_names = self.skill_manager.list_skills(
                domain=domain if domain != "default" else None,
            )
            for name in skill_names:
                skill = self.skill_manager.get_skill(name)
                if skill:
                    entries.append(HandbookEntry(
                        key=f"skill:{name}",
                        display=f"Skill: {name} - {skill.description}",
                        metadata={
                            "name": name,
                            "description": skill.description,
                            "type": "skill",
                        },
                    ))

        return HandbookSection(title="Agents & Skills", content=entries)

    def _build_limitations_section(self, domain: str) -> HandbookSection:
        """Known limitations: ungrounded terms, stale sources."""
        entries: list[HandbookEntry] = []

        # Check for ungrounded glossary terms
        relational = self._get_relational_store()
        if relational is not None:
            terms = relational.list_glossary_terms(
                session_id=self.session_id,
                domain=domain if domain != "default" else None,
                user_id=getattr(self, "user_id", None),
            )
            draft_terms = [t for t in terms if t.status == "draft"]
            if draft_terms:
                names = [t.display_name or t.name for t in draft_terms]
                entries.append(HandbookEntry(
                    key="draft_terms",
                    display=f"{len(draft_terms)} draft glossary terms: {', '.join(names[:10])}",
                    metadata={
                        "count": len(draft_terms),
                        "terms": [t.name for t in draft_terms],
                    },
                ))

        return HandbookSection(title="Known Limitations", content=entries)

    def update_handbook_entry(
        self,
        section: str,
        key: str,
        field_name: str,
        new_value: str,
        reason: str | None = None,
    ) -> bool:
        """Apply a handbook edit to the underlying store.

        Routes edits to the appropriate store based on section:
        - glossary -> RelationalStore glossary update
        - rules -> LearningStore rule update
        - entities -> RelationalStore entity update

        Args:
            section: Section name (glossary, rules, entities, etc.)
            key: Entry key (e.g., "glossary:abc123")
            field_name: Field to update (e.g., "definition", "summary")
            new_value: New value for the field
            reason: Optional reason for the edit

        Returns:
            True if the edit was applied successfully.
        """
        if section == "glossary":
            return self._update_glossary_entry(key, field_name, new_value)
        elif section == "rules":
            return self._update_rule_entry(key, field_name, new_value)
        elif section == "entities":
            return self._update_entity_entry(key, field_name, new_value)
        raise ValueError(f"Section '{section}' does not support edits")

    def _update_glossary_entry(self, key: str, field_name: str, new_value: str) -> bool:
        """Update a glossary term via RelationalStore."""
        relational = self._get_relational_store()
        if relational is None:
            raise ValueError("RelationalStore not available")

        term_id = key.removeprefix("glossary:")
        term = relational.get_glossary_term_by_id(term_id)
        if term is None:
            raise ValueError(f"Glossary term '{term_id}' not found")

        updates: dict[str, Any] = {}
        if field_name == "definition":
            updates["definition"] = new_value
        elif field_name == "name":
            updates["name"] = new_value
            updates["display_name"] = new_value.title()
        elif field_name == "aliases":
            # Expect comma-separated aliases
            updates["aliases"] = [a.strip() for a in new_value.split(",") if a.strip()]
        else:
            raise ValueError(f"Cannot edit field '{field_name}' on glossary terms")

        return relational.update_glossary_term(
            name=term.name,
            session_id=term.session_id,
            updates=updates,
            user_id=getattr(self, "user_id", None),
        )

    def _update_rule_entry(self, key: str, field_name: str, new_value: str) -> bool:
        """Update a learning rule via LearningStore."""
        if not hasattr(self, "learning_store"):
            raise ValueError("LearningStore not available")

        rule_id = key.removeprefix("rule:")
        if field_name == "summary":
            return self.learning_store.update_rule(rule_id, summary=new_value)
        elif field_name == "tags":
            tags = [t.strip() for t in new_value.split(",") if t.strip()]
            return self.learning_store.update_rule(rule_id, tags=tags)
        elif field_name == "confidence":
            return self.learning_store.update_rule(rule_id, confidence=float(new_value))
        raise ValueError(f"Cannot edit field '{field_name}' on rules")

    def _update_entity_entry(self, key: str, field_name: str, new_value: str) -> bool:
        """Update an entity via RelationalStore."""
        relational = self._get_relational_store()
        if relational is None:
            raise ValueError("RelationalStore not available")

        entity_id = key.removeprefix("entity:")
        if field_name == "name":
            relational.update_entity_name(entity_id, new_value, new_value.title())
            return True
        raise ValueError(f"Cannot edit field '{field_name}' on entities")

    def _get_relational_store(self):
        """Get the RelationalStore from doc_tools._vector_store."""
        if hasattr(self, "doc_tools") and self.doc_tools:
            vs = getattr(self.doc_tools, "_vector_store", None)
            if vs and hasattr(vs, "_relational"):
                return vs._relational
        return None

    def _get_domain_config(self, domain: str) -> dict | None:
        """Get domain configuration dict."""
        if not hasattr(self, "config") or not self.config:
            return None

        # Check config.domains (list of domain YAML configs)
        domains = getattr(self.config, "domains", None)
        if domains:
            for d in domains:
                filename = getattr(d, "filename", None) or getattr(d, "name", "")
                if filename == domain or getattr(d, "name", "") == domain:
                    return {
                        "name": getattr(d, "name", domain),
                        "description": getattr(d, "description", ""),
                        "domains": getattr(d, "domains", []),
                    }
        return None
