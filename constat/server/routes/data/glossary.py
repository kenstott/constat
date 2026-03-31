# Copyright (c) 2025 Kenneth Stott
# Canary: 83817908-7077-4127-b320-c65442447aff
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Glossary operation functions shared by GraphQL resolvers and REST routes."""

from __future__ import annotations

import json as _json
import logging
from typing import Any

from constat.discovery.models import display_entity_name

logger = logging.getLogger(__name__)


async def generate_glossary_op(
    session_id: str, managed: Any, sm: Any, phases: Any = None,
) -> dict[str, Any]:
    """Re-trigger LLM glossary generation in background thread."""
    session = managed.session
    vs = None
    if hasattr(session, "doc_tools") and session.doc_tools:
        vs = session.doc_tools._vector_store
    if not vs:
        raise ValueError("Vector store not available")

    managed._glossary_cancelled.set()

    import threading

    def _run():
        sm._run_glossary_generation(session_id, managed, phases=phases)

    thread = threading.Thread(target=_run, name=f"glossary-gen-{session_id}", daemon=True)
    thread.start()

    return {"status": "generating", "message": "Glossary generation started"}


async def rename_term_op(
    session_id: str, managed: Any, vs: Any, name: str, new_name: str,
) -> dict[str, Any]:
    """Rename a glossary term. Raises ValueError on validation failure."""
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("new_name is required")

    result = vs.rename_glossary_term(
        name, new_name, session_id, user_id=managed.user_id if managed else None,
    )
    return {"status": "renamed", **result}


async def draft_definition_op(
    session_id: str, managed: Any, vs: Any, name: str,
) -> dict[str, Any]:
    """AI-generate a draft definition for a glossary term."""
    session = managed.session
    if not session.router:
        raise ValueError("LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    system = (
        "Write a concise business glossary definition for the given term. "
        "Describe its meaning and purpose, not its storage. "
        "Return only the definition text, nothing else."
    )
    user_msg = f"Term: {display_entity_name(name)}\n\nContext:\n{context}"

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success or not result.content:
        raise ValueError("Draft generation failed")

    return {
        "status": "ok",
        "name": name,
        "draft": result.content.strip().strip('"'),
    }


async def draft_aliases_op(
    session_id: str, managed: Any, vs: Any, name: str,
) -> dict[str, Any]:
    """AI-generate draft aliases for a glossary term."""
    session = managed.session
    if not session.router:
        raise ValueError("LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    all_aliases: set[str] = set()
    all_term_names: set[str] = set()
    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    for t in terms:
        all_term_names.add(t.name.lower())
        if t.display_name:
            all_term_names.add(t.display_name.lower())
        for a in (t.aliases or []):
            all_aliases.add(a.lower())

    existing_list = ", ".join(sorted(all_aliases | all_term_names)) if (all_aliases or all_term_names) else "none"

    system = (
        "Generate 3-5 alternative names or aliases for the given glossary term. "
        "These should be synonyms, abbreviations, or alternative phrasings that users might use. "
        "Do NOT include any of the existing aliases or term names listed below. "
        "Return ONLY a JSON array of strings, nothing else."
    )
    user_msg = (
        f"Term: {display_entity_name(name)}\n\n"
        f"Context:\n{context}\n\n"
        f"Existing aliases and term names (DO NOT reuse any of these):\n{existing_list}"
    )

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success or not result.content:
        raise ValueError("Alias generation failed")

    content = result.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        aliases = _json.loads(content)
        if not isinstance(aliases, list):
            aliases = []
    except _json.JSONDecodeError:
        aliases = []

    forbidden = all_aliases | all_term_names
    aliases = [a for a in aliases if isinstance(a, str) and a.lower() not in forbidden]

    return {"status": "ok", "name": name, "aliases": aliases}


async def draft_tags_op(
    session_id: str, managed: Any, vs: Any, name: str,
) -> dict[str, Any]:
    """AI-generate classification tags for a glossary term."""
    session = managed.session
    if not session.router:
        raise ValueError("LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    term = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
    existing_tags = list((term.tags or {}).keys()) if term else []
    existing_str = ", ".join(existing_tags) if existing_tags else "none"

    system = (
        "Suggest classification tags for the given glossary term. "
        "Tags are short uppercase labels for regulatory, sensitivity, or data governance classification. "
        "Common examples: PII (personally identifiable information), PHI (protected health info), "
        "FINANCIAL (monetary/accounting data), SENSITIVE (confidential business data), "
        "GDPR (EU data protection), SOX (Sarbanes-Oxley), HIPAA (health data regulation), "
        "CCPA (California privacy), PUBLIC (non-sensitive), INTERNAL (internal use only), "
        "RESTRICTED (limited access), RETAIN_7Y (7-year retention). "
        "Only suggest tags that clearly apply. Do NOT repeat existing tags. "
        "Return ONLY a JSON array of uppercase strings, nothing else."
    )
    user_msg = (
        f"Term: {display_entity_name(name)}\n\n"
        f"Context:\n{context}\n\n"
        f"Existing tags (DO NOT repeat):\n{existing_str}"
    )

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success or not result.content:
        raise ValueError("Tag generation failed")

    content = result.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        tags = _json.loads(content)
        if not isinstance(tags, list):
            tags = []
    except _json.JSONDecodeError:
        tags = []

    existing_set = {t.upper() for t in existing_tags}
    tags = [t.upper() for t in tags if isinstance(t, str) and t.upper() not in existing_set]

    return {"status": "ok", "name": name, "tags": tags}


async def refine_definition_op(
    session_id: str, managed: Any, vs: Any, name: str,
) -> dict[str, Any]:
    """AI-refine an existing glossary definition."""
    existing = vs.get_glossary_term(name, session_id, user_id=managed.user_id)
    if not existing:
        raise ValueError(f"Term '{name}' not found")

    session = managed.session
    if not session.router:
        raise ValueError("LLM router not available")

    from constat.discovery.glossary_generator import _build_entity_context
    context = _build_entity_context(name, session_id, vs)

    system = (
        "Improve this business glossary definition. Keep it concise. "
        "Describe meaning, not storage. Return only the improved definition text."
    )
    user_msg = (
        f"Term: {existing.display_name}\n"
        f"Current definition: {existing.definition}\n\n"
        f"Context:\n{context}"
    )

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="low",
    )

    if not result.success:
        raise ValueError("Refinement failed")

    refined = result.content.strip().strip('"')
    before = existing.definition

    vs.update_glossary_term(name, session_id, {
        "definition": refined,
        "provenance": "hybrid" if existing.provenance == "llm" else existing.provenance,
    }, user_id=managed.user_id)

    return {"status": "refined", "name": name, "before": before, "after": refined}


async def suggest_taxonomy_op(
    session_id: str, managed: Any, vs: Any,
) -> dict[str, Any]:
    """LLM-suggest taxonomy relationships between glossary terms."""
    session = managed.session
    if not session.router:
        raise ValueError("LLM router not available")

    terms = vs.list_glossary_terms(session_id, user_id=managed.user_id)
    if len(terms) < 2:
        return {"suggestions": [], "message": "Need at least 2 defined terms"}

    term_descriptions = "\n".join(
        f"- {t.display_name}: {t.definition}" for t in terms if t.definition
    )

    system = (
        "You are organizing business glossary terms into a hierarchy. "
        "Suggest parent-child relationships between terms. "
        "Each relationship must be one of two types (from the parent's perspective):\n"
        "- HAS_ONE (composition): parent is composed of child. Example: Order HAS_ONE Line Item\n"
        "- HAS_KIND (taxonomy): child is a kind of parent. Example: Account HAS_KIND Savings Account\n"
        "- HAS_MANY (collection): parent contains many of child. Example: Team HAS_MANY Employee\n"
        "Only suggest relationships where a clear HAS_ONE, HAS_KIND, or HAS_MANY relationship exists. "
        'Respond as a JSON array: [{"child": "...", "parent": "...", "parent_verb": "HAS_ONE" or "HAS_KIND" or "HAS_MANY", "confidence": "high|medium|low", "reason": "..."}]'
    )
    user_msg = f"Terms:\n{term_descriptions}"

    from constat.core.models import TaskType
    result = session.router.execute(
        task_type=TaskType.GLOSSARY_GENERATION,
        system=system,
        user_message=user_msg,
        max_tokens=session.router.max_output_tokens,
        complexity="medium",
    )

    if not result.success:
        raise ValueError("Taxonomy suggestion failed")

    from constat.discovery.glossary_generator import _parse_llm_response
    parsed = _parse_llm_response(result.content)

    suggestions = []
    for item in parsed:
        child = item.get("child", "")
        parent = item.get("parent", "")
        if child and parent:
            verb = item.get("parent_verb", "HAS_KIND")
            if verb not in ("HAS_ONE", "HAS_KIND", "HAS_MANY"):
                verb = "HAS_KIND"
            suggestions.append({
                "child": child,
                "parent": parent,
                "parent_verb": verb,
                "confidence": item.get("confidence", "medium"),
                "reason": item.get("reason", ""),
            })

    return {"suggestions": suggestions}
