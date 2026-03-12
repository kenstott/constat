# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Golden question test runner — Phase 1: DB lookups, Phase 2: real pipeline + LLM judge."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from constat.core.config import Config
from constat.discovery.models import display_entity_name as _dn
from constat.testing.models import (
    DomainTestResult,
    EndToEndAssertion,
    EndToEndResult,
    GoldenQuestion,
    GlossaryAssertion,
    GroundingAssertion,
    LayerResult,
    QuestionResult,
    RelationshipAssertion,
    parse_golden_questions,
)

logger = logging.getLogger(__name__)


def _open_doc_tools(config: Config):
    """Open the DuckDB vector store read-only and return (doc_tools, relational).

    Caller must keep the doc_tools reference alive to prevent GC closing the connection.
    """
    from constat.discovery.doc_tools import DocumentDiscoveryTools
    doc_tools = DocumentDiscoveryTools(config, skip_auto_index=True)
    return doc_tools, doc_tools._vector_store._relational


def run_domain_test(
    config: Config,
    domain_filename: str,
    tags: list[str] | None = None,
    session_id: str = "golden-test",
    user_id: str = "default",
) -> DomainTestResult:
    """Run golden question assertions for a single domain.

    Opens the existing DuckDB vector store read-only and checks entities,
    grounding, glossary, and relationships. All checks are pure DB lookups.
    """
    domain_config = config.load_domain(domain_filename)
    if not domain_config:
        raise ValueError(f"Domain not found: {domain_filename}")

    questions = parse_golden_questions(domain_config.golden_questions)
    if tags:
        tag_set = set(tags)
        questions = [q for q in questions if tag_set & set(q.tags)]

    if not questions:
        return DomainTestResult(domain=domain_filename, questions=[])

    _dt, relational = _open_doc_tools(config)

    results = []
    for q in questions:
        qr = _run_question(relational, q, session_id, user_id, domain_filename)
        results.append(qr)

    return DomainTestResult(domain=domain_filename, questions=results)


def run_domain_test_e2e(
    config: Config,
    domain_filename: str,
    tags: list[str] | None = None,
    session_id: str = "golden-test",
    user_id: str = "default",
) -> DomainTestResult:
    """Run Phase 1 metadata checks, then Phase 2 end-to-end for passing questions."""
    # Phase 1
    result = run_domain_test(config, domain_filename, tags, session_id, user_id)

    domain_config = config.load_domain(domain_filename)
    questions = parse_golden_questions(domain_config.golden_questions)
    if tags:
        tag_set = set(tags)
        questions = [q for q in questions if tag_set & set(q.tags)]

    # Build lookup: question text -> GoldenQuestion
    q_lookup = {q.question: q for q in questions}

    for qr in result.questions:
        gq = q_lookup.get(qr.question)
        if not gq or not gq.expect.end_to_end:
            continue

        # Fail-fast: skip Phase 2 if Phase 1 metadata failed
        if not all(lr.passed == lr.total for lr in qr.layers):
            qr.end_to_end = EndToEndResult(
                passed=False,
                failures=["Skipped: Phase 1 metadata assertions failed"],
            )
            continue

        qr.end_to_end = _run_e2e_question(config, gq, session_id, user_id)

    return result


# ---------------------------------------------------------------------------
# Streaming generators — yield progress events per question
# ---------------------------------------------------------------------------

@dataclass
class TestProgressEvent:
    """Progress event yielded during streaming test execution."""
    event: str  # domain_start, question_start, question_done, domain_done
    domain: str
    domain_name: str = ""
    question: str = ""
    question_index: int = 0
    question_total: int = 0
    phase: str = ""  # "metadata" or "e2e"
    result: QuestionResult | None = None


def iter_domain_test(
    config: Config,
    domain_filename: str,
    tags: list[str] | None = None,
    session_id: str = "golden-test",
    user_id: str = "default",
    include_e2e: bool = False,
):
    """Generator that yields TestProgressEvent per question."""
    domain_config = config.load_domain(domain_filename)
    if not domain_config:
        raise ValueError(f"Domain not found: {domain_filename}")

    domain_name = domain_config.name or domain_filename
    questions = parse_golden_questions(domain_config.golden_questions)
    if tags:
        tag_set = set(tags)
        questions = [q for q in questions if tag_set & set(q.tags)]

    total = len(questions)
    if total == 0:
        yield TestProgressEvent(event="domain_done", domain=domain_filename, domain_name=domain_name)
        return

    _dt, relational = _open_doc_tools(config)

    yield TestProgressEvent(
        event="domain_start", domain=domain_filename, domain_name=domain_name,
        question_total=total,
    )

    results: list[QuestionResult] = []
    for i, q in enumerate(questions):
        yield TestProgressEvent(
            event="question_start", domain=domain_filename, domain_name=domain_name,
            question=q.question, question_index=i, question_total=total,
            phase="metadata",
        )

        qr = _run_question(relational, q, session_id, user_id, domain_filename)

        # Phase 2 e2e if requested
        if include_e2e and q.expect.end_to_end:
            if not all(lr.passed == lr.total for lr in qr.layers):
                qr.end_to_end = EndToEndResult(
                    passed=False,
                    failures=["Skipped: Phase 1 metadata assertions failed"],
                )
            else:
                yield TestProgressEvent(
                    event="question_start", domain=domain_filename, domain_name=domain_name,
                    question=q.question, question_index=i, question_total=total,
                    phase="e2e",
                )
                qr.end_to_end = _run_e2e_question(config, q, session_id, user_id)

        results.append(qr)

        yield TestProgressEvent(
            event="question_done", domain=domain_filename, domain_name=domain_name,
            question=q.question, question_index=i, question_total=total,
            result=qr,
        )

    yield TestProgressEvent(
        event="domain_done", domain=domain_filename, domain_name=domain_name,
        question_total=total,
    )


def _run_e2e_question(
    config: Config,
    gq: GoldenQuestion,
    session_id: str,
    user_id: str,
) -> EndToEndResult:
    """Run a single question through the real pipeline and evaluate."""
    from constat.api.factory import create_api

    assertion = gq.expect.end_to_end
    failures: list[str] = []
    t0 = time.monotonic()

    test_session_id = f"test-{uuid.uuid4().hex[:12]}"
    api = create_api(
        config,
        session_id=test_session_id,
        user_id=user_id,
        require_approval=False,
        auto_approve=True,
    )

    solve_result = api.solve(gq.question, require_approval=False, force_plan=True)
    duration = time.monotonic() - t0

    # Check expect_success
    if assertion.expect_success and not solve_result.success:
        failures.append(f"Expected success but got error: {solve_result.error}")
        return EndToEndResult(
            passed=False, answer=solve_result.answer,
            failures=failures, duration_s=duration,
        )
    if not assertion.expect_success and solve_result.success:
        failures.append("Expected failure but solve succeeded")
        return EndToEndResult(
            passed=False, answer=solve_result.answer,
            failures=failures, duration_s=duration,
        )

    answer = solve_result.answer or ""

    # Check result_contains
    for substring in assertion.result_contains:
        if substring.lower() not in answer.lower():
            failures.append(f'Answer missing expected substring: "{substring}"')

    # Check plan_min_steps
    step_count = len(solve_result.steps)
    if step_count < assertion.plan_min_steps:
        failures.append(
            f"Expected at least {assertion.plan_min_steps} steps, got {step_count}"
        )

    # LLM judge for semantic_match
    judge_reasoning = None
    if assertion.semantic_match and answer:
        passed, reasoning = _llm_judge(
            gq.question, answer, assertion.semantic_match, config,
        )
        judge_reasoning = reasoning
        if not passed:
            failures.append(f"LLM judge failed: {reasoning}")

    return EndToEndResult(
        passed=len(failures) == 0,
        answer=answer[:2000] if answer else None,
        judge_reasoning=judge_reasoning,
        failures=failures,
        duration_s=duration,
    )


def _llm_judge(
    question: str, answer: str, criteria: str, config: Config,
) -> tuple[bool, str]:
    """Use the config's LLM to evaluate if answer meets criteria.

    Returns (passed, reasoning).
    """
    from constat.providers.router import TaskRouter

    router = TaskRouter(config.llm)
    system = (
        "You are a test evaluator. Given a question and an answer, determine if "
        "the answer satisfies the given criteria. Reply with exactly YES or NO on "
        "the first line, then one sentence explaining why."
    )
    user_message = (
        f"Question: {question}\n\n"
        f"Answer: {answer[:3000]}\n\n"
        f"Criteria: {criteria}\n\n"
        f"Does the answer satisfy the criteria?"
    )
    response = router.generate(system=system, user_message=user_message, max_tokens=256)
    first_line = response.strip().split("\n")[0].strip().upper()
    passed = first_line.startswith("YES")
    return passed, response.strip()


def _run_question(
    relational,
    q: GoldenQuestion,
    session_id: str,
    user_id: str,
    domain_id: str,
) -> QuestionResult:
    layers: list[LayerResult] = []
    # Include the domain being tested so cross-domain entities are visible
    domain_ids = [domain_id]

    if q.expect.entities:
        layers.append(_check_entities(relational, q.expect.entities, session_id, domain_ids, user_id))

    if q.expect.grounding:
        layers.append(_check_grounding(relational, q.expect.grounding, session_id, domain_ids, user_id))

    if q.expect.glossary:
        layers.append(_check_glossary(relational, q.expect.glossary, session_id, user_id))

    if q.expect.relationships:
        layers.append(_check_relationships(relational, q.expect.relationships, session_id, user_id, domain_ids))

    return QuestionResult(question=q.question, tags=q.tags, layers=layers)


def _resolve_entity_name(relational, name: str, session_id: str, user_id: str, domain_ids: list[str]):
    """Find an entity by name, falling back to glossary alias resolution."""
    entity = relational.find_entity_by_name(
        name, domain_ids=domain_ids, session_id=session_id, cross_session=True,
    )
    if entity:
        return entity
    # Check if name is a glossary alias — resolve to the canonical entity
    term = relational.get_glossary_term_by_name_or_alias(name, session_id, user_id=user_id)
    if term:
        return relational.find_entity_by_name(
            term.name, domain_ids=domain_ids, session_id=session_id, cross_session=True,
        )
    return None


def _check_entities(
    relational, expected: list[str], session_id: str, domain_ids: list[str],
    user_id: str = "default",
) -> LayerResult:
    failures = []
    for name in expected:
        entity = _resolve_entity_name(relational, name, session_id, user_id, domain_ids)
        if not entity:
            # Entity not in NER store — check if proven grounding exists
            # (proof validated this entity against a real data source)
            if relational.get_proven_grounding(name):
                continue
            failures.append(f'missing "{_dn(name)}"')
    return LayerResult(
        layer="entities",
        passed=len(expected) - len(failures),
        total=len(expected),
        failures=failures,
    )


# Map FactSource type names to document-name prefixes for grounding checks.
# When a grounding assertion uses a source type (e.g. "database") instead of
# a specific document pattern (e.g. "schema:sales.customers"), we match against
# the corresponding document-name prefix in the embeddings table.
_SOURCE_TYPE_PREFIXES: dict[str, list[str]] = {
    "database": ["schema:"],
    "document": ["Document:", "doc:"],
    "api": ["api:", "API:"],
    "embedded": ["schema:", "api:", "Document:", "doc:"],  # any indexed source
    "cache": [],
    "derived": [],
    "llm_knowledge": [],
}


def _matches_grounding(expected: str, doc_names: list[str]) -> bool:
    """Check if an expected grounding pattern matches any document name.

    Handles both specific patterns (substring match) and source-type
    shorthand like "database" or "document" (prefix match).
    """
    prefixes = _SOURCE_TYPE_PREFIXES.get(expected)
    if prefixes is not None:
        # Source-type shorthand — match if ANY doc has a matching prefix
        if not prefixes:
            # Ungroundable source types (cache, derived, llm) — pass trivially
            return True
        return any(
            any(doc_name.startswith(p) for p in prefixes)
            for doc_name in doc_names
        )
    # Specific pattern — substring match (original behaviour)
    return any(expected in doc_name for doc_name in doc_names)


def _check_grounding(
    relational, assertions: list[GroundingAssertion], session_id: str,
    domain_ids: list[str], user_id: str = "default",
) -> LayerResult:
    failures = []
    for ga in assertions:
        entity = _resolve_entity_name(relational, ga.entity, session_id, user_id, domain_ids)

        # Try deterministic proven grounding first — use resolved name or raw assertion name
        proven_name = entity.name if entity else ga.entity
        proven_patterns = relational.get_proven_grounding(proven_name)
        if proven_patterns:
            if not ga.strict:
                continue  # proven patterns exist → grounded
            matched = any(
                _matches_grounding(expected, proven_patterns)
                for expected in ga.resolves_to
            )
            if not matched:
                failures.append(
                    f'"{_dn(ga.entity)}" not grounded to any of {ga.resolves_to} '
                    f"(proven: {proven_patterns})"
                )
            continue

        # Fall back to chunk-entity associations (non-deterministic)
        if not entity:
            failures.append(f'entity "{_dn(ga.entity)}" not found for grounding check')
            continue
        doc_names = relational.get_entity_document_names(entity.id)
        if not ga.strict:
            if not doc_names:
                failures.append(f'"{_dn(ga.entity)}" has no grounding (expected at least one data source)')
            continue
        matched = any(
            _matches_grounding(expected, doc_names)
            for expected in ga.resolves_to
        )
        if not matched:
            failures.append(
                f'"{_dn(ga.entity)}" not grounded to any of {ga.resolves_to} '
                f"(found: {doc_names})"
            )
    return LayerResult(
        layer="grounding",
        passed=len(assertions) - len(failures),
        total=len(assertions),
        failures=failures,
    )


def _check_glossary(
    relational,
    assertions: list[GlossaryAssertion],
    session_id: str,
    user_id: str,
) -> LayerResult:
    failures = []
    for ga in assertions:
        term = relational.get_glossary_term_by_name_or_alias(ga.name, session_id, user_id=user_id)
        if not term:
            # Fall back to entity existence — the unified glossary includes
            # self-describing entities that have no glossary_terms row.
            entity = relational.find_entity_by_name(
                ga.name, session_id=session_id, cross_session=True,
            )
            if not entity:
                failures.append(f'glossary term "{_dn(ga.name)}" not found')
                continue
            # Entity exists but no glossary definition
            if ga.has_definition:
                failures.append(f'"{_dn(ga.name)}" exists as entity but has no glossary definition')
            # Check grounding on bare entity
            doc_names = relational.get_entity_document_names(entity.id)
            if not doc_names:
                failures.append(f'"{_dn(ga.name)}" is not grounded to any data source')
            # Can't check domain/parent on bare entity — skip those checks
            continue
        if ga.has_definition and not term.definition:
            failures.append(f'"{_dn(ga.name)}" has no definition')
        # Check grounding — term's entity must be grounded to at least one data source
        entity = relational.find_entity_by_name(term.name, session_id=session_id, cross_session=True)
        if not entity:
            failures.append(f'"{_dn(ga.name)}" has glossary term but no entity')
        else:
            doc_names = relational.get_entity_document_names(entity.id)
            if not doc_names:
                failures.append(f'"{_dn(ga.name)}" is not grounded to any data source')
        if ga.domain and term.domain != ga.domain:
            failures.append(f'"{_dn(ga.name)}" domain: expected "{ga.domain}", got "{term.domain}"')
        if ga.parent:
            if not term.parent_id:
                failures.append(f'"{_dn(ga.name)}" has no parent (expected "{_dn(ga.parent)}")')
            else:
                parent = relational.get_glossary_term_by_id(term.parent_id)
                parent_name = parent.name if parent else None
                if not parent_name or parent_name.lower() != ga.parent.lower():
                    failures.append(
                        f'"{_dn(ga.name)}" parent: expected "{_dn(ga.parent)}", got "{_dn(parent_name) if parent_name else None}"'
                    )
    return LayerResult(
        layer="glossary",
        passed=len(assertions) - len(failures),
        total=len(assertions),
        failures=failures,
    )


def _check_relationships(
    relational, assertions: list[RelationshipAssertion], session_id: str,
    user_id: str = "default", domain_ids: list[str] | None = None,
) -> LayerResult:
    failures = []
    for ra in assertions:
        # Resolve aliases to canonical entity names for subject and object
        subject_entity = _resolve_entity_name(relational, ra.subject, session_id, user_id, domain_ids or [])
        subject_name = subject_entity.name if subject_entity else ra.subject

        rels = relational.get_relationships_for_entity(subject_name, session_id)

        object_entity = _resolve_entity_name(relational, ra.object, session_id, user_id, domain_ids or [])
        object_name = (object_entity.name if object_entity else ra.object).lower()

        matched = any(
            r["verb"].lower() == ra.verb.lower()
            and r["object_name"].lower() == object_name
            and (r["confidence"] or 0) >= ra.min_confidence
            for r in rels
        )
        if not matched:
            failures.append(
                f'relationship "{_dn(ra.subject)} {ra.verb} {_dn(ra.object)}" '
                f"(min_confidence={ra.min_confidence}) not found"
            )
    return LayerResult(
        layer="relationships",
        passed=len(assertions) - len(failures),
        total=len(assertions),
        failures=failures,
    )
