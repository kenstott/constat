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
import queue
import threading
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
    GroundingAssertion,
    LayerResult,
    QuestionResult,
    RelationshipAssertion,
    TermAssertion,
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
    event: str  # domain_start, question_start, question_done, domain_done, e2e_progress
    domain: str
    domain_name: str = ""
    question: str = ""
    question_index: int = 0
    question_total: int = 0
    phase: str = ""  # "metadata" or "e2e"
    detail: str = ""  # Sub-step description: "Planning...", "Step 2: join tables", etc.
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

        # Yield layer-level detail events during metadata checks
        for layer_evt in _iter_question_layers(
            relational, q, session_id, user_id, domain_filename,
            domain_name, i, total,
        ):
            if isinstance(layer_evt, TestProgressEvent):
                yield layer_evt
            else:
                qr = layer_evt  # final result

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
                # Run e2e with progress events via thread+queue
                yield from _run_e2e_with_events(
                    config, q, session_id, user_id,
                    domain_filename, domain_name, i, total, qr,
                )

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


def _iter_question_layers(
    relational, q: GoldenQuestion, session_id: str, user_id: str,
    domain_id: str, domain_name: str, question_index: int, question_total: int,
):
    """Run metadata checks, yielding detail events for each layer, then the QuestionResult."""
    layers: list[LayerResult] = []
    domain_ids = [domain_id]

    def _detail(text: str):
        return TestProgressEvent(
            event="e2e_progress", domain=domain_id, domain_name=domain_name,
            question=q.question, question_index=question_index,
            question_total=question_total, phase="metadata", detail=text,
        )

    if q.expect.terms:
        yield _detail("Checking terms...")
        layers.append(_check_terms(relational, q.expect.terms, session_id, domain_ids, user_id))

    if q.expect.grounding:
        yield _detail("Checking grounding...")
        layers.append(_check_grounding(relational, q.expect.grounding, session_id, domain_ids, user_id))

    if q.expect.relationships:
        yield _detail("Checking relationships...")
        layers.append(_check_relationships(relational, q.expect.relationships, session_id, user_id, domain_ids))

    yield QuestionResult(question=q.question, tags=q.tags, layers=layers)


# Map event_type -> human-readable detail for e2e progress
def _steps_count(data: dict) -> str:
    """Extract step count from event data (handles both int and list)."""
    steps = data.get("steps", 0)
    n = steps if isinstance(steps, int) else len(steps)
    return f"Plan ready ({n} steps)"


_EVENT_DETAIL_MAP = {
    "planning_start": "Planning...",
    "planning_complete": _steps_count,
    "plan_ready": _steps_count,
    # Premise/inference events (reasoning chain language)
    "premise_resolving": lambda data: f"Resolving premise: {data.get('fact_name', '')[:60]}",
    "premise_resolved": lambda data: f"Premise resolved: {data.get('fact_name', '')[:60]}",
    "inference_executing": lambda data: f"Computing inference: {data.get('operation', data.get('inference_id', ''))[:60]}",
    "inference_complete": lambda data: f"Inference complete: {data.get('inference_name', data.get('inference_id', ''))[:60]}",
    "inference_failed": lambda data: f"Inference failed: {data.get('fact_name', '')[:60]}",
    "fact_executing": lambda data: f"Executing: {data.get('fact_name', '')[:60]}",
    "fact_start": lambda data: f"Resolving: {data.get('fact_name', '')[:60]}",
    "fact_resolved": lambda data: f"Resolved: {data.get('fact_name', '')[:60]}",
    "fact_failed": lambda data: f"Failed: {data.get('fact_name', '')[:60]}",
    "fact_blocked": lambda data: f"Blocked: {data.get('fact_name', '')[:60]}",
    # SQL events
    "sql_generating": lambda data: f"Generating SQL for: {data.get('fact_name', '')[:60]}",
    "sql_executing": lambda data: f"Executing SQL for: {data.get('fact_name', '')[:60]}",
    # Wave execution
    "wave_step_start": lambda data: f"Executing (wave {data.get('wave', '?')}): {data.get('goal', '')[:60]}",
    "synthesizing": "Synthesizing answer...",
    "answer_ready": "Answer ready",
}


def _run_e2e_with_events(
    config: Config,
    gq: GoldenQuestion,
    session_id: str,
    user_id: str,
    domain_filename: str,
    domain_name: str,
    question_index: int,
    question_total: int,
    qr: QuestionResult,
):
    """Run e2e in a background thread, yielding progress events from session StepEvents."""
    evt_queue: queue.Queue = queue.Queue()
    result_holder: list[EndToEndResult] = []

    def _base_evt(detail: str) -> TestProgressEvent:
        return TestProgressEvent(
            event="e2e_progress", domain=domain_filename, domain_name=domain_name,
            question=gq.question, question_index=question_index,
            question_total=question_total, phase="e2e", detail=detail,
        )

    def _solve_thread():
        from constat.api.factory import create_api
        assertion = gq.expect.end_to_end
        failures: list[str] = []
        t0 = time.monotonic()

        test_session_id = f"test-{uuid.uuid4().hex[:12]}"
        logger.info(f"[E2E_TRACE] creating test API, session_id={test_session_id}")
        api = create_api(
            config,
            session_id=test_session_id,
            user_id=user_id,
            require_approval=False,
            auto_approve=True,
        )
        logger.info("[E2E_TRACE] test API created, loading domain resources")

        # Load ALL configured domains — same as real session startup.
        # The test domain may depend on documents/databases from other domains.
        active_domain_ids = []
        for df in config.domains:
            dc = config.load_domain(df)
            if not dc:
                continue
            active_domain_ids.append(df)
            if dc.databases and api.session.schema_manager:
                for db_name, db_cfg in dc.databases.items():
                    try:
                        success = api.session.schema_manager.add_database_dynamic(db_name, db_cfg)
                        if success:
                            logger.info(f"[E2E_TRACE] loaded domain database: {db_name}")
                    except Exception as e:
                        logger.error(f"[E2E_TRACE] error loading domain database {db_name}: {e}")
            api.session.add_domain_resources(
                domain_filename=df,
                databases=dc.databases,
                apis=dc.apis,
                documents=dc.documents,
            )

        if api.session.schema_manager:
            schema_entities = set(api.session.schema_manager.get_entity_names())
            api.session.doc_tools.set_schema_entities(schema_entities)
            schema_metadata = api.session.schema_manager.get_description_text()
            if schema_metadata:
                api.session.doc_tools.process_metadata_through_ner(schema_metadata, source_type="schema")

        api.session.doc_tools._active_domain_ids = active_domain_ids

        logger.info("[E2E_TRACE] registering event handler")

        def _on_event(event_type, data):
            handler = _EVENT_DETAIL_MAP.get(event_type)
            if handler is None:
                return
            detail = handler(data) if callable(handler) else handler
            evt_queue.put(detail)

        api.on_event(_on_event)

        # Set up objectives in the session datastore so prove_conversation()
        # sees the same question + follow-ups as the original reasoning chain.
        import json as _json
        if gq.objectives:
            original_q = gq.objectives[0]
            follow_ups = gq.objectives[1:] if len(gq.objectives) > 1 else []
        else:
            original_q = gq.question
            follow_ups = []

        # Ensure datastore exists
        api.session._ensure_session_datastore(original_q)
        api.session.datastore.set_session_meta("problem", original_q)
        if follow_ups:
            api.session.datastore.set_session_meta("follow_ups", _json.dumps(follow_ups))

        # Provide step code hints from the exploratory session — same as real /reason
        if gq.step_hints:
            api.session._proof_step_hints = gq.step_hints
            logger.info(f"[E2E_TRACE] loaded {len(gq.step_hints)} step code hints")

        try:
            logger.info("[E2E_TRACE] calling api.prove_conversation()")
            proof_result = api.prove_conversation()
            success = proof_result.get("success", False)
            error = proof_result.get("error")
            logger.info(f"[E2E_TRACE] prove_conversation() returned: success={success}, error={error}")
        except Exception as exc:
            logger.error(f"[E2E_TRACE] prove_conversation() raised: {exc}", exc_info=True)
            result_holder.append(EndToEndResult(
                passed=False, failures=[f"prove_conversation() raised: {exc}"],
                duration_s=time.monotonic() - t0,
            ))
            return
        duration = time.monotonic() - t0

        # Assertion checks
        if assertion.expect_success and not success:
            failures.append(f"Expected success but got error: {error}")
            result_holder.append(EndToEndResult(
                passed=False, answer=proof_result.get("final_answer"),
                failures=failures, duration_s=duration,
            ))
            return
        if not assertion.expect_success and success:
            failures.append("Expected failure but proof succeeded")
            result_holder.append(EndToEndResult(
                passed=False, answer=proof_result.get("final_answer"),
                failures=failures, duration_s=duration,
            ))
            return

        answer = proof_result.get("final_answer") or proof_result.get("output") or ""
        logger.info(f"[E2E_TRACE] answer length={len(answer)}, preview={answer[:200]}")

        # Collect artifact table data for judge context
        artifact_data = ""
        if api.session.datastore:
            try:
                tables = api.session.datastore.list_tables()
                for tbl in tables[:5]:
                    tbl_name = tbl["name"] if isinstance(tbl, dict) else tbl
                    df = api.session.datastore.load_dataframe(tbl_name)
                    if df is not None and len(df) > 0:
                        artifact_data += f"\n--- {tbl_name} ({len(df)} rows) ---\n"
                        artifact_data += df.head(20).to_string(index=False) + "\n"
            except Exception as e:
                logger.debug(f"[E2E_TRACE] Failed to collect artifact data: {e}")

        for substring in assertion.result_contains:
            evt_queue.put(f"Checking: result_contains '{substring[:40]}'")
            if substring.lower() not in answer.lower():
                failures.append(f'Answer missing expected substring: "{substring}"')

        # Check proof node count instead of step count
        proof_nodes = proof_result.get("proof_nodes", [])
        if len(proof_nodes) < assertion.plan_min_steps:
            failures.append(
                f"Expected at least {assertion.plan_min_steps} proof nodes, got {len(proof_nodes)}"
            )

        judge_reasoning = None
        if assertion.judge_prompt and answer:
            evt_queue.put("Checking: LLM judge...")
            passed, reasoning = _llm_judge(
                gq.question, answer, assertion.judge_prompt, config,
                artifact_data=artifact_data,
            )
            judge_reasoning = reasoning
            if not passed:
                failures.append(f"LLM judge failed: {reasoning}")

        result_holder.append(EndToEndResult(
            passed=len(failures) == 0,
            answer=answer[:2000] if answer else None,
            judge_reasoning=judge_reasoning,
            failures=failures,
            duration_s=duration,
        ))

    thread = threading.Thread(target=_solve_thread, daemon=True)
    thread.start()

    e2e_timeout = 300  # 5 minute timeout for e2e test
    e2e_start = time.monotonic()
    while thread.is_alive():
        if time.monotonic() - e2e_start > e2e_timeout:
            logger.error(f"[E2E_TRACE] Thread still alive after {e2e_timeout}s — aborting")
            break
        try:
            detail = evt_queue.get(timeout=0.5)
            yield _base_evt(detail)
        except queue.Empty:
            continue

    # Drain remaining events
    while not evt_queue.empty():
        detail = evt_queue.get_nowait()
        yield _base_evt(detail)

    thread.join(timeout=5)

    if result_holder:
        qr.end_to_end = result_holder[0]
    else:
        elapsed = time.monotonic() - e2e_start
        qr.end_to_end = EndToEndResult(
            passed=False,
            failures=[f"No result from e2e thread (elapsed={elapsed:.0f}s, alive={thread.is_alive()})"],
            duration_s=elapsed,
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

    # Collect artifact table data for judge context
    artifact_data = _collect_artifact_data(api, solve_result)

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

    # LLM judge
    judge_reasoning = None
    if assertion.judge_prompt and answer:
        passed, reasoning = _llm_judge(
            gq.question, answer, assertion.judge_prompt, config,
            artifact_data=artifact_data,
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


# Artifact types that can be read as text for the LLM judge
_TEXT_ARTIFACT_TYPES = {"csv", "json", "text", "markdown", "html"}


def _collect_artifact_data(api, solve_result) -> str:
    """Collect data from all artifacts created during solve.

    Includes DuckDB table samples and text-readable file artifacts (CSV, JSON,
    HTML, markdown, text). Binary artifacts (images, charts) are listed by name
    only.
    """
    from pathlib import Path

    parts: list[str] = []
    budget = 6000  # chars budget for artifact data
    used = 0

    # 1. Table artifacts from the session datastore
    try:
        datastore = api.session.datastore
        if datastore:
            table_names = list(solve_result.tables_created) if solve_result.tables_created else []
            if not table_names:
                table_list = datastore.list_tables()
                table_names = [t["name"] for t in table_list]
            for tname in table_names:
                if used >= budget:
                    break
                try:
                    df = datastore.query(f'SELECT * FROM "{tname}" LIMIT 15')
                    if df is not None and len(df) > 0:
                        chunk = f"--- Table: {tname} ({len(df)} rows shown) ---\n"
                        chunk += df.to_string(index=False, max_cols=10)
                        parts.append(chunk)
                        used += len(chunk)
                except Exception:
                    continue
    except Exception:
        pass

    # 2. File-based artifacts from the registry
    try:
        session = api.session
        registry = getattr(session, "registry", None)
        if registry:
            artifacts = registry.list_artifacts(session_id=session.session_id)
            for art in artifacts:
                if used >= budget:
                    break
                atype = art.artifact_type
                fpath = Path(art.file_path) if art.file_path else None

                if atype in _TEXT_ARTIFACT_TYPES and fpath and fpath.exists():
                    try:
                        content = fpath.read_text(errors="replace")[:2000]
                        label = f"--- {atype.upper()}: {art.name or fpath.name} ---"
                        chunk = f"{label}\n{content}"
                        parts.append(chunk)
                        used += len(chunk)
                    except Exception:
                        continue
                elif atype not in _TEXT_ARTIFACT_TYPES and fpath:
                    # Binary artifact — confirm existence for the judge
                    desc = f" - {art.description}" if art.description else ""
                    parts.append(f"--- {atype.upper()}: {art.name or fpath.name} (produced){desc} ---")
    except Exception:
        pass

    return "\n\n".join(parts)


def _llm_judge(
    question: str, answer: str, judge_prompt: str, config: Config,
    *, artifact_data: str = "",
) -> tuple[bool, str]:
    """Use the config's LLM to evaluate if answer meets judge_prompt.

    The judge_prompt is the complete system prompt including any criteria.
    Returns (passed, reasoning).
    """
    from constat.providers.router import TaskRouter

    router = TaskRouter(config.llm)
    user_parts = [
        f"Question: {question}",
        f"Answer summary: {answer[:3000]}",
    ]
    if artifact_data:
        user_parts.append(f"Computed Artifacts:\n{artifact_data[:6000]}")
    user_parts.append("Does this pass the test?")
    user_message = "\n\n".join(user_parts)
    logger.info(f"[LLM_JUDGE] Sending to judge: question={question[:80]}, answer_len={len(answer)}, artifact_len={len(artifact_data)}")
    response = router.generate(system=judge_prompt, user_message=user_message)
    first_line = response.strip().split("\n")[0].strip().upper()
    passed = first_line.startswith("YES")
    logger.info(f"[LLM_JUDGE] Verdict: {'PASS' if passed else 'FAIL'} — {response.strip()[:200]}")
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

    if q.expect.terms:
        layers.append(_check_terms(relational, q.expect.terms, session_id, domain_ids, user_id))

    if q.expect.grounding:
        layers.append(_check_grounding(relational, q.expect.grounding, session_id, domain_ids, user_id))

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


def _check_terms(
    relational, assertions: list[TermAssertion], session_id: str,
    domain_ids: list[str], user_id: str = "default",
) -> LayerResult:
    """Unified term/entity/alias check — merges old entities + glossary layers."""
    failures = []
    for ta in assertions:
        # Try glossary first (richer match: term name, alias, definition)
        term = relational.get_glossary_term_by_name_or_alias(ta.name, session_id, user_id=user_id)
        if not term:
            # Fall back to entity existence
            entity = _resolve_entity_name(relational, ta.name, session_id, user_id, domain_ids)
            if not entity:
                # Last resort: proven grounding
                if relational.get_proven_grounding(ta.name):
                    continue
                failures.append(f'term "{_dn(ta.name)}" not found')
                continue
            # Entity exists but no glossary term
            if ta.has_definition:
                failures.append(f'"{_dn(ta.name)}" exists as entity but has no glossary definition')
            continue
        # Glossary term found — check optional constraints
        if ta.has_definition and not term.definition:
            failures.append(f'"{_dn(ta.name)}" has no definition')
        if ta.domain and term.domain != ta.domain:
            failures.append(f'"{_dn(ta.name)}" domain: expected "{ta.domain}", got "{term.domain}"')
        if ta.parent:
            if not term.parent_id:
                failures.append(f'"{_dn(ta.name)}" has no parent (expected "{_dn(ta.parent)}")')
            else:
                parent = relational.get_glossary_term_by_id(term.parent_id)
                parent_name = parent.name if parent else None
                if not parent_name or parent_name.lower() != ta.parent.lower():
                    failures.append(
                        f'"{_dn(ta.name)}" parent: expected "{_dn(ta.parent)}", got "{_dn(parent_name) if parent_name else None}"'
                    )
    return LayerResult(
        layer="terms",
        passed=len(assertions) - len(failures),
        total=len(assertions),
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
