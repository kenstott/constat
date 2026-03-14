# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Regression testing endpoints — run golden question assertions via the API."""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from constat.server.auth import CurrentUserId
from constat.server.config import ServerConfig
from constat.server.models import (
    GoldenQuestionExpectations,
    GoldenQuestionRequest,
    GoldenQuestionResponse,
    TestableDomainInfo,
    TestEndToEndResult,
    TestLayerResult,
    TestQuestionResult,
)
from constat.server.permissions import get_user_permissions
from constat.server.session_manager import SessionManager
from constat.testing.grounding import (
    _GROUNDABLE_SOURCES,
    build_source_patterns,
)
from constat.testing.models import parse_golden_questions

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class RunTestsRequest(BaseModel):
    domains: list[str] = []
    tags: list[str] = []
    include_e2e: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _get_server_config(request: Request) -> ServerConfig:
    return request.app.state.server_config


def _can_modify_domain(domain, user_id: str, server_config: ServerConfig) -> bool:
    perms = get_user_permissions(server_config, user_id)
    if perms.is_admin:
        return True
    if domain.owner and domain.owner != user_id and domain.steward != user_id:
        return False
    return True


def _resolve_entity_to_store_name(
    relational, name: str, session_id: str, user_id: str, domain_ids: list[str],
) -> str | None:
    """Look up a premise label in the glossary/entity store.

    The glossary is the source of truth. Reasoning chains use glossary terms,
    entities, or aliases at the top of the chain. If a name doesn't resolve,
    it's not a valid grounded expectation and should be dropped.
    """
    # 1. Glossary term by name or alias (primary path)
    term = relational.get_glossary_term_by_name_or_alias(name, session_id, user_id=user_id)
    if term:
        return term.name

    # 2. Entity by name (case-insensitive)
    entity = relational.find_entity_by_name(
        name, domain_ids=domain_ids, session_id=session_id, cross_session=True,
    )
    if entity:
        return entity.name

    return None


def _resolve_expectations(
    relational, expect: dict, session_id: str, user_id: str, domain_ids: list[str],
) -> dict:
    """Resolve entity names in expectations to actual store names.

    When a reasoning chain auto-generates a test, premise labels like
    "Inventory Items" may not match the NER entity name "inventory items".
    This resolves each name, keeping the original if no match is found.
    """
    # Resolve entity names — only keep entities that resolve to real store names.
    # Unresolvable premise labels (transient facts, intermediate tables) are dropped.
    resolved_entities = []
    name_map: dict[str, str] = {}  # original -> resolved
    for name in expect.get("entities", []):
        resolved = _resolve_entity_to_store_name(relational, name, session_id, user_id, domain_ids)
        if resolved:
            name_map[name] = resolved
            resolved_entities.append(resolved)
        else:
            logger.debug(f"Dropping unresolvable entity from expectations: {name!r}")
    expect["entities"] = resolved_entities

    # Resolve entity names in grounding assertions — drop unresolvable
    resolved_grounding = []
    for ga in expect.get("grounding", []):
        entity = ga.get("entity", "")
        if entity in name_map:
            ga["entity"] = name_map[entity]
            resolved_grounding.append(ga)
        else:
            resolved = _resolve_entity_to_store_name(relational, entity, session_id, user_id, domain_ids)
            if resolved:
                ga["entity"] = resolved
                resolved_grounding.append(ga)
            else:
                logger.debug(f"Dropping unresolvable grounding assertion: {entity!r}")
    expect["grounding"] = resolved_grounding

    # Resolve entity names in relationship assertions
    for ra in expect.get("relationships", []):
        for field in ("subject", "object"):
            val = ra.get(field, "")
            if val in name_map:
                ra[field] = name_map[val]
            else:
                resolved = _resolve_entity_to_store_name(relational, val, session_id, user_id, domain_ids)
                if resolved:
                    ra[field] = resolved

    return expect


def _gq_to_response(index: int, raw: dict) -> GoldenQuestionResponse:
    expect_raw = raw.get("expect", {})
    return GoldenQuestionResponse(
        index=index,
        question=raw.get("question", ""),
        tags=raw.get("tags", []),
        expect=GoldenQuestionExpectations(
            entities=expect_raw.get("entities", []),
            grounding=expect_raw.get("grounding", []),
            relationships=expect_raw.get("relationships", []),
            glossary=expect_raw.get("glossary", []),
            end_to_end=expect_raw.get("end_to_end"),
            suggested_question=expect_raw.get("suggested_question"),
            step_hints=expect_raw.get("step_hints", []),
        ),
        objectives=raw.get("objectives", []),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{session_id}/tests/domains",
    response_model=list[TestableDomainInfo],
)
async def list_testable_domains(
    session_id: str,
    sm: SessionManager = Depends(_get_session_manager),
) -> list[TestableDomainInfo]:
    """List domains that have golden_questions configured."""
    managed = sm.get_session(session_id)
    config = managed.session.config

    results: list[TestableDomainInfo] = []
    for filename, dc in config.domains.items():
        if not dc.golden_questions:
            continue
        questions = parse_golden_questions(dc.golden_questions)
        all_tags: set[str] = set()
        for q in questions:
            all_tags.update(q.tags)
        results.append(
            TestableDomainInfo(
                filename=filename,
                name=dc.name,
                question_count=len(questions),
                tags=sorted(all_tags),
            )
        )
    return results


def _resolve_entity_with_term(
    relational, name: str, session_id: str, user_id: str, domain_ids: list[str],
) -> tuple[str | None, object | None]:
    """Resolve a premise name, returning (resolved_name, glossary_term_or_none)."""
    term = relational.get_glossary_term_by_name_or_alias(name, session_id, user_id=user_id)
    if term:
        return term.name, term
    entity = relational.find_entity_by_name(
        name, domain_ids=domain_ids, session_id=session_id, cross_session=True,
    )
    if entity:
        return entity.name, None
    return None, None


def _generate_test_metadata(
    config, original_question: str | None, proof_summary: str | None,
    entities: list[str],
) -> tuple[str | None, str | None]:
    """Generate test question and semantic_match criteria.

    If the original question has no follow-ups, use it directly.
    Only use LLM to merge when follow-ups exist.

    Returns (suggested_question, semantic_match).
    """
    from constat.providers.router import TaskRouter

    has_followups = original_question and "Follow-up requests:" in original_question

    if original_question and not has_followups:
        # Single question — strip any "Original request:" prefix, use as-is
        question = original_question.removeprefix("Original request:").strip()
        # Still need LLM for semantic_match criteria only
        router = TaskRouter(config.llm)
        system = (
            "Given a data analytics question, produce a single-line criteria string "
            "describing what a correct answer should contain. "
            "Reply with exactly one line starting with CRITERIA:"
        )
        response = router.generate(system=system, user_message=f"Question: {question}")
        semantic_match = None
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CRITERIA:"):
                semantic_match = line.split(":", 1)[1].strip()
                break
        return question, semantic_match

    # Multiple questions/follow-ups — LLM merges into single coherent question
    router = TaskRouter(config.llm)
    system = (
        "You are merging a multi-turn conversation into a single test question. "
        "The input contains an original question and follow-up corrections/refinements. "
        "Produce a SINGLE question that incorporates ALL constraints from ALL turns. "
        "If a follow-up contradicts the original, the follow-up takes precedence. "
        "DO NOT reference proof results, data values, row counts, or execution details. "
        "Only reference the user's INTENT: what they want computed, from what sources, "
        "with what filters and exclusions.\n\n"
        "Reply with exactly two lines:\n"
        "QUESTION: <merged question preserving all constraints>\n"
        "CRITERIA: <what a correct answer should demonstrate>"
    )
    user_message = original_question or ""
    response = router.generate(system=system, user_message=user_message)

    suggested_question = None
    semantic_match = None
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("QUESTION:"):
            suggested_question = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CRITERIA:"):
            semantic_match = line.split(":", 1)[1].strip()

    return suggested_question, semantic_match


class ProofNodeInput(BaseModel):
    """A proof node sent from the client."""
    id: str = ""
    name: str = ""
    source: str = ""
    source_name: Optional[str] = None
    table_name: Optional[str] = None
    api_endpoint: Optional[str] = None
    status: str = ""


class ExtractExpectationsRequest(BaseModel):
    """Request body with proof nodes from the client."""
    proof_nodes: list[ProofNodeInput] = []
    original_question: Optional[str] = None
    proof_summary: Optional[str] = None


@router.post(
    "/{session_id}/tests/expectations",
    response_model=GoldenQuestionExpectations,
)
async def extract_expectations(
    session_id: str,
    user_id: CurrentUserId,
    body: Optional[ExtractExpectationsRequest] = None,
    sm: SessionManager = Depends(_get_session_manager),
) -> GoldenQuestionExpectations:
    """Build golden-question expectations from proof nodes.

    Accepts proof nodes in the request body (from client-side proofStore).
    Falls back to server-side last_proof_result if no body provided.

    Uses the glossary/entity store to resolve premise names to real entity
    names.  Only premises with groundable sources (database, document, api)
    are included.  Unresolvable premise labels are dropped — if they don't
    appear in the glossary, the test would not be meaningful.
    """
    managed = sm.get_session(session_id)

    # Server-side proof nodes are richer (source_name, table_name, api_endpoint).
    # Client-side nodes are always available (survive session restore).
    # Use server-side when available, fall back to client-side.
    proof_nodes: list[dict] = []
    proof = managed.session.last_proof_result
    if proof and proof.get("proof_nodes"):
        proof_nodes = proof["proof_nodes"]
        logger.info(f"[extract_expectations] using {len(proof_nodes)} server-side proof nodes")
    elif body and body.proof_nodes:
        proof_nodes = [n.model_dump() for n in body.proof_nodes]
        logger.info(f"[extract_expectations] using {len(proof_nodes)} client-supplied proof nodes")

    if not proof_nodes:
        raise HTTPException(status_code=404, detail="No reasoning chain result available")

    doc_tools = managed.session.doc_tools
    relational = None
    if doc_tools and hasattr(doc_tools, '_vector_store'):
        relational = getattr(doc_tools._vector_store, '_relational', None)

    domain_ids = list(getattr(doc_tools, '_active_domain_ids', None) or []) if doc_tools else []

    entities: list[str] = []
    grounding: list[dict] = []
    glossary: list[dict] = []
    glossary_seen: set[str] = set()
    seen: set[str] = set()

    for node in proof_nodes:
        raw_source = node.get("source", "")
        # Source may be composite like "database:chinook" — split to get type
        source_type = raw_source.split(":")[0] if raw_source else ""
        # Only premises (groundable sources) create expectations;
        # all other outcomes (derived, cache, llm_knowledge, etc.) use LLM-as-judge
        logger.info(
            f"[extract_expectations] node name={node.get('name')!r} source={raw_source!r} "
            f"source_type={source_type!r} source_name={node.get('source_name')!r} "
            f"table_name={node.get('table_name')!r} status={node.get('status')!r}"
        )
        if source_type not in _GROUNDABLE_SOURCES:
            continue

        raw_name = node.get("name", "")
        if not raw_name:
            continue
        # Strip fact-id prefix like "P1: " or "I2: " from DAG node names
        name = re.sub(r"^[A-Z]\d+:\s*", "", raw_name)
        if not name:
            continue

        # Resolve via glossary / entity store
        resolved_name = None
        term_obj = None
        if relational:
            resolved_name, term_obj = _resolve_entity_with_term(
                relational, name, session_id, user_id, domain_ids,
            )
        logger.info(f"[extract_expectations] stripped name={name!r} resolved={resolved_name!r} has_relational={relational is not None}")
        if not resolved_name:
            # Normalize raw name to snake_case for use as entity name
            resolved_name = name.strip().lower().replace(" ", "_")
        if resolved_name in seen:
            continue
        seen.add(resolved_name)

        entities.append(resolved_name)

        # Build grounding assertion from proof node source data.
        resolves_to = build_source_patterns(node)
        if not resolves_to:
            resolves_to = [raw_source]

        grounding.append({"entity": resolved_name, "resolves_to": resolves_to, "strict": True})

        # Build glossary assertion if resolved via glossary term
        if term_obj and resolved_name not in glossary_seen:
            glossary_seen.add(resolved_name)
            ga = {"name": resolved_name, "has_definition": bool(term_obj.definition)}
            if term_obj.domain:
                ga["domain"] = term_obj.domain
            glossary.append(ga)

    # Extract relationships between resolved entities
    relationships: list[dict] = []
    if relational and entities:
        entity_set = set(e.lower() for e in entities)
        rel_seen: set[tuple[str, str, str]] = set()
        for entity_name in entities:
            try:
                rels = relational.get_relationships_for_entity(entity_name, session_id)
            except Exception as e:
                logger.warning(f"[extract_expectations] failed to get relationships for {entity_name!r}: {e}")
                continue
            for r in rels:
                subj = r["subject_name"]
                obj = r["object_name"]
                # Only include if both ends are in resolved entities
                if subj.lower() not in entity_set or obj.lower() not in entity_set:
                    continue
                triple = (subj.lower(), r["verb"].lower(), obj.lower())
                if triple in rel_seen:
                    continue
                rel_seen.add(triple)
                relationships.append({
                    "subject": subj,
                    "verb": r["verb"],
                    "object": obj,
                    "min_confidence": r.get("confidence", 0.0),
                })
        logger.info(f"[extract_expectations] extracted {len(relationships)} relationships from entity store")

    logger.info(
        f"[extract_expectations] result: {len(entities)} entities, {len(grounding)} grounding, "
        f"{len(relationships)} relationships, {len(glossary)} glossary"
    )

    # Use LLM to generate a concise test question name and semantic_match criteria
    suggested_question = None
    end_to_end = None
    proof_summary = body.proof_summary if body else None

    # Server-side proof result has the real analytical question (not slash commands).
    # Always prefer it over client-supplied originalQuestion.
    original_question = None
    proof = getattr(managed.session, 'last_proof_result', None)
    if proof:
        original_question = proof.get("problem")
    if not original_question:
        client_q = body.original_question if body else None
        # Skip slash commands
        if client_q and not client_q.strip().startswith("/"):
            original_question = client_q

    if original_question or entities:
        try:
            config = managed.session.config
            suggested_question, semantic_match = _generate_test_metadata(
                config, original_question, proof_summary, entities,
            )
            if semantic_match:
                end_to_end = {"semantic_match": semantic_match}
        except Exception as e:
            logger.warning(f"Failed to generate test metadata via LLM: {e}")

    # Extract structured objectives from proof result
    objectives: list[str] = []
    if original_question:
        if "Follow-up requests:" in original_question:
            # Parse combined problem into structured objectives
            parts = original_question.split("Follow-up requests:")
            main_q = parts[0].removeprefix("Original request:").strip()
            if main_q:
                objectives.append(main_q)
            if len(parts) > 1:
                followup_text = parts[1].split("Prove all of the above")[0].strip()
                for line in followup_text.split("\n"):
                    line = line.strip().lstrip("- ").strip()
                    if line:
                        objectives.append(line)
        else:
            objectives.append(original_question.removeprefix("Original request:").strip())

    # Capture step code hints from the exploratory session — these provide
    # reference SQL/Python with correct table and column names that the
    # reasoning chain uses to generate accurate inference code.
    step_hints: list[dict] = []
    session = managed.session
    if session.history and session.session_id:
        try:
            step_codes = session.history.list_step_codes(session.session_id)
            step_hints = [
                {"step_number": s.get("step_number"), "goal": s.get("goal", ""), "code": s.get("code", "")}
                for s in step_codes if s.get("code")
            ]
            logger.info(f"[extract_expectations] captured {len(step_hints)} step code hints")
        except Exception as e:
            logger.debug(f"[extract_expectations] could not load step codes: {e}")

    return GoldenQuestionExpectations(
        entities=entities,
        grounding=grounding,
        relationships=relationships,
        glossary=glossary,
        end_to_end=end_to_end,
        suggested_question=suggested_question,
        objectives=objectives,
        step_hints=step_hints,
    )


def _question_result_to_dict(qr) -> dict:
    """Convert a runner QuestionResult to an API-serializable dict."""
    return TestQuestionResult(
        question=qr.question,
        tags=qr.tags,
        passed=qr.passed,
        layers=[
            TestLayerResult(
                layer=lr.layer,
                passed=lr.passed,
                total=lr.total,
                failures=lr.failures,
            )
            for lr in qr.layers
        ],
        end_to_end=TestEndToEndResult(
            passed=qr.end_to_end.passed,
            answer=qr.end_to_end.answer,
            judge_reasoning=qr.end_to_end.judge_reasoning,
            failures=qr.end_to_end.failures,
            duration_s=qr.end_to_end.duration_s,
        ) if qr.end_to_end else None,
    ).model_dump()


@router.post("/{session_id}/tests/run")
async def run_tests(
    session_id: str,
    body: RunTestsRequest,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
):
    """Run golden question regression tests, streaming progress via SSE."""
    managed = sm.get_session(session_id)
    config = managed.session.config

    # Determine which domains to test
    if body.domains:
        domain_filenames = body.domains
    else:
        domain_filenames = [
            name for name, dc in config.domains.items()
            if dc.golden_questions
        ]

    tag_list = body.tags if body.tags else None

    import queue
    import threading
    from constat.testing.runner import iter_domain_test

    async def event_stream():
        domain_results: list[dict] = []

        for df in domain_filenames:
            evt_queue: queue.Queue = queue.Queue()
            thread_error: list[Exception] = []

            def _run(d=df):
                try:
                    for evt in iter_domain_test(
                        config, d, tag_list, session_id, user_id,
                        include_e2e=body.include_e2e,
                    ):
                        evt_queue.put(evt)
                except Exception as exc:
                    logger.exception(f"Test runner error for domain {d}")
                    thread_error.append(exc)
                finally:
                    evt_queue.put(None)  # sentinel

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            domain_questions: list[dict] = []
            domain_name = df
            loop = asyncio.get_event_loop()

            while True:
                evt = await loop.run_in_executor(None, evt_queue.get)
                if evt is None:
                    break

                domain_name = evt.domain_name or df
                payload = {
                    "event": evt.event,
                    "domain": evt.domain,
                    "domain_name": domain_name,
                    "question": evt.question,
                    "question_index": evt.question_index,
                    "question_total": evt.question_total,
                    "phase": evt.phase,
                    "detail": evt.detail,
                }
                if evt.result:
                    qr_dict = _question_result_to_dict(evt.result)
                    payload["result"] = qr_dict
                    domain_questions.append(qr_dict)

                yield f"data: {json.dumps(payload)}\n\n"

            thread.join()

            # If the thread errored, emit an error event
            if thread_error:
                err_payload = {
                    "event": "error",
                    "domain": df,
                    "domain_name": domain_name,
                    "message": str(thread_error[0]),
                }
                yield f"data: {json.dumps(err_payload)}\n\n"

            # Build domain summary
            passed = sum(1 for qd in domain_questions if qd.get("passed"))
            failed = len(domain_questions) - passed
            domain_results.append({
                "domain": df,
                "domain_name": domain_name,
                "passed_count": passed,
                "failed_count": failed,
                "questions": domain_questions,
            })

        # Final complete event with full TestRunResponse
        total_passed = sum(d["passed_count"] for d in domain_results)
        total_failed = sum(d["failed_count"] for d in domain_results)
        final = {
            "domains": domain_results,
            "total_passed": total_passed,
            "total_failed": total_failed,
        }
        yield f"data: {json.dumps({'event': 'complete', 'result': final})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Golden Question CRUD
# ---------------------------------------------------------------------------


def _load_domain_or_404(config, domain_filename: str):
    domain = config.load_domain(domain_filename)
    if not domain:
        raise HTTPException(status_code=404, detail=f"Domain not found: {domain_filename}")
    return domain


def _read_domain_yaml(domain) -> tuple[Path, dict]:
    if not domain.source_path:
        raise HTTPException(status_code=400, detail="Domain has no source path")
    domain_path = Path(domain.source_path)
    if not domain_path.exists():
        raise HTTPException(status_code=404, detail=f"Domain file not found: {domain_path}")
    data = yaml.safe_load(domain_path.read_text()) or {}
    return domain_path, data


def _write_domain_yaml(domain_path: Path, data: dict, domain) -> None:
    domain_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    domain.golden_questions = data.get("golden_questions", [])


@router.get(
    "/{session_id}/tests/{domain}/questions",
    response_model=list[GoldenQuestionResponse],
)
async def list_golden_questions(
    session_id: str,
    domain: str,
    sm: SessionManager = Depends(_get_session_manager),
) -> list[GoldenQuestionResponse]:
    managed = sm.get_session(session_id)
    dc = _load_domain_or_404(managed.session.config, domain)
    return [_gq_to_response(i, q) for i, q in enumerate(dc.golden_questions)]


@router.post(
    "/{session_id}/tests/{domain}/questions",
    response_model=GoldenQuestionResponse,
    status_code=201,
)
async def create_golden_question(
    session_id: str,
    domain: str,
    body: GoldenQuestionRequest,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
    server_config: ServerConfig = Depends(_get_server_config),
) -> GoldenQuestionResponse:
    managed = sm.get_session(session_id)
    dc = _load_domain_or_404(managed.session.config, domain)
    if not _can_modify_domain(dc, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this domain")

    # Resolve entity names against the store so auto-generated tests use
    # actual entity names rather than LLM premise labels.
    # Skip resolution if expectations already have grounding — they came from
    # extract_expectations which already resolved names. Double-resolving drops
    # entities that were normalized to snake_case but don't exist in the store.
    expect = body.expect.model_dump()
    has_grounding = bool(expect.get("grounding"))
    if has_grounding:
        logger.info(f"Skipping re-resolution: expectations already have {len(expect['grounding'])} grounding entries (pre-resolved)")
    else:
        doc_tools = managed.session.doc_tools
        if doc_tools and hasattr(doc_tools, '_vector_store'):
            relational = getattr(doc_tools._vector_store, '_relational', None)
            if relational:
                domain_ids = list(getattr(doc_tools, '_active_domain_ids', None) or [])
                logger.info(f"Resolving expectations: entities={expect.get('entities')}, domain_ids={domain_ids}, session_id={session_id}")
                expect = _resolve_expectations(relational, expect, session_id, user_id, domain_ids)
                logger.info(f"Resolved expectations: entities={expect.get('entities')}, grounding={expect.get('grounding')}")
            else:
                logger.warning("Cannot resolve expectations: no relational store available")
        else:
            logger.warning(f"Cannot resolve expectations: doc_tools={doc_tools is not None}, has _vector_store={hasattr(doc_tools, '_vector_store') if doc_tools else False}")

    domain_path, data = _read_domain_yaml(dc)
    gq_list = data.setdefault("golden_questions", [])
    new_entry = {"question": body.question, "tags": body.tags, "expect": expect}
    if body.objectives:
        new_entry["objectives"] = body.objectives
    # Step hints from expect (captured from exploratory session)
    step_hints = expect.get("step_hints") if isinstance(expect, dict) else getattr(body.expect, 'step_hints', [])
    if step_hints:
        new_entry["step_hints"] = step_hints
    gq_list.append(new_entry)
    _write_domain_yaml(domain_path, data, dc)

    return _gq_to_response(len(gq_list) - 1, new_entry)


@router.put(
    "/{session_id}/tests/{domain}/questions/{index}",
    response_model=GoldenQuestionResponse,
)
async def update_golden_question(
    session_id: str,
    domain: str,
    index: int,
    body: GoldenQuestionRequest,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
    server_config: ServerConfig = Depends(_get_server_config),
) -> GoldenQuestionResponse:
    managed = sm.get_session(session_id)
    dc = _load_domain_or_404(managed.session.config, domain)
    if not _can_modify_domain(dc, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this domain")

    domain_path, data = _read_domain_yaml(dc)
    gq_list = data.get("golden_questions", [])
    if index < 0 or index >= len(gq_list):
        raise HTTPException(status_code=404, detail=f"Golden question index {index} out of range")

    updated_entry = {"question": body.question, "tags": body.tags, "expect": body.expect.model_dump()}
    if body.objectives:
        updated_entry["objectives"] = body.objectives
    gq_list[index] = updated_entry
    data["golden_questions"] = gq_list
    _write_domain_yaml(domain_path, data, dc)

    return _gq_to_response(index, updated_entry)


@router.delete(
    "/{session_id}/tests/{domain}/questions/{index}",
    status_code=204,
)
async def delete_golden_question(
    session_id: str,
    domain: str,
    index: int,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
    server_config: ServerConfig = Depends(_get_server_config),
) -> None:
    managed = sm.get_session(session_id)
    dc = _load_domain_or_404(managed.session.config, domain)
    if not _can_modify_domain(dc, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify this domain")

    domain_path, data = _read_domain_yaml(dc)
    gq_list = data.get("golden_questions", [])
    if index < 0 or index >= len(gq_list):
        raise HTTPException(status_code=404, detail=f"Golden question index {index} out of range")

    gq_list.pop(index)
    data["golden_questions"] = gq_list
    _write_domain_yaml(domain_path, data, dc)


@router.post(
    "/{session_id}/tests/{domain}/questions/{index}/move",
    response_model=GoldenQuestionResponse,
)
async def move_golden_question(
    session_id: str,
    domain: str,
    index: int,
    body: dict,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
    server_config: ServerConfig = Depends(_get_server_config),
) -> GoldenQuestionResponse:
    """Move a golden question from one domain to another."""
    target_domain = body.get("target_domain")
    if not target_domain:
        raise HTTPException(status_code=400, detail="target_domain is required")

    managed = sm.get_session(session_id)
    config = managed.session.config

    # Load source domain and remove the question
    src_dc = _load_domain_or_404(config, domain)
    if not _can_modify_domain(src_dc, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify the source domain")

    src_path, src_data = _read_domain_yaml(src_dc)
    src_list = src_data.get("golden_questions", [])
    if index < 0 or index >= len(src_list):
        raise HTTPException(status_code=404, detail=f"Golden question index {index} out of range")

    entry = src_list.pop(index)

    # Load target domain and add the question
    tgt_dc = _load_domain_or_404(config, target_domain)
    if not _can_modify_domain(tgt_dc, user_id, server_config):
        raise HTTPException(status_code=403, detail="You do not have permission to modify the target domain")

    tgt_path, tgt_data = _read_domain_yaml(tgt_dc)
    tgt_list = tgt_data.setdefault("golden_questions", [])
    tgt_list.append(entry)

    # Write both files
    _write_domain_yaml(src_path, src_data, src_dc)
    _write_domain_yaml(tgt_path, tgt_data, tgt_dc)

    return _gq_to_response(len(tgt_list) - 1, entry)
