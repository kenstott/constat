# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Regression testing endpoints — run golden question assertions via the API."""

import asyncio
import json
import logging
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
    if domain.tier == "system":
        return False
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
        ),
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


_GROUNDABLE_SOURCES = {"database", "document", "api", "embedded"}


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

    # Prefer client-supplied proof nodes, fall back to server-side state
    proof_nodes: list[dict] = []
    if body and body.proof_nodes:
        proof_nodes = [n.model_dump() for n in body.proof_nodes]
    else:
        proof = managed.session.last_proof_result
        if proof:
            proof_nodes = proof.get("proof_nodes", [])

    if not proof_nodes:
        raise HTTPException(status_code=404, detail="No reasoning chain result available")

    doc_tools = managed.session.doc_tools
    relational = None
    if doc_tools and hasattr(doc_tools, '_vector_store'):
        relational = getattr(doc_tools._vector_store, '_relational', None)

    domain_ids = list(getattr(doc_tools, '_active_domain_ids', None) or []) if doc_tools else []

    entities: list[str] = []
    grounding: list[dict] = []
    seen: set[str] = set()

    for node in proof_nodes:
        source = node.get("source", "")
        # Only groundable sources
        if source not in _GROUNDABLE_SOURCES:
            continue
        name = node.get("name", "")
        if not name:
            continue

        # Resolve via glossary / entity store
        resolved_name = None
        if relational:
            resolved_name = _resolve_entity_to_store_name(
                relational, name, session_id, user_id, domain_ids,
            )
        if not resolved_name:
            logger.debug(f"Dropping unresolvable premise from expectations: {name!r} (source={source})")
            continue
        if resolved_name in seen:
            continue
        seen.add(resolved_name)

        entities.append(resolved_name)

        # Default: include specific source grounding. If the underlying resource
        # changes (table renamed, moved from API to DB), the test should flag it.
        # Users can remove grounding assertions if they want looser validation.
        resolves_to: list[str] = []
        source_name = node.get("source_name")  # e.g., database name
        table_name = node.get("table_name")  # e.g., actual table
        api_endpoint = node.get("api_endpoint")

        if source == "database" and source_name and table_name:
            resolves_to.append(f"schema:{source_name}.{table_name}")
        elif source == "database" and source_name:
            resolves_to.append(f"schema:{source_name}")
        elif source == "api" and api_endpoint:
            resolves_to.append(f"api:{api_endpoint}")
        elif source == "api" and source_name:
            resolves_to.append(f"api:{source_name}")
        elif source == "document":
            resolves_to.append("document")
        else:
            resolves_to.append(source)

        grounding.append({"entity": resolved_name, "resolves_to": resolves_to})

    return GoldenQuestionExpectations(
        entities=entities,
        grounding=grounding,
        relationships=[],
        glossary=[],
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
    # Only keep entities that actually resolve — drop unresolvable ones.
    expect = body.expect.model_dump()
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
