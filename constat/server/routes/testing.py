# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Regression testing endpoints — run golden question assertions via the API."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from constat.server.auth import CurrentUserId
from constat.server.config import ServerConfig
from constat.server.models import (
    GoldenQuestionExpectations,
    GoldenQuestionRequest,
    GoldenQuestionResponse,
    TestableDomainInfo,
    TestDomainResult,
    TestEndToEndResult,
    TestLayerResult,
    TestQuestionResult,
    TestRunResponse,
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


@router.post(
    "/{session_id}/tests/run",
    response_model=TestRunResponse,
)
async def run_tests(
    session_id: str,
    body: RunTestsRequest,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
) -> TestRunResponse:
    """Run golden question regression tests."""
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

    from constat.testing.runner import run_domain_test, run_domain_test_e2e

    run_fn = run_domain_test_e2e if body.include_e2e else run_domain_test

    loop = asyncio.get_event_loop()
    domain_results: list[TestDomainResult] = []

    for df in domain_filenames:
        result = await loop.run_in_executor(
            None, run_fn, config, df, tag_list, session_id, user_id,
        )

        # Get display name
        dc = config.load_domain(df)
        domain_name = dc.name if dc else df

        questions = [
            TestQuestionResult(
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
            )
            for qr in result.questions
        ]

        domain_results.append(
            TestDomainResult(
                domain=df,
                domain_name=domain_name,
                passed_count=result.passed_count,
                failed_count=result.failed_count,
                questions=questions,
            )
        )

    total_passed = sum(d.passed_count for d in domain_results)
    total_failed = sum(d.failed_count for d in domain_results)

    return TestRunResponse(
        domains=domain_results,
        total_passed=total_passed,
        total_failed=total_failed,
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

    domain_path, data = _read_domain_yaml(dc)
    gq_list = data.setdefault("golden_questions", [])
    new_entry = {"question": body.question, "tags": body.tags, "expect": body.expect.model_dump()}
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
