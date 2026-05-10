# Copyright (c) 2025 Kenneth Stott
# Canary: 85a8868c-bd27-4419-9a53-243ec22b38a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fine-tuning REST endpoints."""

import logging
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from constat.learning.fine_tune_manager import FineTuneManager
from constat.learning.fine_tune_providers import get_available_providers

logger = logging.getLogger(__name__)

router = APIRouter()


class StartFineTuneRequest(BaseModel):
    name: str
    provider: str
    base_model: str
    task_types: list[str]
    domain: str | None = None
    include: list[str] = ["corrections", "rules"]
    min_confidence: float = 0.0
    hyperparams: dict | None = None


class RecreateFineTuneRequest(BaseModel):
    artifact_dir: str


class FineTuneJobResponse(BaseModel):
    id: str
    name: str
    provider: str
    base_model: str
    fine_tuned_model_id: str | None
    task_types: list[str]
    domain: str | None
    status: str
    created: str
    exemplar_count: int
    metrics: dict | None
    training_data_path: str | None = None


def _get_manager(request: Request, user_id: str = "default") -> FineTuneManager:
    manager = getattr(request.app.state, "fine_tune_manager", None)
    if not manager:
        from constat.learning.fine_tune_registry import FineTuneRegistry
        from constat.storage.learnings import LearningStore

        manager = FineTuneManager(FineTuneRegistry(), LearningStore(user_id=user_id))
        request.app.state.fine_tune_manager = manager
    return manager


def _model_to_response(model) -> FineTuneJobResponse:
    return FineTuneJobResponse(
        id=model.id,
        name=model.name,
        provider=model.provider,
        base_model=model.base_model,
        fine_tuned_model_id=model.fine_tuned_model_id or None,
        task_types=model.task_types,
        domain=model.domain,
        status=model.status,
        created=model.created,
        exemplar_count=model.exemplar_count,
        metrics=model.metrics,
        training_data_path=model.training_data_path,
    )


@router.post("/fine-tune/jobs", response_model=FineTuneJobResponse)
async def start_fine_tune(body: StartFineTuneRequest, request: Request) -> FineTuneJobResponse:
    manager = _get_manager(request)
    try:
        model = manager.start_fine_tune(
            name=body.name,
            provider=body.provider,
            base_model=body.base_model,
            task_types=body.task_types,
            domain=body.domain,
            include=body.include,
            min_confidence=body.min_confidence,
            hyperparams=body.hyperparams,
        )
        return _model_to_response(model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/fine-tune/jobs", response_model=list[FineTuneJobResponse])
async def list_fine_tune_jobs(
    request: Request,
    status: str | None = None,
    domain: str | None = None,
) -> list[FineTuneJobResponse]:
    manager = _get_manager(request)
    models = manager.registry.list(status=status, domain=domain)
    return [_model_to_response(m) for m in models]


@router.get("/fine-tune/jobs/{model_id}", response_model=FineTuneJobResponse)
async def get_fine_tune_job(model_id: str, request: Request) -> FineTuneJobResponse:
    manager = _get_manager(request)
    # Poll status if training
    try:
        model = manager.check_status(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return _model_to_response(model)


@router.post("/fine-tune/jobs/{model_id}/cancel")
async def cancel_fine_tune(model_id: str, request: Request) -> dict:
    manager = _get_manager(request)
    try:
        manager.cancel(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "cancelled", "id": model_id}


@router.delete("/fine-tune/jobs/{model_id}")
async def delete_fine_tune(model_id: str, request: Request) -> dict:
    manager = _get_manager(request)
    try:
        manager.delete(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {"status": "deleted", "id": model_id}


@router.post("/fine-tune/recreate", response_model=FineTuneJobResponse)
async def recreate_fine_tune(body: RecreateFineTuneRequest, request: Request) -> FineTuneJobResponse:
    """Recreate a fine-tune job from a saved training artifact directory."""
    manager = _get_manager(request)
    try:
        model = manager.recreate_from_artifact(body.artifact_dir)
        return _model_to_response(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/fine-tune/providers")
async def list_providers() -> list[dict]:
    return get_available_providers()
