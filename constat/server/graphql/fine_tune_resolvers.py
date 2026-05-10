# Copyright (c) 2025 Kenneth Stott
# Canary: ae610b82-ba09-4149-bb79-cf136092c1a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for fine-tuning operations (Phase 9)."""

from __future__ import annotations

import logging
from typing import Optional

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    DeleteResultType,
    FineTuneJobType,
    FineTuneProviderType,
    StartFineTuneInput,
)

logger = logging.getLogger(__name__)


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


def _get_manager(info: Info):
    from constat.learning.fine_tune_manager import FineTuneManager

    manager = getattr(info.context.request.app.state, "fine_tune_manager", None)
    if not manager:
        from constat.learning.fine_tune_registry import FineTuneRegistry
        from constat.storage.learnings import LearningStore

        user_id = info.context.user_id or "default"
        manager = FineTuneManager(FineTuneRegistry(), LearningStore(user_id=user_id))
        info.context.request.app.state.fine_tune_manager = manager
    return manager


def _model_to_type(model) -> FineTuneJobType:
    return FineTuneJobType(
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


@strawberry.type
class Query:
    @strawberry.field
    async def fine_tune_jobs(
        self,
        info: Info,
        status: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> list[FineTuneJobType]:
        _require_auth(info)
        manager = _get_manager(info)
        models = manager.registry.list(status=status, domain=domain)
        return [_model_to_type(m) for m in models]

    @strawberry.field
    async def fine_tune_job(self, info: Info, model_id: str) -> FineTuneJobType:
        _require_auth(info)
        manager = _get_manager(info)
        try:
            model = manager.check_status(model_id)
        except KeyError:
            raise ValueError(f"Model {model_id} not found")
        return _model_to_type(model)

    @strawberry.field
    async def fine_tune_providers(self, info: Info) -> list[FineTuneProviderType]:
        _require_auth(info)
        from constat.learning.fine_tune_providers import get_available_providers

        providers = get_available_providers()
        return [
            FineTuneProviderType(
                name=p["name"],
                models=p.get("models", []),
            )
            for p in providers
        ]


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def start_fine_tune_job(
        self, info: Info, input: StartFineTuneInput
    ) -> FineTuneJobType:
        _require_auth(info)
        manager = _get_manager(info)
        try:
            model = manager.start_fine_tune(
                name=input.name,
                provider=input.provider,
                base_model=input.base_model,
                task_types=input.task_types,
                domain=input.domain,
                include=input.include or ["corrections", "rules"],
                min_confidence=input.min_confidence or 0.0,
                hyperparams=input.hyperparams,
            )
        except ValueError as e:
            raise ValueError(str(e))
        return _model_to_type(model)

    @strawberry.mutation
    async def cancel_fine_tune_job(self, info: Info, model_id: str) -> FineTuneJobType:
        _require_auth(info)
        manager = _get_manager(info)
        try:
            manager.cancel(model_id)
        except KeyError:
            raise ValueError(f"Model {model_id} not found")
        except ValueError as e:
            raise ValueError(str(e))
        try:
            model = manager.check_status(model_id)
        except KeyError:
            raise ValueError(f"Model {model_id} not found after cancel")
        return _model_to_type(model)

    @strawberry.mutation
    async def delete_fine_tune_job(self, info: Info, model_id: str) -> DeleteResultType:
        _require_auth(info)
        manager = _get_manager(info)
        try:
            manager.delete(model_id)
        except KeyError:
            raise ValueError(f"Model {model_id} not found")
        return DeleteResultType(status="deleted", name=model_id)

    @strawberry.mutation
    async def recreate_fine_tune_job(
        self, info: Info, artifact_dir: str
    ) -> FineTuneJobType:
        _require_auth(info)
        manager = _get_manager(info)
        try:
            model = manager.recreate_from_artifact(artifact_dir)
        except FileNotFoundError as e:
            raise ValueError(str(e))
        except ValueError as e:
            raise ValueError(str(e))
        return _model_to_type(model)
