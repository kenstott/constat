# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""YAML-backed registry of fine-tuned models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path.home() / ".constat" / "fine_tune_registry.yaml"


@dataclass
class FineTunedModel:
    """A fine-tuned model entry."""

    id: str
    name: str
    provider: str                    # "openai" | "together"
    base_model: str                  # "gpt-4o-mini-2024-07-18"
    fine_tuned_model_id: str         # Provider's model ID (ft:gpt-4o-mini:org:...)
    task_types: list[str]            # ["sql_generation", "python_analysis"]
    domain: str | None               # Domain filter
    status: str                      # "training" | "ready" | "failed" | "archived"
    provider_job_id: str             # Provider's job ID for polling
    created: str                     # ISO timestamp
    training_file_id: str            # Provider's uploaded file ID
    training_data_path: str | None = None  # Local path to saved training JSONL + manifest
    metrics: dict | None = None      # Training loss, etc.
    exemplar_count: int = 0          # Number of training examples


class FineTuneRegistry:
    """YAML-backed model registry."""

    def __init__(self, path: Path | None = None):
        self._path = path or DEFAULT_REGISTRY_PATH
        self._models: dict[str, FineTunedModel] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = yaml.safe_load(self._path.read_text()) or {}
            for entry in data.get("models", []):
                model = FineTunedModel(**entry)
                self._models[model.id] = model
        except Exception as e:
            logger.error(f"Failed to load fine-tune registry: {e}")

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"models": [asdict(m) for m in self._models.values()]}
        self._path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def add(self, model: FineTunedModel) -> None:
        self._models[model.id] = model
        self._save()

    def get(self, model_id: str) -> FineTunedModel | None:
        return self._models.get(model_id)

    def list(
        self,
        status: str | None = None,
        domain: str | None = None,
    ) -> list[FineTunedModel]:
        result = list(self._models.values())
        if status:
            result = [m for m in result if m.status == status]
        if domain:
            result = [m for m in result if m.domain == domain or m.domain is None]
        return result

    def update(self, model_id: str, **kwargs) -> None:
        model = self._models.get(model_id)
        if not model:
            raise KeyError(f"Model {model_id} not found")
        for key, value in kwargs.items():
            if not hasattr(model, key):
                raise ValueError(f"Unknown field: {key}")
            setattr(model, key, value)
        self._save()

    def remove(self, model_id: str) -> None:
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} not found")
        del self._models[model_id]
        self._save()

    def get_active_for_task(
        self,
        task_type: str,
        domain: str | None = None,
    ) -> list[FineTunedModel]:
        """Get ready models matching a task type and optional domain."""
        return [
            m
            for m in self._models.values()
            if m.status == "ready"
            and task_type in m.task_types
            and (domain is None or m.domain is None or m.domain == domain)
        ]
