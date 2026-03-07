# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fine-tune lifecycle manager: export -> upload -> submit -> poll -> register."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from constat.learning.fine_tune_providers import (
    FineTuneProviderClient,
    create_provider_client,
)
from constat.learning.fine_tune_registry import FineTuneRegistry, FineTunedModel
from constat.learning.simple_exporter import SimpleExporter
from constat.storage.learnings import LearningStore

logger = logging.getLogger(__name__)


class FineTuneManager:
    """Orchestrates fine-tune lifecycle."""

    def __init__(
        self,
        registry: FineTuneRegistry,
        learning_store: LearningStore,
        vector_store=None,
    ):
        self._registry = registry
        self._exporter = SimpleExporter(learning_store, vector_store)
        self._clients: dict[str, FineTuneProviderClient] = {}

    @property
    def registry(self) -> FineTuneRegistry:
        return self._registry

    def register_provider(self, name: str, client: FineTuneProviderClient) -> None:
        self._clients[name] = client

    def _get_client(self, provider: str) -> FineTuneProviderClient:
        if provider not in self._clients:
            self._clients[provider] = create_provider_client(provider)
        return self._clients[provider]

    def _save_training_artifact(
        self,
        model_id: str,
        name: str,
        jsonl_content: str,
        provider: str,
        base_model: str,
        task_types: list[str],
        domain: str | None,
        include: list[str],
        min_confidence: float,
        hyperparams: dict | None,
        exemplar_count: int,
    ) -> Path:
        """Save training JSONL and manifest to a versioned directory.

        Structure:
            .constat/{user_id}/fine_tune/{name}/
                training_data.jsonl   — the exemplars (git-trackable)
                manifest.yaml         — reproducibility metadata
        """
        user_id = self._exporter.store.user_id
        artifact_dir = Path(".constat") / user_id / "fine_tune" / name
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Save training data
        (artifact_dir / "training_data.jsonl").write_text(jsonl_content, encoding="utf-8")

        # Save manifest
        manifest = {
            "id": model_id,
            "name": name,
            "provider": provider,
            "base_model": base_model,
            "task_types": task_types,
            "domain": domain,
            "include": include,
            "min_confidence": min_confidence,
            "hyperparams": hyperparams,
            "exemplar_count": exemplar_count,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        (artifact_dir / "manifest.yaml").write_text(
            yaml.dump(manifest, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        logger.info(f"Saved training artifact to {artifact_dir}")
        return artifact_dir

    def start_fine_tune(
        self,
        name: str,
        provider: str,
        base_model: str,
        task_types: list[str],
        domain: str | None = None,
        include: list[str] | None = None,
        min_confidence: float = 0.0,
        hyperparams: dict | None = None,
    ) -> FineTunedModel:
        """Export -> save artifact -> upload -> submit -> register as 'training'."""
        include = include or ["corrections", "rules"]
        client = self._get_client(provider)

        # Export training data
        jsonl_content = self._exporter.export(
            include=include,
            fmt="messages",
            domain=domain,
            min_confidence=min_confidence,
        )

        lines = [l for l in jsonl_content.strip().split("\n") if l.strip()]
        exemplar_count = len(lines)
        if exemplar_count == 0:
            raise ValueError("No training examples found for the given filters")

        model_id = str(uuid.uuid4())

        # Save training artifact to disk (git-trackable)
        artifact_dir = self._save_training_artifact(
            model_id=model_id,
            name=name,
            jsonl_content=jsonl_content,
            provider=provider,
            base_model=base_model,
            task_types=task_types,
            domain=domain,
            include=include,
            min_confidence=min_confidence,
            hyperparams=hyperparams,
            exemplar_count=exemplar_count,
        )

        # Upload to provider
        filename = f"constat_{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        training_file_id = client.upload_training_file(jsonl_content, filename)

        # Submit job
        job_id = client.create_job(training_file_id, base_model, hyperparams)

        # Register
        model = FineTunedModel(
            id=model_id,
            name=name,
            provider=provider,
            base_model=base_model,
            fine_tuned_model_id="",
            task_types=task_types,
            domain=domain,
            status="training",
            provider_job_id=job_id,
            created=datetime.now(timezone.utc).isoformat(),
            training_file_id=training_file_id,
            training_data_path=str(artifact_dir),
            metrics=None,
            exemplar_count=exemplar_count,
        )
        self._registry.add(model)
        logger.info(f"Started fine-tune job {name}: provider={provider}, base={base_model}, exemplars={exemplar_count}")
        return model

    def recreate_from_artifact(self, artifact_dir: str | Path) -> FineTunedModel:
        """Recreate a fine-tune job from a saved training artifact.

        Reads manifest.yaml and training_data.jsonl from the artifact directory,
        uploads the training data, and submits a new fine-tuning job.
        """
        artifact_path = Path(artifact_dir)
        manifest_file = artifact_path / "manifest.yaml"
        training_file = artifact_path / "training_data.jsonl"

        if not manifest_file.exists():
            raise FileNotFoundError(f"No manifest.yaml in {artifact_path}")
        if not training_file.exists():
            raise FileNotFoundError(f"No training_data.jsonl in {artifact_path}")

        manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
        jsonl_content = training_file.read_text(encoding="utf-8")

        client = self._get_client(manifest["provider"])

        # Upload
        filename = f"constat_{manifest['name']}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        training_file_id = client.upload_training_file(jsonl_content, filename)

        # Submit
        job_id = client.create_job(
            training_file_id, manifest["base_model"], manifest.get("hyperparams")
        )

        model = FineTunedModel(
            id=str(uuid.uuid4()),
            name=manifest["name"],
            provider=manifest["provider"],
            base_model=manifest["base_model"],
            fine_tuned_model_id="",
            task_types=manifest["task_types"],
            domain=manifest.get("domain"),
            status="training",
            provider_job_id=job_id,
            created=datetime.now(timezone.utc).isoformat(),
            training_file_id=training_file_id,
            training_data_path=str(artifact_path),
            metrics=None,
            exemplar_count=manifest["exemplar_count"],
        )
        self._registry.add(model)
        logger.info(f"Recreated fine-tune job {model.name} from {artifact_path}")
        return model

    def check_status(self, model_id: str) -> FineTunedModel:
        """Poll provider, update registry."""
        model = self._registry.get(model_id)
        if not model:
            raise KeyError(f"Model {model_id} not found")

        if model.status != "training":
            return model

        client = self._get_client(model.provider)
        result = client.get_job_status(model.provider_job_id)

        updates: dict = {}
        if result["status"] == "succeeded":
            updates["status"] = "ready"
            updates["fine_tuned_model_id"] = result["fine_tuned_model"]
            if result.get("metrics"):
                updates["metrics"] = result["metrics"]
            logger.info(f"Fine-tune {model.name} ready: {result['fine_tuned_model']}")
        elif result["status"] == "failed":
            updates["status"] = "failed"
            if result.get("metrics"):
                updates["metrics"] = result["metrics"]
            logger.warning(f"Fine-tune {model.name} failed")
        else:
            # Still running — update metrics if available
            if result.get("metrics"):
                updates["metrics"] = result["metrics"]

        if updates:
            self._registry.update(model_id, **updates)

        return self._registry.get(model_id)  # type: ignore[return-value]

    def check_all_training(self) -> list[FineTunedModel]:
        """Check all models with status='training'. Returns updated models."""
        training = self._registry.list(status="training")
        updated = []
        for model in training:
            try:
                result = self.check_status(model.id)
                if result.status != "training":
                    updated.append(result)
            except Exception as e:
                logger.error(f"Error checking fine-tune {model.name}: {e}")
        return updated

    def cancel(self, model_id: str) -> None:
        model = self._registry.get(model_id)
        if not model:
            raise KeyError(f"Model {model_id} not found")
        if model.status != "training":
            raise ValueError(f"Cannot cancel model with status '{model.status}'")

        client = self._get_client(model.provider)
        client.cancel_job(model.provider_job_id)
        self._registry.update(model_id, status="failed")
        logger.info(f"Cancelled fine-tune job {model.name}")

    def archive(self, model_id: str) -> None:
        model = self._registry.get(model_id)
        if not model:
            raise KeyError(f"Model {model_id} not found")
        self._registry.update(model_id, status="archived")
        logger.info(f"Archived fine-tune model {model.name}")

    def delete(self, model_id: str) -> None:
        """Archive and delete from provider."""
        model = self._registry.get(model_id)
        if not model:
            raise KeyError(f"Model {model_id} not found")

        # Delete from provider if model was ready
        if model.status == "ready" and model.fine_tuned_model_id:
            try:
                client = self._get_client(model.provider)
                client.delete_model(model.fine_tuned_model_id)
            except Exception as e:
                logger.warning(f"Failed to delete model from provider: {e}")

        self._registry.remove(model_id)
        logger.info(f"Deleted fine-tune model {model.name}")
