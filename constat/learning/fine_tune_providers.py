# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fine-tuning provider clients for OpenAI and Together AI."""

from __future__ import annotations

import io
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FineTuneProviderClient(ABC):
    """Abstract interface for fine-tuning provider APIs."""

    @abstractmethod
    def upload_training_file(self, jsonl_content: str, filename: str) -> str:
        """Upload JSONL training data. Returns file_id."""

    @abstractmethod
    def create_job(
        self,
        training_file_id: str,
        base_model: str,
        hyperparams: dict | None = None,
    ) -> str:
        """Start fine-tuning job. Returns job_id."""

    @abstractmethod
    def get_job_status(self, job_id: str) -> dict:
        """Return {"status": "running"|"succeeded"|"failed", "fine_tuned_model": "...", "metrics": {...}}"""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""

    @abstractmethod
    def delete_model(self, model_id: str) -> None:
        """Delete a fine-tuned model from the provider."""


class OpenAIFineTuneClient(FineTuneProviderClient):
    """OpenAI fine-tuning via openai SDK."""

    MODELS = [
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
    ]

    def __init__(self, api_key: str | None = None):
        import openai

        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def upload_training_file(self, jsonl_content: str, filename: str) -> str:
        buf = io.BytesIO(jsonl_content.encode("utf-8"))
        buf.name = filename
        result = self._client.files.create(file=buf, purpose="fine-tune")
        return result.id

    def create_job(
        self,
        training_file_id: str,
        base_model: str,
        hyperparams: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "training_file": training_file_id,
            "model": base_model,
        }
        if hyperparams:
            hp = {}
            if "n_epochs" in hyperparams:
                hp["n_epochs"] = hyperparams["n_epochs"]
            if "learning_rate_multiplier" in hyperparams:
                hp["learning_rate_multiplier"] = hyperparams["learning_rate_multiplier"]
            if "batch_size" in hyperparams:
                hp["batch_size"] = hyperparams["batch_size"]
            if hp:
                kwargs["hyperparameters"] = hp
        job = self._client.fine_tuning.jobs.create(**kwargs)
        return job.id

    def get_job_status(self, job_id: str) -> dict:
        job = self._client.fine_tuning.jobs.retrieve(job_id)
        status_map = {
            "validating_files": "running",
            "queued": "running",
            "running": "running",
            "succeeded": "succeeded",
            "failed": "failed",
            "cancelled": "failed",
        }
        result: dict = {
            "status": status_map.get(job.status, job.status),
            "fine_tuned_model": job.fine_tuned_model,
            "metrics": None,
        }
        if job.result_files:
            try:
                # Attempt to read training metrics
                metrics_content = self._client.files.content(job.result_files[0])
                import json

                lines = metrics_content.text.strip().split("\n")
                if lines:
                    last_line = json.loads(lines[-1])
                    result["metrics"] = {
                        "training_loss": last_line.get("train_loss"),
                        "step": last_line.get("step"),
                    }
            except Exception:
                pass
        return result

    def cancel_job(self, job_id: str) -> None:
        self._client.fine_tuning.jobs.cancel(job_id)

    def delete_model(self, model_id: str) -> None:
        self._client.models.delete(model_id)


class TogetherFineTuneClient(FineTuneProviderClient):
    """Together AI fine-tuning via together SDK."""

    MODELS = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Reference",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    def __init__(self, api_key: str | None = None):
        import together

        self._client = together.Together(api_key=api_key or os.environ.get("TOGETHER_API_KEY"))

    def upload_training_file(self, jsonl_content: str, filename: str) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix="constat_ft_"
        ) as f:
            f.write(jsonl_content)
            tmp_path = f.name

        try:
            result = self._client.files.upload(file=tmp_path, purpose="fine-tune")
            return result.id
        finally:
            import os as _os

            _os.unlink(tmp_path)

    def create_job(
        self,
        training_file_id: str,
        base_model: str,
        hyperparams: dict | None = None,
    ) -> str:
        kwargs: dict = {
            "training_file": training_file_id,
            "model": base_model,
        }
        if hyperparams:
            if "n_epochs" in hyperparams:
                kwargs["n_epochs"] = hyperparams["n_epochs"]
            if "learning_rate_multiplier" in hyperparams:
                kwargs["learning_rate"] = hyperparams["learning_rate_multiplier"]
            if "batch_size" in hyperparams:
                kwargs["batch_size"] = hyperparams["batch_size"]
        job = self._client.fine_tuning.create(**kwargs)
        return job.id

    def get_job_status(self, job_id: str) -> dict:
        job = self._client.fine_tuning.retrieve(job_id)
        status_map = {
            "pending": "running",
            "queued": "running",
            "running": "running",
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "failed",
            "error": "failed",
        }
        status_str = getattr(job, "status", "running")
        result: dict = {
            "status": status_map.get(status_str, status_str),
            "fine_tuned_model": getattr(job, "output_name", None),
            "metrics": None,
        }
        events = getattr(job, "events", None)
        if events and len(events) > 0:
            last_event = events[-1]
            if hasattr(last_event, "metrics"):
                result["metrics"] = {
                    "training_loss": getattr(last_event.metrics, "train_loss", None),
                }
        return result

    def cancel_job(self, job_id: str) -> None:
        self._client.fine_tuning.cancel(job_id)

    def delete_model(self, model_id: str) -> None:
        self._client.models.delete(model_id)


def get_available_providers() -> list[dict]:
    """Return providers with valid API keys."""
    providers = []
    if os.environ.get("OPENAI_API_KEY"):
        providers.append({"name": "openai", "models": OpenAIFineTuneClient.MODELS})
    if os.environ.get("TOGETHER_API_KEY"):
        providers.append({"name": "together", "models": TogetherFineTuneClient.MODELS})
    return providers


def create_provider_client(provider: str, api_key: str | None = None) -> FineTuneProviderClient:
    """Create a provider client by name."""
    if provider == "openai":
        return OpenAIFineTuneClient(api_key=api_key)
    elif provider == "together":
        return TogetherFineTuneClient(api_key=api_key)
    else:
        raise ValueError(f"Unknown fine-tune provider: {provider}")
