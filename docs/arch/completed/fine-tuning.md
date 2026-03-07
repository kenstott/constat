# Fine-Tuning Closed Loop

> **Status:** Implemented. Registry, provider clients, manager, API endpoints, router integration, background polling, and UI tab all functional.

### Implementation Notes

- YAML-backed model registry at `~/.constat/fine_tune_registry.yaml`
- Training artifacts saved to `.constat/{user_id}/fine_tune/{name}/` (git-trackable)
- OpenAI and Together AI provider clients
- Background polling (60s) for training status transitions
- Fine-Tune tab in Learnings accordion with job list, status badges, new job form
- `recreate_from_artifact()` reproduces fine-tune jobs from saved training data

## Problem

Constat captures domain-specific learnings (corrections, rules, glossary) as users work. The `SimpleExporter` exports these as JSONL in standard fine-tuning formats. But there's no way to close the loop: the exported exemplars sit on disk, the user must manually upload to a provider, fine-tune, then manually configure the model in their routing config.

Meanwhile, every query pays full cost for Claude/Sonnet on routine domain-specific tasks (SQL generation, codegen) that a smaller, fine-tuned model could handle after learning the domain's vocabulary, join patterns, and naming conventions.

## Core Idea

Specialist models. Fine-tune small models (gpt-4o-mini, Llama-8B via Together) on domain-specific exemplars for SQL/codegen tasks. Keep Claude for planning/reasoning. Fine-tuned models get prepended to the TaskRouter escalation chain — they're tried first, with Claude as automatic fallback.

```
User corrects mistake
       │
       ▼
  Learning saved ──► Rule promoted ──► Exemplars exported
                                              │
                                              ▼
                                    Upload to provider (OpenAI/Together)
                                              │
                                              ▼
                                    Fine-tune job submitted
                                              │
                                              ▼
                                    Poll until ready
                                              │
                                              ▼
                                    Inject into TaskRouter chain
                                              │
                                              ▼
                              Next SQL query uses fine-tuned model first
                              (escalates to Claude on failure)
```

## Training Artifacts

Training data is saved to disk before upload — the "source code" of the fine-tuned model:

```
.constat/{user_id}/fine_tune/{name}/
    training_data.jsonl   ← the exemplars (git-trackable)
    manifest.yaml         ← reproducibility metadata
```

**manifest.yaml:**
```yaml
id: 550e8400-e29b-41d4-a716-446655440000
name: sales-sql-v1
provider: together
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct-Reference
task_types:
  - sql_generation
domain: sales-analytics
include:
  - corrections
  - rules
min_confidence: 0.6
hyperparams: null
exemplar_count: 87
created: '2026-03-07T12:00:00+00:00'
```

This enables:
- **Version control** — commit training artifacts alongside domain configs
- **Reproducibility** — `recreate_from_artifact()` rebuilds the model from saved data
- **Auditability** — diff training data across versions to see what changed

## Data Model

```python
@dataclass
class FineTunedModel:
    id: str                        # UUID
    name: str                      # User-facing ("sales-sql-v3")
    provider: str                  # "openai" | "together"
    base_model: str                # "gpt-4o-mini-2024-07-18"
    fine_tuned_model_id: str       # Provider's model ID (ft:gpt-4o-mini:org:...)
    task_types: list[str]          # ["sql_generation"]
    domain: str | None             # Domain filter (None = cross-domain)
    status: str                    # "training" | "ready" | "failed" | "archived"
    provider_job_id: str           # For polling
    created: str                   # ISO timestamp
    training_file_id: str          # Provider's uploaded file ID
    training_data_path: str | None # Local path to saved artifact directory
    metrics: dict | None           # Training loss, etc.
    exemplar_count: int            # Number of training examples
```

## Architecture

### Registry (`fine_tune_registry.py`)

YAML-backed storage at `~/.constat/fine_tune_registry.yaml`. Tracks all fine-tuned models across sessions.

| Method | Description |
|--------|-------------|
| `add(model)` | Register a new model |
| `get(id)` | Get by ID |
| `list(status?, domain?)` | Filter models |
| `update(id, **kwargs)` | Update fields |
| `remove(id)` | Delete entry |
| `get_active_for_task(task_type, domain?)` | Get ready models for routing |

### Provider Clients (`fine_tune_providers.py`)

Abstract `FineTuneProviderClient` with concrete implementations:

| Provider | Class | Base Models | Fine-Tune Cost |
|----------|-------|-------------|----------------|
| OpenAI | `OpenAIFineTuneClient` | gpt-4o-mini, gpt-4o, gpt-4.1-mini/nano | ~$3/1M tokens |
| Together | `TogetherFineTuneClient` | Llama-3.1-8B/70B, Mistral-7B, Qwen-7B | ~$0.48/1M tokens |

API keys sourced from environment: `OPENAI_API_KEY`, `TOGETHER_API_KEY`.

`get_available_providers()` returns only providers with valid keys set.

### Manager (`fine_tune_manager.py`)

Orchestrates the full lifecycle:

| Method | Description |
|--------|-------------|
| `start_fine_tune(...)` | Export → save artifact → upload → submit → register |
| `check_status(id)` | Poll provider, transition training→ready/failed |
| `check_all_training()` | Poll all training jobs (called by background task) |
| `cancel(id)` | Cancel running job |
| `archive(id)` | Mark as archived |
| `delete(id)` | Archive + delete from provider |
| `recreate_from_artifact(dir)` | Rebuild from saved training data |

### Router Integration

```python
# TaskRoutingConfig
def prepend_model(self, task_type: str, spec: ModelSpec) -> None

# TaskRouter
def set_domain_models(self, models: list[FineTunedModel]) -> None
```

`set_domain_models()` iterates ready models, creates `ModelSpec` for each, and prepends to the routing chain for matching task types. The escalation pattern handles mismatches — if the fine-tuned model fails, it naturally falls through to the next model (e.g., Claude).

## API Endpoints

All under `/api/fine-tune/`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/fine-tune/jobs` | Start a fine-tuning job |
| `GET` | `/fine-tune/jobs` | List all jobs (filter by status, domain) |
| `GET` | `/fine-tune/jobs/{id}` | Get job + poll status if training |
| `POST` | `/fine-tune/jobs/{id}/cancel` | Cancel training job |
| `DELETE` | `/fine-tune/jobs/{id}` | Delete job + remove from provider |
| `POST` | `/fine-tune/recreate` | Recreate job from saved artifact |
| `GET` | `/fine-tune/providers` | List available providers (based on env keys) |

**Start request:**
```json
{
  "name": "sales-sql-v1",
  "provider": "together",
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
  "task_types": ["sql_generation"],
  "domain": "sales-analytics",
  "include": ["corrections", "rules"],
  "min_confidence": 0.6,
  "hyperparams": {"n_epochs": 3}
}
```

**Response:**
```json
{
  "id": "550e8400-...",
  "name": "sales-sql-v1",
  "provider": "together",
  "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
  "status": "training",
  "exemplar_count": 87,
  "training_data_path": ".constat/default/fine_tune/sales-sql-v1",
  ...
}
```

## Background Polling

A 60-second polling loop runs in the server lifespan:

```python
async def _fine_tune_poll_loop():
    while True:
        await asyncio.sleep(60)
        updated = await asyncio.to_thread(ft_manager.check_all_training)
        for model in updated:
            if model.status in ("ready", "failed"):
                logger.info(f"Fine-tune {model.name}: {model.status}")
```

Transitions `training→ready` or `training→failed` based on provider status.

## Frontend

### Fine-Tune Tab

Added to the Learnings accordion in `ArtifactPanel.tsx` (alongside Rules, Pending, Export).

**Job list:** Table showing name, provider/model, status badge, exemplar count. Status badges:
- Training: yellow, pulsing
- Ready: green
- Failed: red
- Archived: gray

**Row actions:** Cancel (if training), Delete.

**New job form:** Name, provider dropdown (auto-populated from `/fine-tune/providers`), base model dropdown (changes with provider), task types checkboxes, training data inclusion, min confidence slider, "Start Training" button.

**Auto-refresh:** Polls `/fine-tune/jobs` every 30s while any job has `status='training'`.

## Domain Scoping

Fine-tuned models are domain artifacts. The `domain` field scopes which learnings train the model and which routing chains it's injected into:

- `domain = "sales-analytics"` → trained on sales learnings, injected only for sales sessions
- `domain = None` → trained on all learnings, injected for all sessions

This fits the domain composition model: glossary, rules, data sources, and now fine-tuned codegen models all compose at the domain level.

## Intended Workflow

1. Run queries with Claude (planning) + cheap model (SQL/codegen)
2. Correct mistakes as you work — learnings accumulate
3. Compactor promotes learnings to rules (50+ threshold)
4. Once rules are dense enough, start a fine-tune job from the UI
5. Training data saved to disk, uploaded to provider, job submitted
6. Background poller transitions to "ready" when done
7. Fine-tuned model prepended to routing chain for matching task types
8. Subsequent queries try the fine-tuned model first, escalate to Claude on failure
9. Commit training artifacts to git for reproducibility

## Files

### New
| File | Description |
|------|-------------|
| `constat/learning/fine_tune_registry.py` | YAML-backed model registry |
| `constat/learning/fine_tune_providers.py` | OpenAI + Together provider clients |
| `constat/learning/fine_tune_manager.py` | Lifecycle orchestration |
| `constat/server/routes/fine_tune.py` | REST endpoints |

### Modified
| File | Change |
|------|--------|
| `constat/core/config.py` | `TaskRoutingConfig.prepend_model()` |
| `constat/providers/router.py` | `TaskRouter.set_domain_models()` |
| `constat/server/app.py` | Manager init, route registration, poll loop |
| `constat/learning/__init__.py` | Exports |
| `constat-ui/src/types/api.ts` | `FineTuneJob`, `FineTuneProvider` types |
| `constat-ui/src/api/sessions.ts` | 6 API client functions |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | Fine-Tune tab |

## Design Decisions

**Why YAML registry, not database?** The registry is small (dozens of entries), rarely written, and should be human-readable. YAML is inspectable, diffable, and doesn't require schema management.

**Why save training artifacts to disk?** Fine-tuned model weights are ephemeral (hosted by provider, can be deleted). The training data is the reproducible artifact — like source code vs. compiled binary. Saving before upload ensures the exemplars are never lost.

**Why prepend to routing chain (not replace)?** The escalation pattern is the safety net. A fine-tuned model may not cover all cases — prepending means it's tried first, but Claude is always available as fallback. No regression risk.

**Why Together AI for open models?** Fine-tuning requires significant GPU compute. Together provides the training infrastructure for open models (Llama, Mistral) without requiring local GPU hardware. The resulting model can later be downloaded and self-hosted via Ollama.

**Why domain-scoped?** A SQL model fine-tuned on sales vocabulary would hurt on HR queries. Domain scoping ensures training data and routing injection match the context where the model will be used.
