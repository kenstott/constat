# Deployment Script Generator

## Goal

Given a source config directory and a target config directory, produce a deployment script that makes target equivalent to source. Scoped to system (root) and its subdomains only — user and session configs are excluded.

## Scope

**In scope** (system + domain tiers):
- Root `config.yaml` (LLM, execution, storage, email, server, context_preload, system_prompt, ner_stop_list)
- `domains/*/config.yaml` (metadata, sources, glossary, relationships, rights, facts, entity_resolution, task_routing, golden_questions, composition)
- `domains/*/learnings.yaml` (rules, corrections, archive)
- `permissions.yaml` (user permissions, personas, resource access)
- Skills (`skills/*/SKILL.md` + supporting files)
- Agents (system-level agent definitions)

**Out of scope**:
- `.constat/<user_id>/` (user configs, user domains, session state)
- Session `.duckdb` files
- Vector store data (rebuilt from sources on deployment)
- `.env` files and secrets (managed separately)

## Diff Model

### ConfigDiff

```python
@dataclass
class ConfigDiff:
    source_path: str
    target_path: str
    generated_at: str                    # ISO timestamp
    sections: list[SectionDiff]
    summary: DiffSummary

@dataclass
class SectionDiff:
    section: str                         # "root", "domain:sales-analytics", "permissions", "skills:my-skill"
    changes: list[Change]

@dataclass
class Change:
    path: str                            # dot-delimited key path, e.g. "databases.sales.uri"
    kind: Literal["added", "removed", "modified"]
    source_value: Any | None             # value in source (None if removed)
    target_value: Any | None             # value in target (None if added)
    sensitive: bool                      # True if path contains credentials/secrets
    category: str                        # "source", "glossary", "rule", "permission", "skill", "agent", "config", "test"

@dataclass
class DiffSummary:
    total_changes: int
    added: int
    removed: int
    modified: int
    by_category: dict[str, int]          # changes per category
    domains_added: list[str]
    domains_removed: list[str]
    sensitive_changes: int               # count of credential/secret changes
```

### Sensitive Path Detection

Paths containing these keys are flagged `sensitive: true` and values are masked in output:

```python
SENSITIVE_KEYS = {
    "uri", "password", "api_key", "auth_token", "api_token",
    "oauth2_client_secret", "smtp_password", "key", "secret",
    "aws_secret_access_key", "aws_session_token", "admin_token",
    "password_hash", "firebase_api_key", "notion_token",
}
```

Sensitive values are shown as `"***"` in the diff output. The deployment script uses `${ENV_VAR}` references, never literal secrets.

## Diff Algorithm

### Phase 1: Load

```python
source = load_config(source_path)    # root + domains + permissions + skills + agents + learnings
target = load_config(target_path)
```

Loading resolves `$ref` includes and `${ENV_VAR}` references (recording the reference, not the resolved value).

### Phase 2: Structural Diff

Per section, deep-compare source and target:

```
diff_root(source.config, target.config)
    → for each top-level key (databases, apis, documents, glossary, ...):
        deep_diff(source_value, target_value, path=[key])

for domain in union(source.domains, target.domains):
    if domain only in source → all keys are "added"
    if domain only in target → all keys are "removed"
    if domain in both → deep_diff(source.domain, target.domain)

diff_permissions(source.permissions, target.permissions)
diff_skills(source.skills, target.skills)
diff_agents(source.agents, target.agents)
diff_learnings(source.learnings, target.learnings)
```

Deep diff rules:
- **Dict**: recurse keys. Added/removed keys are leaf changes. Modified keys recurse.
- **List**: compare as sets where items have identity (e.g., entity_resolution by entity_type, golden_questions by question text). Otherwise compare by index.
- **Scalar**: direct equality check.
- **Null in source**: marks the key for deletion in target.

### Phase 3: Categorize

Each change is tagged with a category:

| Path pattern | Category |
|---|---|
| `databases.*`, `apis.*`, `documents.*` | `source` |
| `glossary.*` | `glossary` |
| `relationships.*` | `relationship` |
| `learnings.*`, `rights.*`, `facts.*` | `rule` |
| `permissions.*` | `permission` |
| `skills.*` | `skill` |
| `agents.*` | `agent` |
| `golden_questions.*` | `test` |
| Everything else | `config` |

## Deployment Script

The diff produces a deployment script — a YAML manifest of operations to apply:

```yaml
# deploy-2026-03-25T14-30-00.yaml
# Source: /path/to/staging/config
# Target: /path/to/production/config
# Generated: 2026-03-25T14:30:00Z

operations:
  # --- Root config changes ---
  - op: set
    file: config.yaml
    path: llm.model
    value: "claude-sonnet-4-20250514"

  - op: set
    file: config.yaml
    path: databases.inventory.description
    value: "Product inventory and warehouse data"

  - op: delete
    file: config.yaml
    path: databases.legacy_crm

  # --- Domain changes ---
  - op: create_domain
    domain: logistics
    source_dir: domains/logistics/

  - op: set
    file: domains/sales-analytics/config.yaml
    path: glossary.ARR
    value:
      definition: "Annual Recurring Revenue — sum of active subscription values normalized to 12 months"
      aliases: ["annual recurring revenue"]
      category: "finance"

  - op: set
    file: domains/sales-analytics/learnings.yaml
    path: rules.join_on_id
    value:
      text: "Always join orders with regions on region_id, not region_name"
      scope: "domain"
      confidence: 0.92

  - op: delete
    file: domains/hr-reporting/config.yaml
    path: glossary.deprecated_metric

  # --- Permissions changes ---
  - op: set
    file: permissions.yaml
    path: users.jane@company.com.persona
    value: "domain_builder"

  - op: set
    file: permissions.yaml
    path: users.jane@company.com.domains
    value: ["sales-analytics", "logistics"]

  # --- Skills ---
  - op: copy_skill
    skill: quarterly-report
    source_dir: skills/quarterly-report/

  - op: delete_skill
    skill: deprecated-export

  # --- Golden questions ---
  - op: set
    file: domains/sales-analytics/config.yaml
    path: golden_questions[question="What is total revenue by region?"]
    value:
      question: "What is total revenue by region?"
      tags: ["smoke", "e2e"]
      expect:
        terms: { revenue: true, region: true }
        end_to_end:
          answer_contains: ["North America", "Europe"]

  # --- Sensitive (env var references only) ---
  - op: set
    file: config.yaml
    path: databases.production_db.uri
    value: "${PRODUCTION_DB_URI}"
    sensitive: true
```

### Operation Types

| Op | Description |
|---|---|
| `set` | Create or update a value at a YAML path |
| `delete` | Remove a key from a YAML file |
| `create_domain` | Copy an entire domain directory from source |
| `delete_domain` | Remove an entire domain directory |
| `copy_skill` | Copy a skill directory from source |
| `delete_skill` | Remove a skill directory |
| `copy_file` | Copy a non-YAML file (e.g., skill scripts, assets) |
| `delete_file` | Remove a non-YAML file |

## CLI

```bash
# Generate diff (review only)
constat deploy diff --source /path/to/staging --target /path/to/production

# Generate deployment script
constat deploy generate --source /path/to/staging --target /path/to/production -o deploy.yaml

# Apply deployment script (with confirmation)
constat deploy apply deploy.yaml --target /path/to/production

# Dry run (show what would change, don't write)
constat deploy apply deploy.yaml --target /path/to/production --dry-run

# Apply specific categories only
constat deploy apply deploy.yaml --target /path/to/production --only glossary,rules

# Exclude categories
constat deploy apply deploy.yaml --target /path/to/production --exclude permissions
```

### Diff Output (human-readable)

```
=== Deployment Diff: staging → production ===

Root Config:
  ~ llm.model: "claude-haiku-3-5-20241022" → "claude-sonnet-4-20250514"
  + databases.inventory: { type: sql, uri: ${INVENTORY_DB_URI}, ... }
  - databases.legacy_crm

Domain: sales-analytics
  + glossary.ARR: "Annual Recurring Revenue — ..."
  ~ glossary.MRR.definition: "Monthly revenue" → "Monthly Recurring Revenue — sum of..."
  + learnings.rules.join_on_id: "Always join orders with regions on region_id"
  + golden_questions: "What is total revenue by region?"

Domain: logistics (NEW)
  + [entire domain — 2 databases, 5 glossary terms, 3 rules]

Permissions:
  + users.jane@company.com: { persona: domain_builder, domains: [sales-analytics, logistics] }
  ~ users.bob@company.com.persona: "viewer" → "domain_user"

Skills:
  + quarterly-report
  - deprecated-export

Summary: 23 changes (12 added, 3 removed, 8 modified) | 2 sensitive | 1 new domain
```

## Apply Engine

```python
class DeployApplier:
    def __init__(self, target_path: str, dry_run: bool = False):
        self.target_path = target_path
        self.dry_run = dry_run
        self.applied: list[str] = []
        self.skipped: list[str] = []
        self.errors: list[str] = []

    def apply(self, script: DeployScript, only: set[str] | None = None, exclude: set[str] | None = None):
        """Apply operations from the deployment script."""
        # 1. Backup target config directory
        # 2. Filter operations by category (only/exclude)
        # 3. Apply operations in order:
        #    - create_domain / delete_domain first (directory ops)
        #    - copy_skill / delete_skill (directory ops)
        #    - set / delete (YAML mutations, grouped by file)
        # 4. Validate target config loads without errors
        # 5. Report results

    def _apply_set(self, op: Operation):
        """Set a value at a YAML path in the target file."""
        # Load YAML, navigate to parent path, set value, write back
        # Preserves comments and formatting (ruamel.yaml)

    def _apply_delete(self, op: Operation):
        """Remove a key at a YAML path in the target file."""

    def _backup(self):
        """Create timestamped backup of target config directory."""
        # target_path/.backups/2026-03-25T14-30-00/
```

### Safety

- **Backup before apply** — Timestamped copy of target config before any mutation.
- **Validation after apply** — Load the modified target config with `load_config()`. If it fails validation, roll back from backup.
- **Sensitive masking** — Deployment scripts never contain literal secrets. Only `${ENV_VAR}` references.
- **Dry run default** — First run always dry-run unless `--no-dry-run` is explicit.
- **Category filtering** — Apply subsets of changes (e.g., only glossary, exclude permissions).

## API (Optional)

For web-based deployment management:

```
POST /admin/deploy/diff
    body: { source_path: str, target_path: str }
    returns: ConfigDiff

POST /admin/deploy/generate
    body: { source_path: str, target_path: str }
    returns: DeployScript (YAML)

POST /admin/deploy/apply
    body: { script: DeployScript, target_path: str, dry_run: bool, only: list[str], exclude: list[str] }
    returns: ApplyResult
```

Requires `platform_admin` persona.

## File Changes

| File | Change |
|---|---|
| `constat/deploy/__init__.py` | New — package init |
| `constat/deploy/differ.py` | New — config diff engine (load, deep_diff, categorize) |
| `constat/deploy/script.py` | New — deployment script model (Operation, DeployScript) |
| `constat/deploy/applier.py` | New — apply engine (backup, mutate, validate, rollback) |
| `constat/deploy/cli.py` | New — CLI commands (diff, generate, apply) |
| `constat/deploy/sensitive.py` | New — sensitive path detection and masking |
| `constat/server/routes/admin.py` | Add deploy API routes (optional) |
| `tests/test_deploy_diff.py` | New — diff engine tests |
| `tests/test_deploy_apply.py` | New — apply engine tests |

## Testing

- **Diff accuracy**: Two known configs → verify every change detected, no false positives
- **Sensitive masking**: Verify credentials never appear in diff output or deployment scripts
- **Idempotency**: Apply script twice → no additional changes on second run
- **Rollback**: Apply script → corrupt validation → verify backup restored
- **Category filtering**: Apply with `--only glossary` → verify only glossary changes applied
- **New domain**: Source has domain not in target → verify `create_domain` op and full directory copy
- **Deleted domain**: Target has domain not in source → verify `delete_domain` op
- **Null deletion**: Source sets key to null → verify key removed from target
- **$ref resolution**: Configs with `$ref` includes → verify diff sees through includes
- **Round-trip**: Generate script from diff → apply → re-diff → zero changes
