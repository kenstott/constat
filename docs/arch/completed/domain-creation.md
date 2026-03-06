# Domain Creation & Composition

> **Status:** Partially implemented. Domain tiers, domain panel, move resources, and promote are done. `includes` composition is not yet implemented.

## Problem

Domains are created either by hand-editing YAML or through a minimal "New Domain" form that produces an empty shell. There's no way to save a session's active domain selection as a reusable composition.

## Current State

### Domain Storage

Three tiers, each a directory of YAML files:

```
{config_dir}/domains/           → system (shipped, immutable via UI)
.constat/shared/domains/        → shared (promoted from user, visible to all)
.constat/{user_id}/domains/     → user (private, promotable)
```

### What a Domain Owns

| Content | Storage |
|---------|---------|
| Databases, APIs, Documents | Domain YAML |
| Glossary terms | Vector store (`glossary_term.domain`) |
| Skills | `{domain_dir}/skills/` |
| Agents | `{domain_dir}/agents.yaml` |
| Rules | Learning store (`rule.domain`) |
| Permissions | `{domain_dir}/permissions.yaml` |
| System prompt, NER stop list | Domain YAML |

### Existing Creation Paths

1. **POST /domains** — creates empty YAML in user tier
2. **YAML editor** — edit domain content directly (three-dot menu → Edit YAML)
3. **Promote** — copies user domain to shared tier
4. **Drag-and-drop** — move resources between domains

## Design

### New Concept: `includes`

A domain can reference other domains via `includes`. When activated, included domains are activated transitively. Resources stay owned by their original domains.

```yaml
name: Analytics Bundle
description: Sales + HR for quarterly reviews

includes:
  - sales-analytics
  - hr-reporting
```

`includes` is not `$ref`. It does not merge or copy resources. It's a selection preset — activating one domain activates several.

| | `$ref` | `includes` |
|---|--------|------------|
| **Merges content** | Yes — flattens into parent | No — activates separately |
| **Resource ownership** | Parent owns merged copy | Original domains retain ownership |
| **Use case** | Config factoring | Session composition presets |

A composed domain can layer its own resources on top:

```yaml
name: Analytics Bundle
includes:
  - sales-analytics
  - hr-reporting

documents:
  quarterly_template:
    path: templates/quarterly_review.md

glossary:
  cross-sell ratio:
    definition: Revenue from cross-department referrals divided by total revenue

system_prompt: |
  Focus on cross-department metrics and quarterly trends.
```

Further customization is done through the YAML editor.

### Compose from Active Domains

"Save as Domain" takes the current checkbox selection and creates a composed domain pre-populated with `includes`.

```
POST /domains/compose
{
  "name": "Analytics Bundle",
  "description": "Sales + HR for quarterly reviews",
  "include_domains": ["sales-analytics", "hr-reporting"]
}
```

The endpoint writes a YAML file to `.constat/{user_id}/domains/` and registers it in `config.domains`. Everything beyond the initial composition — adding resources, glossary, skills, system prompt — is done through the YAML editor or existing resource management UI.

### Activation

`_load_domains_into_session` resolves `includes` transitively:

```python
def _resolve_includes(filename: str, config: Config, seen: set[str]) -> list[str]:
    """Resolve transitive includes, detecting cycles."""
    if filename in seen:
        return []
    seen.add(filename)
    domain = config.load_domain(filename)
    if not domain:
        return []
    result = []
    for inc in domain.includes:
        result.extend(_resolve_includes(inc, config, seen))
    result.append(filename)
    return result
```

When `set_active_domains(["analytics-bundle"])` is called:
1. Resolve includes: `["sales-analytics", "hr-reporting", "analytics-bundle"]`
2. Load all resolved domains (conflict detection applies across the full set)
3. Store the user's explicit selection in preferences
4. Store the expanded set as `managed.active_domains`

**Circular inclusion:** Rejected at activation time.

**Missing includes:** Warn and proceed with available domains.

### Promotion Guards

Composed domains promote like any other. If includes reference user-tier domains, promotion fails with a clear error listing which includes need promotion first.

### Frontend

- **"Save as Domain" button** — appears when 2+ domains are checked. Name + description prompt, creates composed domain.
- **Includes badge** — composed domains show included domain count in the tree.
- **YAML editor** — existing editor is the primary path for all further customization.

## Implementation Plan

### Phase 1: `includes` Support

1. Add `includes: list[str]` to `DomainConfig`
2. Update `_load_domains_into_session` to resolve includes transitively
3. Cycle detection in resolution

### Phase 2: Compose Endpoint + UI

4. `POST /domains/compose` — validates includes, writes YAML, registers in config
5. "Save as Domain" button in domain panel
6. Promotion guard for included domain tier validation