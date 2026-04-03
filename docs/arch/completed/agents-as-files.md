# Externalize Agents as AGENT.md Files

## Goal

Migrate agents from `agents.yaml` (single YAML file) to per-agent directories with `AGENT.md` files (YAML frontmatter + markdown body), matching the SKILL.md pattern. This makes agent prompts editable in markdown, compatible with Claude Desktop/Claude Code, and version-controllable per-agent.

## Current State

Agents live in `{base_dir}/{user_id}/agents.yaml`:
```yaml
data-analyst:
  description: Focus on metrics, trends, and actionable insights
  prompt: |
    Prioritize quantitative analysis. Always include:
    - Key metrics with period-over-period comparisons
    ...
  model: claude-sonnet-4-20250514
  skills:
    - financial-analysis
    - sales-analysis
```

## Target State

Each agent becomes a directory with an `AGENT.md` file:
```
agents/
  data-analyst/
    AGENT.md
    references/          # optional: few-shot examples, reference docs
  executive/
    AGENT.md
```

### AGENT.md Format

```markdown
---
name: data-analyst
description: Focus on metrics, trends, and actionable insights
model: claude-sonnet-4-20250514
skills:
  - financial-analysis
  - sales-analysis
---

Prioritize quantitative analysis. Always include:
- Key metrics with period-over-period comparisons
- Statistical context (averages, medians, standard deviations)
- Trend analysis with directional language
...
```

The markdown body IS the agent prompt (same pattern as SKILL.md).

## Implementation

### AgentManager Rewrite

Mirror SkillManager's file-loading pattern:
- Constructor scans `agents/` directory for `AGENT.md` files
- Tiered loading: system < domain < user (user overrides)
- `create_agent()` creates `agents/{name}/AGENT.md`
- `update_agent()` rewrites the AGENT.md file
- `delete_agent()` removes the directory
- `get_agent_prompt()` returns the markdown body

### One-Time Migration

On first access: if `agents.yaml` exists and `agents/` does not, convert each entry to an `AGENT.md` directory. Rename `agents.yaml` to `agents.yaml.imported`.

### File Changes

| File | Change |
|------|--------|
| `constat/core/agents.py` | Rewrite AgentManager to load from `agents/{name}/AGENT.md` directories |
| `constat/core/agent_parser.py` | New — parse AGENT.md frontmatter + markdown body (reuse skill parser pattern) |
| `constat/server/routes/agents.py` | Update to match new AgentManager API |
| `constat/server/graphql/learning_resolvers.py` | Update agent mutations for new file-based API |
| Tests | Update agent creation/loading tests |

## Benefits

- Markdown body natural for long persona prompts
- Consistent with SKILL.md — one format to learn
- Agent directories can hold reference docs, few-shot examples
- Claude Desktop / Claude Code can consume AGENT.md directly
- Meaningful git diffs (vs YAML multiline string changes)
