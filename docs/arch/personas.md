# Platform Roles & Persona Architecture

> **Status:** Design. No implementation yet.

## Problem

Constat is a general-purpose data intelligence engine. As-is, every user sees everything: codegen, DAG, schema explorer, skill editor, source config. This works for builders but overwhelms business users who just want answers.

The same engine can serve multiple audiences — from data engineers building domain models to executives consuming a chat interface — if the UX adapts to the user's role. The engine doesn't change. The UI surface, write permissions, and feedback mechanisms do.

## Core Idea

Five platform roles. Each role defines what the user sees and what they can write. The engine underneath is identical for all roles.

```
Platform Admin → Domain Builder → SME → Domain User → Viewer
       │               │           │          │           │
   all domains    builds domain   curates   chats +     chats
   + users        model          knowledge  flags       only
```

### Role Summary

| Role | Who | Primary function |
|------|-----|-----------------|
| **Platform Admin** | Ops / central IT | Domain registry, user management, API keys, billing, system health |
| **Domain Builder** | Data engineer / analyst | Connect sources, run discovery, define entities, build skills/agents, compose domains |
| **SME** | Business subject matter expert | Curate glossary, manage learnings, validate query results, provide domain context to builders |
| **Domain User** | Business user | Chat with a configured domain, flag wrong answers, suggest glossary corrections |
| **Viewer** | Executive / external | Chat only, no feedback loop, pure consumption |

Each role down the list sees less, configures less, but the user base gets wider. Pyramid shape.

## Architecture

The role system is a UX layer. It does not affect the execution engine.

```
┌─────────────────────────────────┐
│          Role Config            │  ← What you see, what buttons exist
├─────────────────────────────────┤
│              UI                 │  ← Renders based on role
├─────────────────────────────────┤
│           Same API              │  ← Same routes, role checked server-side
├─────────────────────────────────┤
│         Same Engine             │  ← Doesn't know about roles
└─────────────────────────────────┘
```

Server-side enforcement is a thin middleware check before route handlers. The frontend reads the role config once at login and conditionally renders UI components.

## Role Definitions

Role definitions ship with the platform as a config file (`roles.yaml`). Slowly changing — edited during development, then effectively constant. Developer-facing, not user-facing.

```yaml
# roles.yaml
roles:
  platform_admin:
    description: "Manages domains, users, system config"
    visibility:
      codegen: true
      dag: true
      schema_explorer: true
      skill_editor: true
      agent_builder: true
      domain_composer: true
      glossary_manager: true
      entity_manager: true
      source_config: true
      learnings_manager: true
      query_review: true
      query_history: true
    writes:
      sources: true
      glossary: true
      entities: true
      skills: true
      agents: true
      facts: true
      learnings: true
      domains: true

  domain_builder:
    description: "Builds and configures domain models"
    visibility:
      codegen: true
      dag: true
      schema_explorer: true
      skill_editor: true
      agent_builder: true
      domain_composer: true
      glossary_manager: true
      entity_manager: true
      source_config: true
      learnings_manager: true
      query_review: true
      query_history: true
    writes:
      sources: true
      glossary: true
      entities: true
      skills: true
      agents: true
      facts: true
      learnings: true
      domains: true

  sme:
    description: "Domain SME — curates glossary, manages learnings, validates results"
    visibility:
      codegen: false
      dag: false
      schema_explorer: false
      skill_editor: false
      agent_builder: false
      domain_composer: false
      glossary_manager: true
      entity_manager: true
      source_config: false
      learnings_manager: true
      query_review: true
      query_history: true
    writes:
      sources: false
      glossary: true
      entities: true
      skills: false
      agents: false
      facts: true
      learnings: true
      domains: false
    feedback:
      flag_answers: true
      suggest_glossary: false    # Doesn't suggest — directly edits
      suggest_entities: true
      auto_approve: true         # SME edits go straight in

  domain_user:
    description: "Chats with a configured domain"
    visibility:
      codegen: false
      dag: false
      schema_explorer: false
      skill_editor: false
      agent_builder: false
      domain_composer: false
      glossary_manager: false
      entity_manager: false
      source_config: false
      learnings_manager: false
      query_review: false
      query_history: true         # Own history only
    writes:
      sources: false
      glossary: false
      entities: false
      skills: false
      agents: false
      facts: false
      learnings: false
      domains: false
    feedback:
      flag_answers: true          # "This answer is wrong"
      suggest_glossary: true      # "We call this X not Y"
      suggest_entities: false
      auto_approve: false         # Needs SME/builder approval

  viewer:
    description: "Read-only chat access"
    visibility:
      codegen: false
      dag: false
      schema_explorer: false
      skill_editor: false
      agent_builder: false
      domain_composer: false
      glossary_manager: false
      entity_manager: false
      source_config: false
      learnings_manager: false
      query_review: false
      query_history: false
    writes: {}
    feedback: {}
```

## User-Role Assignment

Roles are assigned per-user in the server permissions config. Each user gets one platform role and a list of domain scopes.

```yaml
# config.yaml
server:
  permissions:
    users:
      alice@company.com:
        role: platform_admin
        # domains: [] — implicit all

      bob@company.com:
        role: domain_builder
        domains: [sales.yaml, hr.yaml]

      carol@company.com:
        role: sme
        domains: [sales.yaml]

      dave@company.com:
        role: domain_user
        domains: [sales.yaml]

      external@client.com:
        role: viewer
        domains: [sales.yaml]

    default:
      role: viewer
      domains: []
```

Platform admin scope is cross-domain. All other roles are scoped to their assigned domains.

## Role Boundaries

### Platform Admin vs Domain Builder

Admin decides **what domains exist and who can build them**. Builder decides **what's inside the domain**.

- Admin: creates/archives domains, assigns builders, sets quotas, manages users
- Builder: connects sources, builds entities/skills/agents, composes domains, publishes

### Domain Builder vs SME

Builder builds the machine. SME certifies it speaks the right language.

| | Builder needs from SME | SME needs from Builder |
|---|---|---|
| **Glossary** | "What does 'active customer' mean?" | "I added the definition, wire it in" |
| **Entities** | "Which entities matter?" | "I tagged them, build relationships" |
| **Facts** | "What are business constants?" | "Fiscal year starts April 1" |
| **Learnings** | "What keeps going wrong?" | "Users confuse X with Y" |
| **Skills** | "What analyses are needed?" | "Weekly churn by segment" → builder makes skill |
| **Validation** | "Does this query match reality?" | "No, margin excludes returns" |

SME also has read access to skills and recent queries for review/validation purposes.

### SME vs Domain User

SME owns the domain language. Domain user consumes it and provides feedback.

- Domain user flags: "this answer used the wrong definition of margin"
- Creates a learning + glossary suggestion
- SME reviews, edits the glossary term directly, marks the learning as resolved
- Next query uses the corrected definition

### Domain User vs Viewer

Domain user participates in the feedback loop. Viewer is pure consumption — no flags, no suggestions, no history.

## Glossary Evolution Loop

The feedback mechanism that makes roles valuable:

1. Domain user asks "what's our margin by region?"
2. Answer uses gross margin
3. User flags: "margin means contribution margin for us"
4. Glossary suggestion created with context
5. SME reviews and edits the glossary term
6. All future queries use the corrected definition

This is why business users maintain the glossary — they see wrong answers, they're motivated to fix the cause, correction happens in-context, and they see improvement immediately.

## Composite Domains

Domain builder can compose multiple source domains into a composite domain. Example: HR + Finance = workforce cost analysis, with its own glossary, entities, and skills spanning both sources.

Platform admin controls which domains exist. Builder wires them together. SME curates the composite glossary.

## Product Implications

### One Codebase, Five Products

Each role is a different window into the same engine:

- **Platform Admin** → ops console
- **Domain Builder** → domain IDE / chatbot builder
- **SME** → knowledge curation + QA tool
- **Domain User** → domain-specific data chatbot
- **Viewer** → embedded chat widget

### Pricing Axis

Charge for seats by role tier, not features. The engine is the same — access level is the product dimension.

### Go-to-Market

Sell to teams that know their domain but don't want to build from scratch:
- Data teams building internal analytics tools
- SaaS companies adding analytics chat to their product
- Consultancies packaging domain expertise as deliverables

### Competitive Moat

The glossary + entity model is the differentiator. Generic "chat with your data" tools let the LLM guess at semantics. Constat forces the builder/SME to define them, so answers are grounded in actual domain meaning.

## Implementation Notes

### What Exists Today

- `DomainConfig` already has: sources, glossary, entities, skills, system_prompt, rights, facts, learnings
- `UserPermissions` has: admin bool + resource allow-lists (databases, apis, documents, domains)
- `TieredConfigLoader` merges 5 config tiers with attribution
- `SkillManager` handles skill CRUD and loading from system/domain/user dirs
- `PermissionsConfig` assigns permissions per user email

### What Changes

- `UserPermissions.admin: bool` → `UserPermissions.role: str` (enum of five roles)
- Add `RoleConfig` model with `visibility`, `writes`, `feedback` sections
- Load role definitions from `roles.yaml` at startup
- Server middleware checks `role.visibility` and `role.writes` before route handlers
- Frontend reads role config at login, conditionally renders UI components
- Add glossary suggestion/approval workflow (ties into glossary architecture)
- Add answer flagging mechanism (creates learning + optional glossary suggestion)

### What Doesn't Change

- Query execution engine
- Glossary resolution
- Skill execution
- DAG/proof engine
- Storage layer
- LLM calls
- Tiered config merging