# Domain Management Architecture

> **Status:** Design. No phases implemented yet.

## Problem

Domain management is buried in the hamburger menu (`constat-ui/src/components/layout/HamburgerMenu.tsx`) as a flat checkbox list. This causes three problems:

1. **Hidden** — Users don't discover domains until they open the menu. Domains scope everything (data sources, entities, glossary terms) but appear as an afterthought.
2. **Flat** — Domains have a `path` field for hierarchy (`sales.north-america.retail`) but the checkbox list can't represent it. The `/domains/tree` endpoint returns nested `DomainTreeNode` structures that nobody renders.
3. **No scoping for skills/agents/rules** — These are user-global. A skill created while exploring sales analytics is indistinguishable from one created for HR reporting. No way to move, promote, or scope them.

Glossary terms already have domain badges, ownership (`owner` field), and promotion workflows (`status: draft → reviewed → approved`). This extends that pattern to skills, agents, and rules.

## Core Idea

Domains become the **primary organizational unit** — visible, hierarchical, and scoping everything. They move from a hidden checkbox to a first-class panel. Skills, agents, and rules gain domain affiliation with the same ownership/promotion model glossary terms use.

```
┌─────────────────────────────────────────────────┐
│                  Domain Panel                    │
│  (first section in ArtifactPanel)               │
│                                                  │
│  ▼ system                          🔒 read-only │
│    ├── base                                      │
│    └── shared-analytics                          │
│  ▼ shared                          ✏️  editable  │
│    ├── sales-analytics                           │
│    │   ├── 📊 3 databases                        │
│    │   ├── 🔧 2 skills                           │
│    │   └── 🤖 1 agent                            │
│    └── hr-reporting                              │
│  ▼ user (you)                      🧪 sandbox   │
│    └── my-experiments                            │
│        ├── 🔧 revenue-calc (draft)               │
│        └── 📏 1 rule                             │
└─────────────────────────────────────────────────┘
```

Three ownership tiers. Content flows upward: user → shared → system (read-only).

## Data Model

### Ownership Tiers

| Tier | Source | Editable | Purpose |
|------|--------|----------|---------|
| **system** | `config.yaml` domains | No | Curated by admin. Read-only to all users. |
| **shared** | `.constat/shared/domains/` | Yes (owner only) | Promoted from user. Visible to all users. |
| **user** | `.constat/{user_id}/domains/` | Yes (owner) | Personal sandbox. Persists across sessions. |

User domains are **not sessions**. Sessions are ephemeral per-query (`session_id` in entities table, cleared on new session). A user domain is a persistent staging area — experiments, draft skills, and what-if rules survive across sessions until the user promotes or deletes them.

### DomainConfig Extensions

Existing `DomainConfig` (`constat/core/config.py`) already has `owner` and `path`. Add:

```python
class DomainConfig(BaseModel):
    # ... existing fields ...
    owner: str = ""                     # Already exists
    path: str = ""                      # Already exists (dot-delimited hierarchy)

    # New fields
    tier: Literal["system", "shared", "user"] = "user"
    active: bool = True                 # Active/inactive toggle (replaces checkbox)
    order: int = 0                      # Sort order within parent

    # Domain-scoped content (existing)
    glossary: dict[str, Any] = Field(default_factory=dict)
    relationships: dict[str, Any] = Field(default_factory=dict)
    rights: dict[str, Any] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    learnings: dict[str, Any] = Field(default_factory=dict)

    # Domain-scoped content (new)
    skills: list[str] = Field(default_factory=list)    # Skill names in this domain
    agents: list[str] = Field(default_factory=list)    # Agent names in this domain
    rules: list[str] = Field(default_factory=list)     # Rule IDs in this domain
```

### Skill, Agent, and Rule Domain Scoping

Add `domain` and `source` fields to each model, following the glossary term pattern (`glossary_terms.domain`, `glossary_terms.owner`).

**Skills** (`constat/core/skills.py`):

```python
@dataclass
class Skill:
    name: str
    description: str = ""
    domain: str | None = None           # Owning domain filename (None = unscoped)
    source: str = "user"                # "system" | "shared" | "user"
    # ... existing fields ...
```

Storage moves from flat `.constat/{user_id}/skills/{name}/` to domain-scoped:

```
.constat/{user_id}/domains/{domain}/skills/{skill-name}/SKILL.md
.constat/shared/domains/{domain}/skills/{skill-name}/SKILL.md
```

Unscoped skills remain at `.constat/{user_id}/skills/` for backward compatibility during migration.

**Agents** (`constat/core/agents.py`):

```python
@dataclass
class Agent:
    name: str
    prompt: str
    description: str = ""
    skills: list[str] = field(default_factory=list)
    domain: str | None = None           # Owning domain filename
    source: str = "user"                # "system" | "shared" | "user"
```

**Rules** (`constat/server/models.py`):

```python
class RuleInfo(BaseModel):
    id: str
    summary: str
    category: str
    confidence: float
    source_count: int = 0
    tags: list[str] = Field(default_factory=list)
    domain: str | None = None           # Owning domain filename
    source: str = "user"                # "system" | "shared" | "user"
```

Rules start user-scoped. Promotion copies the rule to a shared domain.

## Fully Qualified Skill Names

Skills gain qualified names: `{domain}/{skill-name}`.

```
sales-analytics/revenue-calc
hr-reporting/headcount-formula
my-experiments/revenue-calc         (user domain, same skill name, no conflict)
```

### Resolution Order

When an agent references a skill by unqualified name:

```
1. Current domain (the domain the agent belongs to)
2. User domains (all active user domains)
3. Shared domains (all active shared domains)
4. System domains
5. Unscoped skills (legacy .constat/{user_id}/skills/)
```

First match wins. Qualified names (`domain/skill-name`) bypass resolution and resolve directly.

### Agent Portability

Agents reference skills by qualified name in their `skills` list:

```yaml
data-analyst:
  description: Revenue analysis agent
  skills:
    - sales-analytics/revenue-calc      # Qualified — always resolves
    - headcount-formula                  # Unqualified — resolves by search order
```

Moving an agent between domains doesn't break qualified references. This is "renting a tool" — the agent in `hr-reporting` can use `sales-analytics/revenue-calc` without owning or copying it.

### SkillManager Changes

`SkillManager` (`constat/core/skills.py`) currently scans `.constat/{user_id}/skills/`. Extended to:

```python
class SkillManager:
    def resolve_skill(self, name: str, context_domain: str | None = None) -> Skill | None:
        """Resolve a skill name (qualified or unqualified).

        Qualified: 'domain/skill-name' → direct lookup in that domain.
        Unqualified: 'skill-name' → search order: context_domain → user → shared → system → unscoped.
        """

    def qualified_name(self, skill: Skill) -> str:
        """Return fully qualified name: '{domain}/{name}' or '{name}' if unscoped."""

    def list_skills(self, domain: str | None = None) -> list[Skill]:
        """List skills, optionally filtered by domain."""
```

## UI Design

### Domain Panel

New `AccordionSection` in `ArtifactPanel` (`constat-ui/src/components/artifacts/ArtifactPanel.tsx`), positioned **first** (before Tables, Artifacts, etc.). Domains scope everything below.

```typescript
<AccordionSection
  id="domains"
  title="Domains"
  count={domains.length}
  icon={<GlobeAltIcon />}
  action={<AddDomainButton />}
  command="/domains"
>
  <DomainTree domains={domainTree} />
</AccordionSection>
```

### Tree UI

Reuses the tree pattern from `GlossaryPanel` (`constat-ui/src/components/artifacts/GlossaryPanel.tsx`) which already renders domain-grouped hierarchies with expand/collapse.

```typescript
interface DomainTreeProps {
  domains: DomainTreeNode[]           // From GET /domains/tree
  activeDomains: string[]             // From session.active_domains
  onToggle: (filename: string) => void
  onDrag: (source: DragItem, target: string) => void
}
```

Each node shows:

```
▼ sales-analytics                    shared  ✓ active
    📊 sales, customers (databases)
    🔧 revenue-calc, margin-calc (skills)
    🤖 data-analyst (agent)
    📏 2 rules
```

### Badges

Follow glossary badge pattern (`GlossaryPanel.tsx`):

| Tier | Badge | Color |
|------|-------|-------|
| system | `🔒 system` | `bg-gray-100 text-gray-600` |
| shared | `shared` | `bg-blue-100 text-blue-700` |
| user | `🧪 user` | `bg-amber-100 text-amber-700` |

### CRUD Operations

| Operation | System | Shared | User |
|-----------|--------|--------|------|
| View | Yes | Yes | Yes |
| Activate/deactivate | Yes | Yes | Yes |
| Add | No | Owner only | Yes |
| Rename | No | Owner only | Yes |
| Delete | No | Owner only | Yes |
| Reorder | No | Owner only | Yes |
| Edit content | No | Owner only | Yes |
| Promote | — | — | Yes (→ shared) |

### Drag-and-Drop

HTML5 native drag-and-drop. Draggable items: data sources, skills, agents, rules. Drop targets: domain tree nodes.

```typescript
interface DragItem {
  type: "database" | "api" | "document" | "skill" | "agent" | "rule"
  name: string
  sourceDomain: string | null          // Current domain (null = unscoped)
}
```

Drop on a domain node → calls the move endpoint. Gated by ownership: can only drop into domains the user owns (shared domains they own, or their user domains).

The existing `/domains/move-source` endpoint (`constat/server/routes/learnings.py`) handles data source moves. Extend with parallel endpoints for skills, agents, and rules.

## Backend Endpoints

### Existing (unchanged)

| Endpoint | Method | File | Purpose |
|----------|--------|------|---------|
| `/domains/tree` | GET | `learnings.py` | Nested domain hierarchy |
| `/domains` | GET | `learnings.py` | List all domains |
| `/domains` | POST | `learnings.py` | Create domain |
| `/domains/{filename}` | GET | `learnings.py` | Domain details |
| `/domains/{filename}/content` | GET | `learnings.py` | Domain YAML for editing |
| `/domains/{filename}/content` | PUT | `learnings.py` | Update domain YAML |
| `/domains/move-source` | POST | `learnings.py` | Move database/API/document between domains |
| `/{session_id}/domains` | POST | `sessions.py` | Set active domains for session |

### New Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/domains/{filename}` | PATCH | Rename, reorder, activate/deactivate |
| `/domains/{filename}` | DELETE | Delete domain (shared/user only) |
| `/domains/{filename}/promote` | POST | Promote user domain → shared |
| `/domains/move-skill` | POST | Move skill between domains |
| `/domains/move-agent` | POST | Move agent between domains |
| `/domains/move-rule` | POST | Move rule between domains |
| `/domains/{filename}/skills` | GET | List skills in domain |
| `/domains/{filename}/agents` | GET | List agents in domain |
| `/domains/{filename}/rules` | GET | List rules in domain |

### Move Endpoint Shape

```python
class MoveItemRequest(BaseModel):
    item_type: Literal["skill", "agent", "rule"]
    item_name: str
    source_domain: str | None           # None = unscoped
    target_domain: str

# POST /domains/move-skill
# POST /domains/move-agent
# POST /domains/move-rule
```

Move validates: source exists, target exists, user owns target, no name conflict in target.

### Promote Endpoint

```python
# POST /domains/{filename}/promote
class PromoteRequest(BaseModel):
    target_name: str | None = None      # Optional rename on promote

# Copies domain YAML + all scoped content (skills, agents, rules)
# from .constat/{user_id}/domains/{filename}
# to   .constat/shared/domains/{target_name or filename}
# Sets tier="shared", owner=user_id
```

## TypeScript Types

Extend existing types in `constat-ui/src/api/sessions.ts`:

```typescript
// Existing — add tier and active
export interface DomainInfo {
  filename: string
  name: string
  description: string
  tier: "system" | "shared" | "user"
  active: boolean
  owner: string
}

// Existing — add tier, contents
export interface DomainTreeNode {
  filename: string
  name: string
  path: string
  description: string
  tier: "system" | "shared" | "user"
  active: boolean
  databases: string[]
  apis: string[]
  documents: string[]
  skills: string[]
  agents: string[]
  rules: string[]
  children: DomainTreeNode[]
}
```

## User Domain vs Session

| | User Domain | Session |
|---|---|---|
| **Lifetime** | Persistent until deleted | Ephemeral per query |
| **Storage** | `.constat/{user_id}/domains/` | `.constat/{user_id}/sessions/{id}/` |
| **Purpose** | Staging area for experiments | Query execution context |
| **Contains** | Data sources, skills, agents, rules | Messages, artifacts, steps |
| **Visibility** | Only the owning user | Only the owning user |
| **Promotion** | → shared domain | N/A |

A user creates a domain, adds data sources, writes skills, defines rules. All of this persists. When they start a new session, they activate that domain alongside shared/system domains. The session uses the domain's content but doesn't own it.

## Migration Phases

### Phase 1: Panel + Coexistence

- Add `DomainPanel` as first `AccordionSection` in `ArtifactPanel`
- Render existing domain tree (already returned by `/domains/tree`)
- Activate/deactivate from panel (same `setActiveDomains` call)
- Hamburger menu domain section remains — both work

### Phase 2: Ownership + CRUD

- Add `tier`, `active`, `order` to `DomainConfig`
- Implement create/rename/delete in panel
- Ownership badges
- User domain creation flow

### Phase 3: Scoped Content

- Add `domain`/`source` fields to Skill, Agent, Rule
- Qualified skill names + resolution
- Domain-scoped storage paths
- Move endpoints

### Phase 4: Drag-and-Drop + Promotion

- HTML5 drag-and-drop between domain nodes
- Promote user → shared workflow
- Agent cross-domain skill references ("renting")

### Phase 5: Cleanup

- Remove hamburger menu domain section
- Remove unscoped skill/agent fallback (or keep as "global" domain)

## Design Decisions

### Why panel, not modal or sidebar

Domains scope everything in the artifact panel (tables, glossary, skills). Placing the domain selector in the same panel, above the content it scopes, makes the relationship visible. The hamburger menu hid this relationship.

### Why three tiers, not two

Two tiers (system + user) forces a choice: either everything is personal and can't be shared, or everything is shared and there's no safe experimentation space. The shared tier is the promotion target — user domains are drafts, shared domains are published.

### Why qualified names, not UUIDs

`sales-analytics/revenue-calc` is readable in agent configs, debuggable in logs, and meaningful in error messages. UUIDs would require lookup everywhere. The `/` separator is unambiguous (domain filenames and skill names are both slug-format, no slashes).

### Why not rename the existing move-source endpoint

The existing `/domains/move-source` endpoint (`learnings.py`) moves databases, APIs, and documents. Skills, agents, and rules are structurally different (file-based, not config-based). Separate endpoints keep the request/response shapes clean.

### Why rules start user-scoped

Rules are learned from corrections — inherently personal and unvalidated. Promoting to shared is an explicit decision that the rule is generally applicable, not just a one-off preference.
