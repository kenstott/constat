# 3-Tier Configuration Architecture Plan

## Overview

Implement a hierarchical configuration system with three levels:
- **System** (lowest precedence) - co-located with config.yaml
- **User** (middle precedence) - `.constat/<user_id>/`
- **Session** (highest precedence) - `.constat/<user_id>/sessions/<session_id>/`

Each level can define: `facts.yaml`, `learnings.yaml`, `roles.yaml`, `preferences.yaml`, `config.yaml`, and `skills/` directory.

## Key Design Decisions

### 1. Directory Structure
```
<config_dir>/                        # System tier (alongside config.yaml)
├── config.yaml                      # Main system config
├── facts.yaml                       # System facts (optional)
├── learnings.yaml                   # System learnings (optional)
├── roles.yaml                       # System roles (optional)
├── skills/                          # System skills
└── projects/                        # System project definitions (existing)
    └── *.yaml                       # Individual project files

.constat/<user_id>/                  # User tier
├── config.yaml                      # User config overrides (NEW)
├── facts.yaml                       # User facts (existing)
├── learnings.yaml                   # User learnings (existing)
├── roles.yaml                       # User roles (existing)
├── preferences.yaml                 # User preferences (existing)
├── skills/                          # User skills (existing)
└── projects/                        # User project definitions (NEW)
    └── *.yaml                       # User's personal projects

.constat/<user_id>/sessions/<id>/    # Session tier
├── config.yaml                      # Session config overrides (NEW)
├── facts.yaml                       # Session facts (NEW)
├── state.json                       # Session state (existing)
└── skills/                          # Session skills for testing (NEW)
# Note: No learnings.yaml at session level - learnings are transient until compacted to user
```

**Note:** Session-level projects are not supported (projects are persistent by nature).

### 2. Enable/Disable Cascade Model
- All resources have an `enabled` field (default: true)
- **Disabled cascades down** - if system disables, user/session can't re-enable
- **Enabled can be disabled** - if system enables, user can disable (session can't re-enable)

```python
def is_enabled(system_val, user_val, session_val) -> bool:
    """Cascade disable logic."""
    if system_val is False:
        return False  # System disabled = always disabled
    if user_val is False:
        return False  # User disabled = disabled for this user
    if session_val is False:
        return False  # Session disabled = disabled for this session
    return True
```

### 3. Merge Behavior by Resource Type

| Resource     | Key Field | Merge Behavior | Rationale |
|-------------|-----------|----------------|-----------|
| databases   | name      | Deep merge     | User adds credentials to system-defined DB |
| apis        | name      | Deep merge     | User adds auth to system-defined API |
| documents   | name      | Deep merge     | User adds metadata to system doc |
| projects    | filename  | Deep merge     | User extends system project with additional sources |
| roles       | name      | Replace        | Role prompt is a complete unit |
| skills      | dir name  | Replace        | Skill is a complete unit |
| facts       | name      | Replace        | Higher tier fact value wins for same name |
| learnings   | id        | Special        | Session=transient, User=persisted, System=rules only (see below) |
| preferences | key       | Deep merge     | User overrides specific preferences |

**Facts merge example:**
```yaml
# system facts.yaml
company_name: "Acme Corp"
fiscal_year_start: "January 1"

# user facts.yaml
fiscal_year_start: "April 1"  # Overrides system
my_department: "Engineering"   # Adds new fact

# Result: company_name (system), fiscal_year_start=April 1 (user), my_department (user)
```

**Learnings/Rules tier model:**

| Tier | Raw Learnings | Rules |
|------|---------------|-------|
| Session | Captured during session (transient, not persisted) | N/A |
| User | Persisted raw learnings | Compacted from user learnings |
| System | N/A | Promoted from user rules (admin only) |

**Workflow:**
1. **Session**: Raw learnings captured during execution (transient)
2. **Compact to User**: `POST /learnings/compact` promotes session learnings to user rules
3. **Promote to System**: Admin can promote individual user rules to system level

**Promotion to System (admin only, uses LLM inference):**
```python
async def promote_rule_to_system(rule_id: str, llm: LLM) -> Rule:
    """
    Promote user rule to system tier.
    Uses LLM to:
    - Check for similar existing system rules
    - Merge concepts if similar rule exists
    - Generate unified rule summary
    """
    user_rule = get_user_rule(rule_id)
    similar_system = find_similar_system_rules(user_rule)

    if similar_system:
        # LLM merges concepts into existing system rule
        merged = await llm.merge_rule_concepts(similar_system, user_rule)
        update_system_rule(similar_system.id, merged)
    else:
        # Add as new system rule
        save_system_rule(user_rule)
```

### 4. Source Attribution
Every item tracks its source tier:
```python
class ConfigSource(Enum):
    SYSTEM = "system"
    USER = "user"
    SESSION = "session"
```

### 5. Tier Promotion
Items can be promoted from lower to higher tiers:

| Promotion Path | Who Can Do It | Use Case |
|---------------|---------------|----------|
| Session → User | Any user | "Remember this fact for future sessions" |
| User → System | Admin only | "Make this available to all users" |
| Session → System | Admin only | "Make this globally available" |

**Promotion behavior:**
- Promoting copies the item to the target tier's file
- Original item can be optionally deleted from source tier
- On conflict (name already exists at target), prompt user for action

**API endpoints:**
```
POST /facts/{name}/promote
  body: { target_tier: "user" | "system" }

POST /rules/{id}/promote
  body: { target_tier: "system" }  # User rules → System only (admin)

POST /databases/{name}/promote
  body: { target_tier: "user" | "system" }
```

**Note:** Learnings use existing `/learnings/compact` to promote session → user rules.

**Admin check:** Uses existing `UserPermissions.admin` field

### 6. Embeddings and Entities Refresh
When sources are added/removed at any tier, vector store and entities must be updated:

**Triggers for refresh:**
- Database added/removed (any tier)
- Document added/removed (any tier)
- API added/removed (any tier)
- Tier promotion of a source
- Enable/disable state change

**Refresh actions:**
1. **Vector store** (`VectorStore.refresh_entities()`):
   - Re-extract entities from all active sources
   - Update embeddings for new/changed sources
   - Remove embeddings for deleted/disabled sources

2. **Schema cache** (`SchemaManager`):
   - Invalidate cache for changed databases
   - Re-introspect schema if database config changed

3. **Entity extraction**:
   - NER runs on newly added documents
   - Entities removed when document/database disabled or removed

**Implementation:**
- `TieredConfigLoader` emits change events when config changes
- `SessionManager.on_config_change()` triggers appropriate refreshes
- Leverage existing `refresh_entities()` call pattern from dynamic resource changes

## Implementation Phases

### Phase 1: Core Infrastructure
**Files to create:**
- `constat/core/tiered_config.py` - New module for tiered config loading

**Key classes:**
```python
@dataclass
class SourcedItem(Generic[T]):
    """Wrapper tracking source of any config item."""
    value: T
    source: ConfigSource
    source_path: Optional[str] = None
    enabled: bool = True

class TieredConfigLoader:
    """Loads and merges config from all three tiers."""
    def __init__(self, config_dir: Path, user_id: str, base_dir: Path, session_id: Optional[str])
    def load_facts(self) -> dict[str, SourcedItem[Any]]       # Keyed by fact name
    def load_learnings(self) -> dict[str, SourcedItem[dict]]  # Keyed by learning ID
    def load_roles(self) -> dict[str, SourcedItem[Role]]      # Keyed by role name
    def load_skills(self) -> dict[str, SourcedItem[Skill]]    # Keyed by skill name
    def load_databases(self) -> dict[str, SourcedItem[DatabaseConfig]]
    def load_apis(self) -> dict[str, SourcedItem[APIConfig]]
    def load_preferences(self) -> dict[str, SourcedItem[Any]]
```

### Phase 2: Modify Existing Stores
**Files to modify:**

1. **`constat/storage/facts.py`** - FactStore
   - Add `source: ConfigSource` field to stored facts
   - Add `load_from_tier(tier_dir, source)` method
   - Modify `list_facts()` to include source in response

2. **`constat/storage/learnings.py`** - LearningStore
   - Add `source: ConfigSource` field to learnings and rules
   - Add `load_from_tier(tier_dir, source)` method
   - Modify list methods to include source

3. **`constat/core/roles.py`** - RoleManager
   - Modify constructor to accept tiered directories
   - Implement shadow resolution (session > user > system)
   - Track source per role

4. **`constat/core/skills.py`** - SkillManager
   - Modify `_find_skill_directories()` to search all tiers
   - Implement shadow resolution by skill name
   - Track source per skill

5. **`constat/core/config.py`** - Project loading
   - Extend project loading to search system and user tiers
   - Projects from user tier merge with system projects by filename
   - Track source tier per project

### Phase 3: Integrate with Config Loading
**Files to modify:**

1. **`constat/core/config.py`**
   - Add `TieredConfigLoader` integration to `Config.from_yaml()`
   - Add `user_dir` and `session_dir` parameters
   - Store merged source tracking information
   - Add `get_source(resource_type, name)` method

2. **`constat/session.py`**
   - Initialize with tiered loader
   - Use merged config for all resource access
   - Pass source information to fact_resolver

3. **`constat/server/session_manager.py`**
   - Pass session_id to tiered loader
   - Handle session-level config persistence in save_resources/restore_resources

### Phase 4: API Updates
**Files to modify:**

1. **`constat/server/routes/data.py`**
   - Include source tier in facts response (already partially done with `source="config"`)
   - Include source tier in learnings response

2. **`constat/server/routes/databases.py`**
   - Include source tier in database list response
   - Include source tier in API list response

3. **`constat-ui/src/components/artifacts/ArtifactPanel.tsx`**
   - Display source tier badge (system/user/session) similar to "core" badge
   - Color coding: system=gray, user=blue, session=green
   - Add "promote" button for session/user items
   - Show promote-to-system option only for admins

4. **`constat/server/routes/promote.py`** (NEW)
   - `POST /facts/{name}/promote` - promote fact to higher tier
   - `POST /learnings/{id}/promote` - promote learning to higher tier
   - `POST /databases/{name}/promote` - promote database config to higher tier
   - Admin check for promotion to system tier

### Phase 5: Session Persistence
**Files to modify:**

1. **`constat/storage/history.py`** - SessionHistory
   - Add methods to save/load session-tier config files
   - Persist facts.yaml, learnings.yaml to session directory

2. **`constat/server/session_manager.py`**
   - Save session config on changes
   - Restore session config on resume

## Critical Files Summary

| File | Changes |
|------|---------|
| `constat/core/tiered_config.py` | **NEW** - Core tiered loading logic |
| `constat/server/routes/promote.py` | **NEW** - Tier promotion endpoints |
| `constat/core/config.py` | Add tiered integration |
| `constat/storage/facts.py` | Add source tracking, add promote method |
| `constat/storage/learnings.py` | Add source tracking, add promote method |
| `constat/core/roles.py` | Multi-tier loading |
| `constat/core/skills.py` | Multi-tier loading |
| `constat/session.py` | Use tiered config |
| `constat/server/session_manager.py` | Session config persistence |
| `constat/server/routes/data.py` | API source responses |
| `constat/server/routes/databases.py` | API source responses |
| `constat-ui/src/components/artifacts/ArtifactPanel.tsx` | UI source badges + promote buttons |
| `constat-ui/src/api/sessions.ts` | Add promote API calls |
| `constat/discovery/vector_store.py` | Entity refresh on tier changes |
| `constat/catalog/schema_manager.py` | Cache invalidation on tier changes |
| `constat/learning/tiered_merger.py` | **NEW** - LLM-based rule merging across tiers |
| `constat/learning/compactor.py` | Extend for cross-tier rule clustering |

## Verification Plan

1. **Unit tests for merge logic:**
   - Test cascade disable: system disabled → always disabled
   - Test cascade disable: user disabled → session can't enable
   - Test deep merge for databases (credentials merge)
   - Test replace for roles (complete override)
   - Test facts merge by name (user overrides system for same name)
   - Test learnings are transient at session level (not persisted to disk)
   - Test compact promotes session learnings to user rules
   - Test rule promotion to system requires admin
   - Test LLM-based rule merge when promoting similar rule to system

2. **Integration tests:**
   - Create system + user + session facts with overlapping names, verify higher tier wins
   - Create user role that shadows system role, verify user version used
   - Disable database at user level, verify not available in session
   - Add credentials at user level to system database, verify merged

3. **Promotion tests:**
   - Promote session fact to user tier, verify appears in user facts.yaml
   - Promote user fact to system tier as admin, verify appears in system facts.yaml
   - Attempt promote to system as non-admin, verify 403 Forbidden
   - Promote learning from session to user, verify ID moved correctly
   - Promote database config, verify credentials preserved

4. **Project tier tests:**
   - Create project at user tier, verify visible in project list
   - Create project with same name as system project, verify deep merge
   - User adds database to system-defined project, verify merged
   - Promote user project to system tier (admin), verify moved

5. **Embeddings/entities tests:**
   - Add database at user tier, verify entities extracted and available
   - Disable database at session tier, verify entities removed from session
   - Add document at session tier, verify indexed and entities extracted
   - Promote document to user tier, verify embeddings preserved
   - Remove source, verify embeddings and entities cleaned up

5. **Manual testing:**
   - Add `facts.yaml` to config directory, verify "system" badge in UI
   - Add role to user directory, verify shadows system role of same name
   - Create session with custom facts, restart, verify persisted
   - Disable API at user level, verify hidden in session
   - Click promote button on session fact, verify promoted to user
   - As admin, click promote-to-system, verify promoted
   - Add/remove database, verify entities update in UI

## Backward Compatibility

- Existing user-level files continue to work unchanged
- Existing config.yaml continues to work unchanged
- New system-level files are optional
- New session-level files are optional
- Source tracking is additive (defaults to appropriate tier)
