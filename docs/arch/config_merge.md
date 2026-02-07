# 4-Tier Configuration Architecture Plan

## Overview

Implement a hierarchical configuration system with four levels:
- **System** (lowest precedence) - co-located with config.yaml
- **Project** (second precedence) - self-contained project directories
- **User** (third precedence) - `.constat/<user_id>/`
- **Session** (highest precedence) - `.constat/<user_id>/sessions/<session_id>/`

**Merge order:** system → project → user → session

Each level can define: `config.yaml`, `facts.yaml`, `learnings.yaml`, `roles.yaml`, `preferences.yaml`, and `skills/` directory.

## Key Design Decisions

### 1. Directory Structure

**Projects are directories, not YAML files.** A project has the same structure as system/user tiers, making it a complete, portable unit of configuration.

```
<config_dir>/                        # System tier (alongside config.yaml)
├── config.yaml                      # Main system config
├── facts.yaml                       # System facts (optional)
├── learnings.yaml                   # System learnings (optional)
├── roles.yaml                       # System roles (optional)
├── skills/                          # System skills
└── projects/                        # Project directories
    └── hr-reporting/                # Each project is a directory
        ├── config.yaml              # Project config (databases, apis, documents)
        ├── facts.yaml               # Project facts
        ├── learnings.yaml           # Project learnings/rules
        ├── roles.yaml               # Project roles
        ├── preferences.yaml         # Project preferences
        └── skills/                  # Project skills
            └── hr_analysis/
                └── skill.yaml

.constat/<user_id>/                  # User tier
├── config.yaml                      # User config overrides
├── facts.yaml                       # User facts (existing)
├── learnings.yaml                   # User learnings (existing)
├── roles.yaml                       # User roles (existing)
├── preferences.yaml                 # User preferences (existing)
├── skills/                          # User skills (existing)
└── projects/                        # User project directories (NEW)
    └── my-analytics/                # User's personal projects
        ├── config.yaml
        ├── roles.yaml
        └── skills/

.constat/<user_id>/sessions/<id>/    # Session tier
├── config.yaml                      # Session config overrides
├── facts.yaml                       # Session facts
├── state.json                       # Session state (existing)
└── skills/                          # Session skills for testing
# Note: No learnings.yaml at session level - learnings are transient until compacted to user
```

### 2. Project Structure Example

```
projects/hr-reporting/
├── config.yaml              # Databases, APIs, documents
│   databases:
│     hr:
│       uri: sqlite:///data/hr.db
│       description: HR data with employees and reviews
│   documents:
│     compensation_policy:
│       path: docs/compensation_policy.pdf
│
├── roles.yaml               # Project-specific roles
│   hr_analyst:
│     description: Analyzes employee data for HR decisions
│     prompt: |
│       You are an HR analyst. Focus on retention,
│       performance trends, and compensation equity.
│       Never expose individual salaries.
│
├── facts.yaml               # Project-specific facts
│   rating_scale: "1-5 (5=exceptional, 1=unsatisfactory)"
│   fiscal_year_start: "January 1"
│
├── learnings.yaml           # Project rules/learnings
│   rules:
│     - id: hr_privacy_rule
│       pattern: "salary queries"
│       action: "aggregate, never individual"
│
└── skills/                  # Project-specific skills
    └── headcount_report/
        └── skill.yaml
```

### 3. Enable/Disable Cascade Model
- All resources have an `enabled` field (default: true)
- **Disabled cascades down** - if system disables, project/user/session can't re-enable
- **Enabled can be disabled** - each tier can disable what higher precedence tiers enabled

```python
def is_enabled(system_val, project_val, user_val, session_val) -> bool:
    """Cascade disable logic."""
    if system_val is False:
        return False  # System disabled = always disabled
    if project_val is False:
        return False  # Project disabled = disabled for this project
    if user_val is False:
        return False  # User disabled = disabled for this user
    if session_val is False:
        return False  # Session disabled = disabled for this session
    return True
```

### 4. Merge Behavior by Resource Type

| Resource     | Key Field | Merge Behavior | Rationale |
|-------------|-----------|----------------|-----------|
| databases   | name      | Deep merge     | User adds credentials to project-defined DB |
| apis        | name      | Deep merge     | User adds auth to project-defined API |
| documents   | name      | Deep merge     | User adds metadata to project doc |
| roles       | name      | Replace        | Role prompt is a complete unit |
| skills      | dir name  | Replace        | Skill is a complete unit |
| facts       | name      | Replace        | Higher tier fact value wins for same name |
| learnings   | id        | Special        | Session=transient, others=persisted |
| preferences | key       | Deep merge     | User overrides specific preferences |

**Facts merge example:**
```yaml
# system facts.yaml
company_name: "Acme Corp"
fiscal_year_start: "January 1"

# project facts.yaml (hr-reporting)
fiscal_year_start: "April 1"  # Overrides system for this project
rating_scale: "1-5"

# user facts.yaml
my_department: "Engineering"   # Adds user-specific fact

# Result when hr-reporting selected:
# company_name (system), fiscal_year_start=April 1 (project),
# rating_scale (project), my_department (user)
```

**Roles merge example:**
```yaml
# system roles.yaml
analyst:
  prompt: "Generic analyst prompt..."

# project roles.yaml (hr-reporting)
analyst:
  prompt: "HR-specific analyst prompt..."  # Completely replaces system analyst

# user roles.yaml
analyst:
  prompt: "My custom analyst..."  # Completely replaces project analyst
```

### 5. Source Attribution
Every item tracks its source tier:
```python
class ConfigSource(Enum):
    SYSTEM = "system"
    PROJECT = "project"
    USER = "user"
    SESSION = "session"
```

### 6. Tier Promotion

| Promotion Path | Who Can Do It | Use Case |
|---------------|---------------|----------|
| Session → User | Any user | "Remember this fact for future sessions" |
| Session → Project | Project owner | "Add this to the project" |
| User → System | Admin only | "Make this available to all users" |
| Project → System | Admin only | "Make this a system-wide project" |

### 7. Project Selection and Loading

When a session selects a project:
1. Load system tier config
2. Load selected project directory (merge over system)
3. Load user tier config (merge over project)
4. Load session tier config (merge over user)

```python
class TieredConfigLoader:
    def __init__(
        self,
        config_dir: Path,
        user_id: str,
        base_dir: Path,
        session_id: Optional[str],
        project_name: Optional[str] = None  # Selected project
    ):
        self.tiers = [
            ("system", config_dir),
            ("project", config_dir / "projects" / project_name) if project_name else None,
            ("user", base_dir / user_id),
            ("session", base_dir / user_id / "sessions" / session_id) if session_id else None,
        ]
        self.tiers = [(name, path) for name, path in self.tiers if path]
```

### 8. Embeddings and Entities Refresh
When sources are added/removed at any tier, vector store and entities must be updated:

**Triggers for refresh:**
- Project selection changed
- Database added/removed (any tier)
- Document added/removed (any tier)
- API added/removed (any tier)
- Tier promotion of a source
- Enable/disable state change

## Implementation Phases

### Phase 1: Project Directory Structure
**Changes:**
1. Modify `ProjectConfig` to load from directory instead of YAML file
2. Add `ProjectConfig.from_directory(path: Path)` method
3. Support all resource types in project: config.yaml, roles.yaml, facts.yaml, learnings.yaml, skills/
4. Update project discovery to find directories, not YAML files

```python
class ProjectConfig(BaseModel):
    """Project configuration loaded from a directory."""
    name: str
    description: str = ""
    source_path: str = ""  # Directory path

    # Data sources (from config.yaml)
    databases: dict[str, DatabaseConfig] = Field(default_factory=dict)
    apis: dict[str, APIConfig] = Field(default_factory=dict)
    documents: dict[str, DocumentConfig] = Field(default_factory=dict)

    # Additional resources (from their respective files)
    roles: dict[str, Role] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)
    learnings: list[dict] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)  # Skill directory names
    preferences: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_directory(cls, path: Path) -> "ProjectConfig":
        """Load project from directory."""
        if not path.is_dir():
            raise ValueError(f"Project must be a directory: {path}")

        # Load config.yaml (databases, apis, documents)
        config_path = path / "config.yaml"
        data = {"name": path.name, "source_path": str(path)}
        if config_path.exists():
            with open(config_path) as f:
                data.update(yaml.safe_load(f) or {})

        # Load roles.yaml
        roles_path = path / "roles.yaml"
        if roles_path.exists():
            with open(roles_path) as f:
                data["roles"] = yaml.safe_load(f) or {}

        # Load facts.yaml
        facts_path = path / "facts.yaml"
        if facts_path.exists():
            with open(facts_path) as f:
                data["facts"] = yaml.safe_load(f) or {}

        # Load learnings.yaml
        learnings_path = path / "learnings.yaml"
        if learnings_path.exists():
            with open(learnings_path) as f:
                data["learnings"] = yaml.safe_load(f) or []

        # Discover skills
        skills_path = path / "skills"
        if skills_path.is_dir():
            data["skills"] = [d.name for d in skills_path.iterdir() if d.is_dir()]

        return cls.model_validate(data)
```

### Phase 2: 4-Tier Config Loader
**New file:** `constat/core/tiered_config.py`

```python
@dataclass
class SourcedItem(Generic[T]):
    """Wrapper tracking source of any config item."""
    value: T
    source: ConfigSource
    source_path: Optional[str] = None
    enabled: bool = True

class TieredConfigLoader:
    """Loads and merges config from all four tiers."""

    def __init__(
        self,
        config_dir: Path,
        user_id: str,
        base_dir: Path,
        session_id: Optional[str],
        project_name: Optional[str] = None
    ):
        self.config_dir = config_dir
        self.project_name = project_name
        self.user_dir = base_dir / user_id
        self.session_dir = base_dir / user_id / "sessions" / session_id if session_id else None

    def load_facts(self) -> dict[str, SourcedItem[Any]]:
        """Load facts from all tiers, later tiers override earlier."""
        result = {}

        # System
        result.update(self._load_facts_file(
            self.config_dir / "facts.yaml", ConfigSource.SYSTEM
        ))

        # Project
        if self.project_name:
            project_dir = self.config_dir / "projects" / self.project_name
            result.update(self._load_facts_file(
                project_dir / "facts.yaml", ConfigSource.PROJECT
            ))

        # User
        result.update(self._load_facts_file(
            self.user_dir / "facts.yaml", ConfigSource.USER
        ))

        # Session
        if self.session_dir:
            result.update(self._load_facts_file(
                self.session_dir / "facts.yaml", ConfigSource.SESSION
            ))

        return result

    def load_roles(self) -> dict[str, SourcedItem[Role]]:
        """Load roles from all tiers, later tiers replace earlier (by name)."""
        # Similar pattern...

    def load_skills(self) -> dict[str, SourcedItem[Skill]]:
        """Load skills from all tiers, later tiers replace earlier (by name)."""
        # Similar pattern...
```

### Phase 3: Migrate Existing Projects
**Migration script:**
```python
def migrate_project_yaml_to_directory(yaml_path: Path):
    """Convert old project.yaml to new directory structure."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    project_dir = yaml_path.parent / yaml_path.stem
    project_dir.mkdir(exist_ok=True)

    # Write config.yaml (databases, apis, documents)
    config = {
        "name": data.get("name"),
        "description": data.get("description"),
        "databases": data.get("databases", {}),
        "apis": data.get("apis", {}),
        "documents": data.get("documents", {}),
    }
    with open(project_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Move original yaml to backup
    yaml_path.rename(yaml_path.with_suffix(".yaml.bak"))
```

### Phase 4: Update Project Loading
**Files to modify:**
- `constat/core/config.py` - `Config.load_project()` to load from directory
- `constat/server/routes/databases.py` - Project listing to scan directories
- `constat/server/session_manager.py` - Project selection to load full tier

### Phase 5: API and UI Updates
- List projects shows directories, not YAML files
- Project detail view shows all resources (roles, skills, facts, etc.)
- Project creation creates directory structure
- Promote resources to project tier

## Critical Files Summary

| File | Changes |
|------|---------|
| `constat/core/config.py` | `ProjectConfig.from_directory()`, project discovery |
| `constat/core/tiered_config.py` | **NEW** - 4-tier loading with project support |
| `constat/core/roles.py` | Load from project tier |
| `constat/core/skills.py` | Load from project tier |
| `constat/storage/facts.py` | Load from project tier |
| `constat/storage/learnings.py` | Load from project tier |
| `constat/server/routes/databases.py` | Project listing, creation |
| `constat/server/session_manager.py` | Project selection |

## Backward Compatibility

1. **Migration script** converts existing `project.yaml` files to directories
2. **Fallback loading**: If a project path is a file, load as old YAML format (with deprecation warning)
3. **Existing user directories** continue to work unchanged
4. **Session-level config** is optional

## Example: Complete HR Project

```
projects/hr-reporting/
├── config.yaml
│   name: HR Reporting
│   description: Employee and performance data for HR analysis
│
│   databases:
│     hr:
│       uri: sqlite:///data/hr.db
│       description: HR data with departments, employees, reviews
│
│   documents:
│     compensation_policy:
│       path: docs/compensation_policy.pdf
│       description: Salary bands and bonus structure
│
├── roles.yaml
│   hr_analyst:
│     description: Analyzes employee data with privacy focus
│     prompt: |
│       You are an HR analyst. Your responsibilities:
│       - Analyze retention and turnover patterns
│       - Review performance trends by department
│       - Ensure compensation equity
│
│       Privacy rules:
│       - Never expose individual salaries
│       - Aggregate sensitive data (min 5 employees)
│       - Mask employee names in outputs
│
│   hr_manager:
│     description: Reviews team performance and approves changes
│     prompt: |
│       You are an HR manager with elevated access...
│
├── facts.yaml
│   rating_scale: "1-5 scale (5=exceptional, 1=unsatisfactory)"
│   review_cycle: "Annual, completed by December 31"
│   salary_bands:
│     junior: "$50,000 - $70,000"
│     senior: "$70,000 - $100,000"
│     lead: "$100,000 - $140,000"
│
├── learnings.yaml
│   rules:
│     - id: privacy_aggregation
│       pattern: "salary|compensation queries"
│       action: "Aggregate by department, min 5 employees"
│
│     - id: performance_context
│       pattern: "low performer|underperforming"
│       action: "Include tenure and improvement plans"
│
└── skills/
    └── headcount_report/
        └── skill.yaml
```
