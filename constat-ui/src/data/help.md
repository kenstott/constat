# Constat Quick Reference

Constat prioritizes accuracy and provability over speed. Results are auditable and reproducible.

## Workflow

1. **Ask a question** — Type a natural-language analysis request
2. **Review the plan** — Constat shows numbered steps it will execute
3. **Approve or edit** — Approve to proceed, edit step goals, or delete steps
4. **View results** — Tables, charts, and insights appear in the right panel

## Commands

Type `/` in the input to see autocomplete. Key commands:

### Data Exploration

| Command | Description |
|---------|-------------|
| `/tables` | List available tables |
| `/show <table>` | Preview table contents |
| `/export <table>` | Export table to CSV or XLSX |
| `/query <sql>` | Run SQL query on datastore |
| `/code` | Show generated code |
| `/artifacts` | Show artifacts (use `all` for intermediate) |

### Session

| Command | Description |
|---------|-------------|
| `/reset` | Clear session state and start fresh |
| `/redo` | Retry last query (with optional modifications) |
| `/update` / `/refresh` | Refresh metadata and rebuild cache |
| `/context` | Show context size and token usage |
| `/compact` | Compact context to reduce token usage |
| `/state` | Show session state |

### Facts & Memory

| Command | Description |
|---------|-------------|
| `/facts` | Show cached facts from this session |
| `/remember <fact>` | Persist a session fact |
| `/forget <fact>` | Forget a remembered fact |

### Plans & History

| Command | Description |
|---------|-------------|
| `/save <name>` | Save current plan for replay |
| `/share <name>` | Save plan as shared (all users) |
| `/plans` | List saved plans |
| `/replay <name>` | Replay a saved plan |
| `/history` / `/sessions` | List recent sessions |
| `/resume <id>` | Resume a previous session |

### Data Sources

| Command | Description |
|---------|-------------|
| `/databases` | List configured databases |
| `/database <conn>` | Add a database to this session |
| `/apis` | List configured APIs |
| `/api <url>` | Add an API to this session |
| `/documents` / `/docs` | List all documents |
| `/doc <path>` | Add a document to this session |

### Analysis

| Command | Description |
|---------|-------------|
| `/discover <query>` | Search all data sources (returns structured JSON) |
| `/summarize <target>` | Summarize plan, session, facts, or a table |
| `/reason` | Verify claims with auditable reasoning chain |
| `/correct <text>` | Record a correction as a learning |
| `/learnings` | Show learnings and rules |
| `/compact-learnings` | Promote similar learnings into rules |

### Preferences

| Command | Description |
|---------|-------------|
| `/verbose` | Toggle verbose mode |
| `/raw` | Toggle raw output display |
| `/insights` | Toggle insight synthesis |
| `/preferences` | Show current preferences |
| `/user` | Show or set current user |

## Clarification Widgets

When a query is ambiguous, Constat presents clarifying questions before planning. Questions use specialized interactive widgets:

| Widget | Purpose |
|--------|---------|
| **Choice** | Radio group for selecting from suggested answers |
| **Curation** | Keep/remove list for filtering items |
| **Ranking** | Drag-to-reorder for prioritizing items |
| **Table** | Tabular selection for structured data |
| **Mapping** | Match items from two lists |
| **Tree** | Hierarchical selection (expand/collapse) |
| **Annotation** | Label or tag items inline |

Each question can also be answered with free-form text via "Other". You can skip clarification to proceed with defaults.

## Plan Editing

When reviewing a plan:
- **Edit step goals** — Click a step to modify its objective
- **Delete steps** — Click the X to remove unwanted steps
- **Provide feedback** — Add comments to trigger replanning
- **Agent assignment** — Steps can be assigned to specialist agents

## Right Panel

The right panel contains these sections:

| Section | Content |
|---------|---------|
| **Results** | Published tables and artifacts (starred or key results) |
| **Databases** | Configured database connections |
| **APIs** | Configured API sources |
| **Documents** | Uploaded or linked documents |
| **Roles** | Domain-specialist personas that guide analysis |
| **Skills** | Reusable capabilities applied to steps |
| **Learnings** | Patterns learned from corrections and interactions |
| **Exploratory Code** | Python code generated for each step (collapsible, downloadable) |
| **Facts** | Concrete values with provenance |
| **Domains** | Domain hierarchy with tiers, resources, and grants |
| **Glossary** | Business terms, definitions, relationships, taxonomy, and aliases |

Use the `«`/`»` button on the panel edge to hide or show the panel.

## Toolbar

The bottom toolbar provides:
- **New** — Start a new session
- **Cancel** — Stop running execution
- **Reasoning Chain** — Run auditable reasoning chain verification (available after execution)
- **Brief** — Toggle brief mode (skips insight synthesis for faster results)
- **Help** — Show this help

## Knowledge Types

| Type | Purpose | Created by |
|------|---------|------------|
| **Source** | Database, API, or document connection | Configuration |
| **Fact** | Concrete value with audit trail | Query execution |
| **Learning** | Behavioral preference or domain knowledge | Corrections (`/correct`) |
| **Rule** | Consolidated learnings with higher confidence | Automatic compaction |
| **Glossary term** | Business term definition | Discovery and refinement |

- **Facts** appear in reasoning chains and provide values for calculations
- **Learnings** influence how the system approaches problems
- **Rules** are automatically promoted from multiple related learnings

## Regression Testing

Golden questions let you define expected outcomes for a domain and verify them automatically.

- Define questions with expected entities, grounding, relationships, and glossary assertions in domain YAML
- Run from UI (Regression panel) or CLI: `constat test -c config.yaml`
- Filter by domain (`-d sales-analytics`) or tag (`--tags smoke`)
- **Unit tests** verify glossary and entity integrity (fast, no LLM)
- Use `--e2e` to include **integration tests** that run the full NLQ pipeline (slower, costs tokens)

## Additional Commands

Beyond the commands listed above, Constat supports command families for managing domain resources:

- `/rule`, `/rule-edit`, `/rule-delete` — Manage rules
- `/agent`, `/agent-edit`, `/agent-delete` — Manage agents
- `/skill`, `/skill-edit`, `/skill-delete` — Manage skills

Type `/` in the input to see the full autocomplete list.
