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
| `/prove` | Verify claims with auditable proof |
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
| **Glossary** | Business terms and definitions discovered from data |

Use the `«`/`»` button on the panel edge to hide or show the panel.

## Toolbar

The bottom toolbar provides:
- **New** — Start a new session
- **Cancel** — Stop running execution
- **Proof** — Run auditable proof verification (available after execution)
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

- **Facts** appear in proof trees and provide values for calculations
- **Learnings** influence how the system approaches problems
- **Rules** are automatically promoted from multiple related learnings
