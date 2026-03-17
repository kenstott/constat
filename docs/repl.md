# Terminal REPL Guide

The terminal REPL is Constat's keyboard-driven interface — fastest iteration, full command set, and scriptable.

## Starting the REPL

```bash
# Interactive session
constat repl -c config.yaml

# Solve a single problem
constat solve "What are the top 5 customers by revenue?" -c config.yaml
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `solve <problem>` | Solve a single problem with multi-step planning |
| `repl` | Start interactive REPL session (Textual TUI) |
| `serve` | Start the REST API server for web UI access |
| `history` | List recent sessions |
| `resume <id>` | Resume a previous session |
| `validate` | Validate a config file |
| `schema` | Show database schema overview |
| `init` | Generate a sample config.yaml |
| `test` | Run golden question regression tests for domain quality |

**`serve` options:**
```bash
constat serve -c config.yaml                  # Start on localhost:8000
constat serve -c config.yaml --port 8080      # Custom port
constat serve -c config.yaml --host 0.0.0.0   # Bind to all interfaces
constat serve -c config.yaml --reload         # Auto-reload for development
constat serve -c config.yaml --debug          # Enable debug logging
```

**`test` options:**
```bash
constat test -c config.yaml                    # Run all domain tests
constat test -c config.yaml -d sales-analytics # Specific domain
constat test -c config.yaml --tags smoke       # Filter by tag
constat test -c config.yaml --e2e              # Include end-to-end (LLM)
```

## REPL Commands

### Session and Navigation

| Command | Description |
|---------|-------------|
| `/help`, `/h` | Show all commands |
| `/quit`, `/q` | Exit |
| `/reset` | Clear session state and start fresh |
| `/redo [instruction]` | Retry last query (optionally with modifications) |
| `/user [name]` | Show or set current user |

### Data Inspection

| Command | Description |
|---------|-------------|
| `/tables` | List tables in session datastore |
| `/show <table>` | Show table contents |
| `/query <sql>` | Run SQL query on datastore |
| `/export <table> [file]` | Export table to CSV or XLSX |
| `/code [step]` | Show generated code (all or specific step) |
| `/state` | Show session state |
| `/artifacts [all]` | Show artifacts (use 'all' to include intermediate) |

### Data Sources

| Command | Description |
|---------|-------------|
| `/databases`, `/db` | List configured databases |
| `/apis`, `/api` | List configured APIs |
| `/documents`, `/docs` | List configured documents |
| `/files` | List all data files |
| `/doc <path> [name]` | Add a document to this session |
| `/discover [scope] <query>` | Search data sources (scope: database\|api\|document) |
| `/update`, `/refresh` | Refresh metadata and rebuild cache |

### Facts and Memory

| Command | Description |
|---------|-------------|
| `/facts` | Show cached facts from this session |
| `/remember <fact>` | Persist a session fact across sessions |
| `/forget <name>` | Forget a remembered fact |
| `/correct <text>` | Record a correction for future reference |
| `/learnings` | Show learnings and rules |
| `/compact-learnings` | Promote similar learnings into rules |

### Plans and History

| Command | Description |
|---------|-------------|
| `/save <name>` | Save current plan for replay |
| `/share <name>` | Save plan as shared (all users) |
| `/plans` | List saved plans |
| `/replay <name>` | Replay a saved plan |
| `/history`, `/sessions` | List recent sessions |
| `/resume <id>` | Resume a previous session |
| `/summarize <target>` | Summarize plan\|session\|facts\|<table> |

### Verification

| Command | Description |
|---------|-------------|
| `/reason`, `/audit` | Verify conversation claims with auditable reasoning chain |

### Settings

| Command | Description |
|---------|-------------|
| `/verbose [on\|off]` | Toggle verbose mode |
| `/raw [on\|off]` | Toggle raw output display |
| `/insights [on\|off]` | Toggle insight synthesis |
| `/preferences` | Show current preferences |
| `/context` | Show context size and token usage |
| `/compact` | Compact context to reduce token usage |

## Saved Plans and Replay

- `/save` stores the executed code (not just the plan) for deterministic replay
- `/replay` executes the stored code without regenerating it via LLM
- Relative terms ("today", "last month", "within policy") are evaluated dynamically on each replay
- Explicit values ("January 2006", "above 100 units") are hardcoded as specified

## Session Replay

Any exploratory session can be replayed — the stored scratchpad code is re-executed without LLM codegen. This is useful for demos, testing with updated data, or resuming work after a break. Each query's steps are tracked by `objective_index`, so individual objectives can be replayed independently.

## Interactive Visualizations

Constat generates interactive visualizations saved as HTML files you can open in your browser:

```
> Show me an interactive map of countries using the Euro

Interactive map: /Users/you/.constat/outputs/euro_countries.html
```

**Supported visualization types:**

| Type | Library | Example |
|------|---------|---------|
| Interactive maps | Folium | Geographic data, markers, choropleth maps |
| Interactive charts | Plotly | Bar, line, scatter, pie, treemap, etc. |
| Statistical charts | Altair | Declarative statistical visualizations |
| Static plots | Matplotlib/Seaborn | Traditional Python plotting |

Generated visualizations are:
- Saved to `~/.constat/outputs/` as self-contained HTML files
- Stored as artifacts in the session datastore (for UI display)
- Fully interactive in your browser (zoom, hover, pan)

**Example queries:**
- "Create an interactive map showing customer locations"
- "Show me a bar chart of revenue by region"
- "Visualize the correlation between price and quantity"

## Dashboards

Request a "dashboard" to generate multi-panel visualizations automatically:

```
> Create a dashboard of sales performance

[Generates 2x2 grid with: revenue trend, breakdown by category, top products, KPI summary]
```

Dashboard layouts adapt to data:
- **Time series**: Trend + summary stats (1x2)
- **Categories**: Overview, breakdown, comparison, detail (2x2)
- **KPI-focused**: KPI cards on top, supporting charts below (3x2)

## Brief Mode

Use keywords like "briefly", "tl;dr", "just show" in your query to skip the synthesis step and get raw results faster.
