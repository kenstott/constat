# Web UI Guide

The web UI is Constat's visual interface — best for exploration, real-time streaming, DAG visualization, domain management, and team collaboration.

## Starting the Web UI

```bash
# Start both the API server and web UI
./scripts/dev.sh demo/config.yaml        # macOS / Linux
scripts\dev.bat demo\config.yaml          # Windows

# Or start separately:
constat serve -c config.yaml              # API server on :8000
cd constat-ui && npm install && npm run dev  # UI dev server on :5173
```

Open http://localhost:5173.

## Conversation Panel

The left panel is the primary interaction area.

### Message Flow

1. **Input** — Type a question or command in the input box. Supports autocomplete for `/` commands and `@` mentions.
2. **Planning** — Vera generates a multi-step plan. Shows "Thinking..." animation.
3. **Plan Approval** — Review the plan, edit step goals, delete steps, or reject with feedback.
4. **Step Execution** — Each step shows real-time progress:
   - **Step header**: Step number, assigned agent (`@domain/role`), skill badges, elapsed timer
   - **Action chips (green)**: "Reading source" chips while step executes (shows which tables are being queried)
   - **Action chips (purple)**: "Created table" chips on completion (clickable — jumps to table in artifact panel)
   - **Inline table preview**: When a step creates exactly one table, a compact 5-row preview appears inline
5. **Completion** — Steps collapse to a "Completed N steps" summary with expand/collapse toggle. The final answer renders below with inline artifacts.

### Inline Artifacts

Final answers with `isFinalInsight` display published artifacts inline:
- **Tables**: Sortable preview with column headers and first 5 rows (click to open fullscreen)
- **Markdown artifacts**: Rendered inline with expand button (reports, summaries)
- **Other artifacts**: Banner with title and "View" link

### Follow-Up Pills

When the answer includes "Next Steps" suggestions, they render as clickable pill buttons below the response. Clicking a pill submits it as a follow-up query.

### Domain Context Chips

The bot message header shows active domain badges (e.g., "Sales Analytics", "Hr Reporting") so you know which data domains are in scope.

### Collapsible Groups

- **Step groups**: Collapse to "Completed N steps" when execution finishes. Click to expand individual steps.
- **Standalone outputs**: Results from commands like `/reason` show a collapsible summary with expand/collapse toggle.
- **Individual messages**: Long messages show "Show more" / "Show less" when content exceeds ~5 lines.

### Reasoning Chain Mode

Click the **Reason-Chain** button in the input toolbar (or type `/reason`) to verify conversation claims with an auditable reasoning chain.

- The conversation switches to reason-chain mode with a DAG visualization overlay
- Each fact node shows status: pending → planning → executing → resolved/failed
- Click nodes to inspect derivation details, SQL code, and source data
- Click **Explore** to return to exploratory mode
- Completed chains can be extracted as reusable skills

## Artifact Panel

The right panel is organized into collapsible sections:

### Results
- **Tables**: Sortable columns, star/unstar, row counts, table/view indicator (eye icon for views, table icon for tables)
- **Artifacts**: Versioned outputs (code, markdown, charts) grouped by step
- Step-grouped display with "Inference #N" headers for reasoning chain results

### Sources
- **Databases**: Expandable schema browser (tables → columns)
- **APIs**: GraphQL/OpenAPI schema explorer
- **Documents**: Document list with descriptions
- **Facts**: User-provided and session-extracted facts

### Reasoning
- **Config**: System prompt editor, active agents, active skills
- **Code Log**: Scratchpad (execution narrative), Exploratory Code (per-step), Inference Code (auditable)
- **Glossary**: Terms with definitions, taxonomy tree, relationships, aliases, tags
- **Session Store**: Session metadata browser

### Regression Testing
- **Golden question CRUD**: Add, edit, delete test questions per domain
- **System prompt capture**: Tests capture the active system prompt at creation time. Editable textarea for prompt override.
- **Streaming execution**: SSE-based real-time progress during test runs
- **Per-question selection**: Exclude specific questions from test runs
- **Expected outputs**: Define expected tables with column assertions
- **LLM judge**: End-to-end evaluation with customizable judge prompt

## Clarification Dialog

When Constat detects ambiguity, it presents interactive widgets:

| Widget | Use Case |
|--------|----------|
| **Choice** | Select one option from a list |
| **Ranking** | Order items by preference |
| **Curation** | Include/exclude items from a set |
| **Mapping** | Map values between categories |
| **Tree** | Navigate hierarchical options |
| **Table** | Tabular data selection |
| **Annotation** | Annotate text with labels |

## File Upload

Upload documents directly into a session via the attachment button. Uploaded files are ingested through the document pipeline and become available for queries within that session.

## Keyboard & Toolbar

| Action | Control |
|--------|---------|
| Submit query | Enter |
| New line | Shift+Enter |
| Reason-Chain | Toolbar button or `/reason` |
| Brief mode | Toolbar toggle or "briefly" in query |

## Toast Notifications

Transient notifications appear for async operations (session sharing, background tasks). They auto-dismiss after a few seconds.

## Deep Linking

URL-based navigation for artifact panel sections:

- `/db/{dbName}/{tableName}` — database table preview
- `/apis/{apiName}` — API schema
- `/doc/{documentName}` — document viewer
- `/glossary/{termName}` — glossary term detail
- `/s/{sessionId}` — session restoration (shareable)

## Authentication

- **Firebase auth**: JWT-based with role-based visibility and write permissions
- **Local auth**: Username/password with scrypt hashing
- **Disabled mode**: All permissions granted (development)

## Tech Stack

React 18, TypeScript, Vite, Tailwind CSS, Zustand, React Query, D3 + d3-dag, Plotly.js, Headless UI + Heroicons, Vitest + Testing Library
