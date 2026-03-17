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

## Features

- **Real-time streaming** via WebSocket — see plan generation, step execution, and synthesis as they happen
- **Plan approval dialog** before execution — review, edit, or reject plans
- **Clarification dialog** with interactive widgets (choice, ranking, curation, mapping, tree, table, annotation)
- **Artifact panel** with tabs for tables, charts, code, scratchpad (execution narrative), and glossary
- **Reasoning Chain DAG visualization** (D3-based directed acyclic graph) with skill extraction
- **Domain management** with tier promotion (user → shared → system)
- **Golden question regression testing UI** with streaming progress (SSE)
- **Deep linking** for artifact panel navigation
- **Firebase authentication** with role-based visibility/write permissions
- **Responsive layout** with collapsible panels

## Tech Stack

React 18, TypeScript, Vite, Tailwind CSS, Zustand, React Query, D3 + d3-dag, Plotly.js, Headless UI + Heroicons
