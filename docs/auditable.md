# Auditable Mode UX Design

**Status:** Design (not implemented)

## Overview

Auditable mode provides verifiable, traceable answers with full provenance. Unlike exploratory mode's linear step execution, auditable mode resolves a graph of interdependent facts.

## User Flow

1. User clicks **proof button** or uses `/prove` command
2. System generates proof plan (propositions + inferences with dependencies)
3. **Plan approval dialog** appears (same pattern as exploratory mode)
4. On approval, **floating DAG panel** shows proof execution in real-time
5. On completion, proof trace saved as **artifact**

## DAG Visualization

Facts have dependencies, so we visualize as a directed acyclic graph (top-down):

```
        ┌─────────────┐
        │ is_vip(C01) │ ← root proposition
        └──────┬──────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│ revenue(C01)│ │ tier(C01)   │
└──────┬──────┘ └──────┬──────┘
       │               │
       ▼               ▼
┌─────────────┐ ┌─────────────┐
│ orders table│ │ customers   │
└─────────────┘ └─────────────┘
```

### Node States

| Symbol | State | Description |
|--------|-------|-------------|
| `○` | pending | Not started |
| `◐` | planning | Generating derivation logic |
| `●` | executing | Running query/inference |
| `✓` | resolved | Value determined with source |
| `✗` | failed | Couldn't resolve |
| `⊘` | blocked | Waiting on dependencies |

### Interactions

- **Click node**: Expand to show details (source, SQL query, confidence, value)
- **Animated edges**: Show data flow as facts resolve
- **Critical path highlight**: Indicate which unresolved facts block completion
- **Collapse/expand**: Handle large DAGs by collapsing subtrees

## UI Placement

The DAG appears in a **floating panel** over the conversation. This allows users to:
- See proof progress in real-time
- Maintain conversation context
- Dismiss or minimize when not needed

## WebSocket Events

| Event | UI Update |
|-------|-----------|
| `fact_start` | Node appears in pending state |
| `fact_planning` | Node shows planning indicator |
| `fact_executing` | Node shows executing indicator |
| `fact_resolved` | Node shows checkmark + value |
| `fact_failed` | Node shows error state |
| `proof_complete` | DAG finalizes, proof artifact created |

## Artifacts

The final proof trace is saved as an artifact containing:
- All resolved facts with values
- Source for each fact (DATABASE, CONFIG, LLM_KNOWLEDGE)
- SQL queries executed
- Confidence levels
- Derivation logic (how facts were combined)

## Open Questions

- Floating panel dimensions and positioning?
- How to handle very wide DAGs (many parallel facts)?
- Export format for proof artifact (JSON, PDF, both)?