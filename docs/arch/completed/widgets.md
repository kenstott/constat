# Clarification Widgets API

Rich input widgets for agent-user clarification flow. Extends the existing clarification mechanism with structured input when the ambiguity class benefits from it.

## Core Principle

Every widget always returns both structured data and freeform text. The user never chooses between them — they use both to whatever degree is useful.

```typescript
interface WidgetResponse<T> {
  structured: T          // current state of the widget, always serialized
  freeform: string       // text input, may be empty
}
```

The agent merges both signals:
1. Apply structured data literally
2. Interpret freeform as additional instruction

Partial widget state is still captured even if the user only types — selection context can confirm or clarify their freeform intent.

## Request Shape

```typescript
interface ClarificationRequest {
  question: string            // always present — the "why"
  widget: ClarificationWidget // what kind of input to render
}
```

The `question` string gives context that makes the widget usable. The widget is the *how*, the question is the *why*.

## Widget Types

```typescript
type ClarificationWidget =
  | ChoiceWidget          // multiple choice (today's AskUserQuestion)
  | CurationWidget        // filterable checklist
  | MappingWidget         // two-column drag/connect
  | RankingWidget         // ordered list
  | AnnotationWidget      // image/screenshot markup
  | TreeWidget            // hierarchy editor
  | TableWidget           // editable grid
```

### ChoiceWidget

What exists today. No changes needed.

```typescript
interface ChoiceWidget {
  type: "choice"
  options: { label: string; description: string; markdown?: string }[]
  multiSelect: boolean
}
```

### CurationWidget

Filterable checklist for large item sets. The entity noise problem is the canonical use case.

```typescript
interface CurationWidget {
  type: "curation"
  items: {
    id: string
    label: string
    metadata?: Record<string, string>  // e.g. { domain: "hr-reporting", type: "concept" }
  }[]
  columns?: string[]          // metadata keys to show as columns
  defaultState: "all-selected" | "none-selected"
  groupBy?: string            // metadata key to group by
  searchable: boolean
}
// Structured return: { kept: string[], removed: string[] }
```

### MappingWidget

Two-column connector for schema mapping, field pairing, API-to-handler routing.

```typescript
interface MappingWidget {
  type: "mapping"
  left: { id: string; label: string; group?: string }[]   // source
  right: { id: string; label: string; group?: string }[]   // target
  allowUnmapped: boolean
  allowManyToOne: boolean
}
// Structured return: { mappings: { left: string; right: string }[] }
```

### RankingWidget

Priority ordering with optional tier grouping.

```typescript
interface RankingWidget {
  type: "ranking"
  items: { id: string; label: string; description?: string }[]
  tiers?: string[]  // optional tier labels like ["Must have", "Nice to have", "Skip"]
}
// Structured return: { ranked: string[] } or { tiers: Record<string, string[]> }
```

### AnnotationWidget

Image markup for "circle what's wrong" interactions.

```typescript
interface AnnotationWidget {
  type: "annotation"
  image: string              // base64 or file path
  tools: ("rectangle" | "circle" | "arrow" | "text")[]
}
// Structured return: { annotations: { tool: string; coords: number[]; text?: string }[] }
```

### TreeWidget

Hierarchy editor for taxonomy, nav structure, module organization.

```typescript
interface TreeWidget {
  type: "tree"
  nodes: {
    id: string
    label: string
    parentId?: string        // null = root
    metadata?: Record<string, string>
  }[]
  allowReparent: boolean
  allowDelete: boolean
  allowRename: boolean
  allowAdd: boolean
}
// Structured return: full tree with edits applied
```

### TableWidget

Editable spreadsheet-style grid for config tuning, threshold setting, rule editing.

```typescript
interface TableWidget {
  type: "table"
  columns: {
    key: string
    label: string
    editable: boolean
    type: "text" | "boolean" | "select"
    options?: string[]
  }[]
  rows: Record<string, any>[]
}
// Structured return: { rows: Record<string, any>[] }
```

## Ambiguity Class to Widget Mapping

The planner already knows what class of clarification it needs. The type of ambiguity determines the widget:

| Ambiguity class | Widget | Example |
|---|---|---|
| "Which of these?" | `choice` | "JWT or session auth?" |
| "Which of these N items are valid?" | `curation` | Entity whitelist, dependency audit |
| "How do A's map to B's?" | `mapping` | Source-to-target columns, API-to-handler routing |
| "What order/priority?" | `ranking` | Feature backlog, migration ordering |
| "Where exactly in the UI?" | `annotation` | Bug reports, layout feedback |
| "How should this hierarchy look?" | `tree` | Taxonomy, nav structure, module organization |
| "Review/edit these values" | `table` | Config tuning, threshold setting, rule editing |

## Rendering Strategy

**Declarative, not imperative.** The skill says *what* input it needs, not *how* to render it. The host IDE (VS Code, terminal, web) picks the best rendering for its environment.

**Progressive enhancement.** Every widget has a text fallback. If the host doesn't support `tree`, it renders as indented text. The structured response is an optimization, not a requirement. The freeform text input is always available.

**The widget is a hint, not a constraint.** The user can always dismiss any widget and just type. "I don't want any of these — the real problem is we're crawling Wikipedia pages that aren't relevant to HR" is a response no widget can capture, but it's the most useful clarification possible.

## Why Both Fields Always

The user checks 40 boxes in the curation list, *then* types "also remove anything that looks like a journal name." The agent gets both signals — checkboxes give precision, text gives intent the widget couldn't express.

Even partial widget interaction is useful context. A user who selected KPMG and Deloitte then typed "keep only the big 4 accounting firms" — the structured data confirms the freeform intent.

No information is lost. The user never has to choose between "use the widget" or "type instead."
