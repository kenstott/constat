# Refactored Conversation Flow Architecture

## Overview

This document describes the simplified state model for controlling conversation flow. The goal is to reduce complexity while preserving expressiveness.

---

## Global Mode (Session-Level)

Two modes that affect **how** work is done:

| Mode | Description | Planning Style | Fact Handling |
|------|-------------|----------------|---------------|
| `proof` | Formal verification with provenance | Complete e2e plan, standalone | Each plan is self-contained |
| `exploratory` | Iterative analysis | Incremental, builds on prior work | Facts accumulate across plans in session |

### Mode Selection
- Explicit: user command (`/proof`, `/explore`)
- Inferred: keyword analysis on first task in session
- Sticky: mode persists until explicitly changed

### Mode Behavior Differences

**Proof Mode:**
- Plans must be complete and self-contained
- Every conclusion has a provenance chain
- Output is auditable/defensible
- No implicit dependencies on prior session state

**Exploratory Mode:**
- Plans can reference facts/data from previous plans
- Iterative refinement encouraged
- Session builds up a working context
- Faster iteration, less formal

---

## Turn Intent (Per-Message Classification)

### Primary Intent

Four values that determine the **code path**:

| Intent | Description | Requires Approval | Routes To |
|--------|-------------|-------------------|-----------|
| `query` | Answer from knowledge or current context | No | Query handler |
| `plan_new` | Start planning a new task | Yes (before execution) | Planning flow |
| `plan_continue` | Refine or extend the active plan | No (still planning) | Planning flow |
| `control` | System/session commands | No | REPL command handlers |

### Sub-Intent

Optional refinement that **informs behavior** within the primary intent handler:

```
query:
  - detail      # drill down, explain further
  - provenance  # show proof chain / how we got here
  - summary     # condense results
  - lookup      # simple fact retrieval
  - (default)   # general answer

plan_new:
  - compare     # evaluate alternatives
  - predict     # what-if / forecast
  - (default)   # standard new task

plan_continue:
  - (no sub-intents - user's message provides context for replanning)

control:
  - mode_switch # /proof, /explore - change execution mode
  - reset       # /reset - clear session state
  - redo_cmd    # /redo - re-execute last plan
  - help        # /help - show available commands
  - status      # /status - show current state
  - exit        # /exit, /quit - end session
  - (default)   # other REPL commands
```

### Intent Data Structure

```python
class PrimaryIntent(Enum):
    QUERY = "query"
    PLAN_NEW = "plan_new"
    PLAN_CONTINUE = "plan_continue"
    CONTROL = "control"

class SubIntent(Enum):
    # Query sub-intents
    DETAIL = "detail"
    PROVENANCE = "provenance"
    SUMMARY = "summary"
    LOOKUP = "lookup"

    # Plan new sub-intents
    COMPARE = "compare"
    PREDICT = "predict"

    # Plan continue: no sub-intents (user's message is the context)

    # Control sub-intents (session management)
    MODE_SWITCH = "mode_switch"
    RESET = "reset"
    REDO_CMD = "redo_cmd"
    HELP = "help"
    STATUS = "status"
    EXIT = "exit"

    # Control sub-intents (execution management)
    CANCEL = "cancel"
    REPLAN = "replan"

@dataclass
class TurnIntent:
    primary: PrimaryIntent
    sub: Optional[SubIntent] = None
    target: Optional[str] = None  # what to modify, drill into, etc.
```

---

## Phase (Task Lifecycle State)

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌──────┐    plan_new    ┌──────────┐                               │
│  │ idle │ ─────────────> │ planning │ <─────────────────┐           │
│  └──────┘                └──────────┘                   │           │
│      ^                        │                         │           │
│      │                        │ plan ready              │           │
│      │                        v                         │           │
│      │               ┌─────────────────┐                │           │
│      │               │ awaiting_approval│               │           │
│      │               └─────────────────┘                │           │
│      │                   │         │                    │           │
│      │          approve  │         │ reject/suggest     │           │
│      │                   v         └────────────────────┘           │
│      │              ┌───────────┐                                   │
│      │              │ executing │ ──────────┐                       │
│      │              └───────────┘           │                       │
│      │                   │                  │ fails                 │
│      │         complete  │                  v                       │
│      │                   │             ┌────────┐                   │
│      │                   │             │ failed │                   │
│      │                   │             └────────┘                   │
│      │                   │              │  │  │                     │
│      │                   │        retry │  │  │ replan              │
│      │                   │   ┌──────────┘  │  └─────────────────┐   │
│      │                   │   │             │ abandon             │   │
│      │                   │   │     ┌───────┘                     │   │
│      └───────────────────┴───│─────┘                             │   │
│                              │                                   │   │
│                              └──> (back to executing) ───────────│   │
│                                                                  │   │
│                                         (back to planning) <─────┘   │
│                                                                      │
│  * query intent can occur at ANY phase without changing it           │
│  * plan_continue returns to planning phase                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Phase Transitions

| From | Trigger | To |
|------|---------|-----|
| `idle` | `plan_new` intent | `planning` |
| `planning` | plan generation complete | `awaiting_approval` |
| `awaiting_approval` | user approves | `executing` |
| `awaiting_approval` | user rejects/suggests | `planning` |
| `executing` | execution complete | `idle` |
| `executing` | execution fails | `failed` |
| `failed` | user says retry | `executing` (re-run same plan) |
| `failed` | user says replan/modify | `planning` |
| `failed` | user abandons | `idle` |
| any | `plan_continue` intent | `planning` |
| any | `query` intent | (no change) |

**Failed State**: When execution fails, the system presents suggestions:
- "Try again" → retry execution (for transient/probabilistic failures)
- "Modify the plan" → return to planning with failure context
- "Start over" → return to idle

---

## Combined State

```python
class Phase(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    FAILED = "failed"

@dataclass
class ConversationState:
    mode: Mode                          # proof | exploratory
    phase: Phase                        # idle | planning | awaiting_approval | executing | failed
    active_plan: Optional[Plan]         # current plan if any
    session_facts: FactStore            # accumulated facts (used in exploratory)
    failure_context: Optional[str]      # error details when phase == FAILED

    def can_execute(self) -> bool:
        return self.phase == Phase.AWAITING_APPROVAL

    def can_retry(self) -> bool:
        return self.phase == Phase.FAILED

    def is_planning(self) -> bool:
        return self.phase in (Phase.PLANNING, Phase.AWAITING_APPROVAL)
```

---

## Turn Processing Flow

```
User Input
    │
    ├── starts with "/" ─────────> Direct REPL command dispatch (fast path)
    │
    └── natural language
            │
            v
      ┌─────────────────────────┐
      │ Classify Turn Intent    │
      │ (primary + sub + target)│
      └─────────────────────────┘
            │
            ├─── control ────────────> Map sub-intent to REPL command
            │                          - mode_switch → change mode
            │                          - reset → clear session
            │                          - etc.
            │
            ├─── query ──────────────> Answer immediately (no phase change)
            │                          - Use session_facts if available
            │                          - Use doc search if needed
            │                          - Use LLM knowledge as fallback
            │
            ├─── plan_new ───────────> Start new plan
            │                          - In proof mode: fresh context
            │                          - In exploratory mode: inherit session_facts
            │                          - Transition to planning phase
            │
            └─── plan_continue ──────> Modify active plan
                                       - Apply sub-intent (steer/modify/extend/etc.)
                                       - Regenerate affected steps
                                       - Stay in/return to planning phase
```

---

## Intent Resolution

### Approach: Hierarchical Exemplars with LLM Fallback

Intent classification uses a two-tier approach: fast exemplar matching for common cases, LLM fallback for ambiguous input.

### Resolution Flow

```
User Input (natural language)
    │
    v
┌────────────────────────────┐
│ Primary Intent Match       │  ← small exemplar set (~30-40 total across 4 intents)
│ (embedding similarity)     │
└────────────────────────────┘
    │
    ├── high confidence (>0.85) ──> use match, proceed to sub-intent
    │
    └── low confidence (<0.85) ───> LLM classification
                                        │
                                        v
                                   (optionally add to exemplars for future)
```

Then for sub-intent (scoped to the matched primary):

```
Primary = plan_continue
    │
    v
┌─────────────────────────────┐
│ Sub-Intent Match            │  ← exemplars scoped to plan_continue (~10-15)
│ (embedding similarity)      │
└─────────────────────────────┘
    │
    ├── high confidence ──> use match
    │
    └── low confidence ───> use default sub-intent for that primary
```

### Benefits of Hierarchical Approach

1. **Smaller search spaces** - 4 exemplar sets for primary, then scoped sets for sub-intent
2. **Natural grouping** - "extend this" vs "change the approach" live in different buckets
3. **Tunable per level** - tighter thresholds for primary, looser for sub
4. **Easy to grow** - add exemplars to specific categories without polluting others
5. **Debuggable** - "matched primary=plan_continue at 0.87, sub=steer at 0.72"

### Exemplar Structure

```yaml
primary_intents:
  query:
    - "what does this mean"
    - "explain the results"
    - "how did you get that"
    - "show me the proof"
    - "what's the current status"
  plan_new:
    - "analyze the sales data"
    - "build a dashboard for"
    - "verify that revenue"
    - "compare these two options"
  plan_continue:
    - "actually, also include"
    - "change that to use"
    - "narrow it down to"
    - "run it again with"
    - "use a different approach"
  control:
    - "switch to proof mode"
    - "start over"
    - "let's reset"
    - "help me understand commands"
    - "quit"

sub_intents:
  query:
    detail:
      - "drill down into"
      - "explain more about"
      - "what specifically"
    provenance:
      - "how did you determine"
      - "show me the proof"
      - "where did that come from"
    summary:
      - "summarize the findings"
      - "give me the key points"
    lookup:
      - "what is the value of"
      - "what did we find for"

  # plan_continue: no sub-intents (user's message is the context)

  control:
    mode_switch:
      - "switch to proof mode"
      - "let's be more exploratory"
      - "use auditable mode"
    reset:
      - "start over"
      - "clear everything"
      - "reset the session"
    help:
      - "what can you do"
      - "show me commands"
      - "how do I"
    status:
      - "where are we"
      - "what's the current plan"
      - "show current state"
```

### Embedding Model

Use `BAAI/bge-large-en-v1.5` for intent classification:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-en-v1.5')

# Encode exemplars (do once, cache)
exemplar_embeddings = model.encode(exemplars)

# Encode user input
input_embedding = model.encode(user_input)

# Cosine similarity for matching
from sklearn.metrics.pairwise import cosine_similarity
scores = cosine_similarity([input_embedding], exemplar_embeddings)[0]
```

**Why this model:**
- Top performer on MTEB semantic similarity benchmarks
- 1024 dimensions provides high resolution for fine-grained intent distinctions
- Well-suited for short text (our exemplars are typically <10 words)
- Widely used in production, well-tested

**Alternatives if needed:**
| Model | Trade-off |
|-------|-----------|
| `bge-base-en-v1.5` | Smaller (440MB), 95% quality |
| `gte-large-en-v1.5` | Slightly higher MTEB, similar size |
| `bge-m3` | Better multilingual, larger (2.2GB) |

### LLM Fallback Prompt

When exemplar confidence is low, use a structured LLM call:

```
Given the user input and conversation context, classify the intent.

User input: "{input}"
Current phase: {phase}
Has active plan: {has_plan}

Respond with:
PRIMARY: query | plan_new | plan_continue | control
SUB: {valid sub-intents for that primary} | none
TARGET: {extracted target if applicable} | none
CONFIDENCE: high | medium | low
```

---

## Mapping from Current Code

### Current State (as of implementation)

**Files that exist:**
- `constat/execution/mode.py` - Contains `ExecutionMode` enum (KNOWLEDGE, EXPLORATORY, AUDITABLE), `PlanApproval` enum, mode selection logic
- `constat/execution/intent.py` - Contains `FollowUpIntent` enum (20 intents), `DetectedIntent`, `IntentClassification` classes
- `constat/execution/repl_matcher.py` - Token-based REPL command matching (Jaccard + overlap), `ReplCommand` dataclass

**Key differences from plan:**
- Current system has 3 modes, plan proposes 2 (removing KNOWLEDGE)
- Current system has 20 granular intents, plan proposes 4 primary + ~10 sub-intents
- No embedding-based intent classification exists (plan proposes BAAI/bge-large-en-v1.5)
- No Phase enum or ConversationState tracking exists
- `repl_matcher.py` uses token similarity, not embeddings

### Modes
| Current | New |
|---------|-----|
| `AUDITABLE` | `proof` |
| `EXPLORATORY` | `exploratory` |
| `KNOWLEDGE` | (removed - behavior moves to `query` handling) |

### Intents (Current → New)

| Current Intent | New Primary | New Sub |
|----------------|-------------|---------|
| `NEW_QUESTION` | `plan_new` | - |
| `STEER_PLAN` | `plan_continue` | - |
| `MODIFY_FACT` | `plan_continue` | - |
| `EXTEND` | `plan_continue` | - |
| `REFINE_SCOPE` | `plan_continue` | - |
| `REDO` | `plan_continue` | - |
| `DRILL_DOWN` | `query` | `detail` |
| `PROVENANCE` | `query` | `provenance` |
| `SUMMARIZE` | `query` | `summary` |
| `LOOKUP` | `query` | `lookup` |
| `CHALLENGE` | `query` | `detail` |
| `COMPARE` | `plan_new` | `compare` |
| `PREDICT` | `plan_new` | `predict` |
| `MODE_SWITCH` | `control` | `mode_switch` |
| `RESET` | `control` | `reset` |
| `EXPORT` | `plan_new` | - |
| `CREATE_ARTIFACT` | `plan_new` | - |
| `TRIGGER_ACTION` | `plan_new` | - |
| `ALERT` | `plan_new` | - |
| `QUERY` | `plan_new` | - |

### Files to Modify

1. **`constat/execution/mode.py`**
   - Replace `ExecutionMode` enum (3 → 2 values)
   - Add `Mode`, `Phase`, `PrimaryIntent`, `SubIntent` enums
   - Add `TurnIntent` and `ConversationState` dataclasses
   - Update `suggest_mode()` to only choose proof/exploratory
   - Remove `KNOWLEDGE_KEYWORDS` and related logic
   - Update `MODE_SYSTEM_PROMPTS` to only have PROOF and EXPLORATORY

2. **`constat/execution/intent.py`**
   - Retain `FollowUpIntent` temporarily for backward compatibility during migration
   - Add mapping functions from old intents to new primary/sub structure
   - Or: Replace entirely with new intent structure

3. **`constat/execution/repl_matcher.py`** (exists)
   - Update commands to use new mode names (`/proof` instead of `/mode auditable`)
   - Remove `/mode knowledge` command
   - Consider migrating to embedding-based matching per plan

4. **NEW: `constat/execution/intent_classifier.py`**
   - Implement `IntentClassifier` class with embedding model
   - Create `exemplars.yaml` with primary + sub-intent examples
   - Hierarchical matching (primary threshold: 0.80, sub threshold: 0.65)

5. **`constat/session.py`**
   - Add `ConversationState` tracking
   - Add `_intent_classifier` instance
   - Refactor `_analyze_question()` to return `TurnIntent`
   - Add handler methods: `_handle_query_intent()`, `_handle_plan_new_intent()`, etc.
   - Simplify intent handling to 4 primary branches
   - Move KNOWLEDGE mode logic into `_handle_query_intent()`

6. **`constat/execution/planner.py`**
   - Update to work with new mode names
   - Adjust context handling for proof vs exploratory

7. **`constat/feedback.py`** / **`constat/repl.py`**
   - Update display for new mode names
   - Adjust approval flow for phase model
   - Add `StatusLine` class implementation

---

## Status Line UI

A persistent status line at the bottom of the terminal showing current conversation state.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ [MODE] phase    context info                              [indicators]  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Examples by Phase

**Idle:**
```
[PROOF] idle
[EXPLORE] idle
```

**Planning:**
```
[PROOF] planning ⠋ "Analyze revenue by region"
[EXPLORE] planning ⠋ "Compare Q3 vs Q4 metrics"
```

**Awaiting Approval:**
```
[PROOF] awaiting_approval    plan: "Revenue by region" (3 steps)    [y/n/suggest]
[EXPLORE] awaiting_approval    plan: "Q3 vs Q4 comparison" (5 steps)    [y/n/suggest]
```

**Executing:**
```
[PROOF] executing ⠋ step 2/5 "Loading sales data"
[EXPLORE] executing ⠋ step 3/4 "Calculating metrics"    [queued: 1]
```

**Failed:**
```
[PROOF] failed ✗ step 3    "Database connection error"    [retry/replan/abandon]
[EXPLORE] failed ✗ step 2    "API timeout"    [retry/replan/abandon]
```

### Status Line Elements

| Element | Description | When Shown |
|---------|-------------|------------|
| Mode | `[PROOF]` or `[EXPLORE]` | Always |
| Phase | `idle`, `planning`, `awaiting_approval`, `executing`, `failed` | Always |
| Spinner | `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` animation | During planning/executing |
| Plan name | Truncated problem statement | When plan exists |
| Step progress | `step N/M "description"` | During executing |
| Queue indicator | `[queued: N]` | When plan_new queued during execution |
| Action hints | `[y/n/suggest]`, `[retry/replan/abandon]` | Awaiting approval, failed |
| Error summary | Brief error message | Failed phase |

### Implementation Notes

- Use Rich library's `Layout` with fixed bottom panel
- Status line should not scroll with conversation
- Update in place (no flicker)
- Colors:
  - `[PROOF]` = yellow/bold
  - `[EXPLORE]` = cyan
  - `failed` = red
  - `executing` = green
  - Spinner = dim

### Integration with REPL

```python
class StatusLine:
    def __init__(self):
        self._mode: Mode = Mode.EXPLORATORY
        self._phase: Phase = Phase.IDLE
        self._plan_name: str | None = None
        self._step_current: int = 0
        self._step_total: int = 0
        self._step_description: str = ""
        self._queue_count: int = 0
        self._error_message: str | None = None

    def render(self) -> str:
        """Render current status line."""
        pass

    def update(self, state: ConversationState) -> None:
        """Update from conversation state."""
        pass
```

---

## Background Execution & Natural Language Control

### Current Behavior (Blocking)
- Execution blocks user input
- User must interrupt (Ctrl+C) to stop
- Feedback only possible after interrupt

### Proposed Behavior (Non-Blocking)
- Execution runs in background thread/task
- User can type at any time during execution
- Natural language interpreted to control execution

### Execution Control Intents

Add sub-intents to `control` for execution management:

```yaml
control:
  # ... existing sub-intents ...
  cancel:         # stop execution entirely
    - "stop"
    - "cancel"
    - "never mind"
  replan:         # stop and revise the plan
    - "actually, let's change"
    - "wait, I want to modify"
    - "stop, different approach"
```

### Intent Handling During Execution

```
User Input (during executing phase)
    │
    v
┌─────────────────────────┐
│ Classify Turn Intent    │
└─────────────────────────┘
    │
    ├─── control.cancel ────────> Stop execution, return to idle
    │                             (discard in-progress work)
    │
    ├─── control.replan ────────> Stop execution, return to planning
    │                             with user's modification
    │
    ├─── query ─────────────────> Answer in parallel (execution continues)
    │
    ├─── plan_continue ─────────> Stop execution, return to planning
    │                             (implicit replan with user's input)
    │
    └─── plan_new ──────────────> QUEUE - wait for execution to complete,
                                  then start new plan
                                  (preserves current work)
```

**Rationale**: `plan_new` queues rather than interrupts because the user has a new idea but hasn't asked to abandon the current work. Completed execution results may be useful context for the new plan. If user wants to abandon, they can use `control.cancel` first.

**Queue Behavior**:
- `plan_new`: Queue 1, latest wins (new request replaces any queued request)
- `control` (mode switch, etc.): Queue in order, process after execution
- `query`: No queue, answered in parallel immediately

### Implementation Considerations

1. **Thread safety** - Execution state must be thread-safe for concurrent access
2. **Graceful cancellation** - Steps should check for cancellation between operations
3. **Partial results** - Decide what to do with completed steps when cancelled
4. **Progress visibility** - User needs to see execution progress while typing
5. **Interrupt as shortcut** - Ctrl+C remains supported as a keyboard shortcut for `control.cancel`, equivalent to typing "stop"

### UI Flow Example

```
User: analyze sales by region
Assistant: [shows plan, awaits approval]
User: looks good
Assistant: [starts executing in background]
         ⠋ Step 1: Loading sales data...

User: wait, only include Q4            <- user types while executing
Assistant: [stops execution]
         ✓ Step 1: Loading sales data (completed)
         ○ Step 2: (cancelled)

         I'll revise the plan to filter for Q4 only.
         [shows updated plan]
```

---

## Resolved Decisions

| Question | Decision |
|----------|----------|
| `plan_new` queue depth | Queue 1, latest wins (new request replaces queued) |
| Mode change during execution | Queue (like other control intents) |
| Approval timeout | None - wait indefinitely for user |
| Failed execution recovery | Return to prompt with suggestions (replan, retry, modify). User chooses next action. |
| Embedding model | `BAAI/bge-large-en-v1.5` (1024 dims, ~1.3GB, MTEB 64.23) |
| Partial execution state | Preserve facts from completed steps when cancelled |
| Intent ambiguity | Ask user to clarify when intent is unclear |
| Session persistence | Ephemeral - `ConversationState` not persisted across restarts |
| Exemplar storage | YAML file |
| Confidence thresholds | Primary: 0.80, Sub: 0.65, with logging and per-intent tuning |
| Exemplar curation | Manual review required before adding new exemplars |
| Pause semantics | No pause/resume - just cancel + replan |
| Multi-intent messages | Split on universal delimiters (`.`, `;`), classify each, latest wins on conflict |
| Fact scope in proof mode | Fact-name-based reuse: planner instructed to reuse original fact names, execution uses cached value if name matches, planner can force re-resolution by renaming |

---

## Open Questions

None - all resolved.

---

## Fact Reuse in Proof Mode (Resolved)

Proof mode uses **fact-name-based caching**:

```
Replan in Proof Mode
    │
    v
┌─────────────────────────────────┐
│ Planner instructed to reuse     │
│ original fact names where       │
│ appropriate                     │
└─────────────────────────────────┘
    │
    v
┌─────────────────────────────────┐
│ Execution checks each fact:     │
│ "Is there a cached fact with    │
│  this name?"                    │
└─────────────────────────────────┘
    │
    ├─── yes ──> Use cached value (skip resolution)
    │
    └─── no ───> Resolve fresh
```

**Planner controls reuse:**
- Same fact name → reuse cached value
- Different fact name → force fresh resolution
- Execution can still update a fact if value has changed

**Example:**
```
Original plan:
  - fact: q4_revenue → resolved to $1.2M

Replan (user wants to add comparison):
  - fact: q4_revenue → uses cached $1.2M (same name)
  - fact: q3_revenue → resolves fresh (new fact)
```

**Benefits:**
- Auditability: provenance tracked per-fact
- Efficiency: no redundant resolution
- Flexibility: planner controls what gets refreshed

---

## Multi-Intent Message Handling (Resolved)

Split-then-classify approach for messages with multiple parts:

```
User Input
    │
    v
┌─────────────────────────┐
│ Split on delimiters     │  ← ".", ";"
└─────────────────────────┘
    │
    v
┌─────────────────────────┐
│ Classify each segment   │  ← higher confidence on shorter, focused text
└─────────────────────────┘
    │
    v
┌─────────────────────────┐
│ Resolve conflicts       │  ← latest wins (user self-correcting)
└─────────────────────────┘
    │
    v
┌─────────────────────────┐
│ Execute as sequence     │
└─────────────────────────┘
```

**Example:**
- Input: "analyze sales. wait, I got that wrong. analyze revenue instead."
- Split: ["analyze sales", "wait, I got that wrong", "analyze revenue instead"]
- Classify: [`plan_new(sales)`, `control.cancel`, `plan_new(revenue)`]
- Resolve: `plan_new(revenue)` wins (latest of same type)
- Execute: `plan_new(revenue)`

**Rationale:** Latest wins handles natural self-correction patterns. Users often type stream-of-consciousness and correct themselves mid-message.

---

## Implementation Plan

### Scope Clarification

**What changes:** Control layer around execution (state management, intent routing, phase transitions)

**What stays same:** Core execution logic (parallel scheduler, step execution, fact resolution, DAG parallelization)

| Changes | No Changes |
|---------|------------|
| Phase state management | `parallel_scheduler.execute_plan_sync()` |
| Intent classification & routing | Step execution logic |
| Cancellation hooks between steps | Fact resolution |
| Background execution wrapper | DAG-based parallelization |
| Queue for intents during execution | `StepExecutor`, `TaskRouter` |
| Failure recovery flow | Planner internals |

---

### Files & Complexity

| File | Complexity | Key Changes |
|------|------------|-------------|
| `mode.py` | HIGH | Add enums (Mode, Phase, PrimaryIntent, SubIntent), dataclasses (TurnIntent, ConversationState), rename AUDITABLE→PROOF, remove KNOWLEDGE |
| `session.py` | VERY HIGH | Add ConversationState, replace intent detection with IntentClassifier, add 4 handler methods, phase transitions |
| `intent.py` | HIGH | New IntentClassifier class, exemplar loading, embedding integration, LLM fallback |
| `feedback.py` | MEDIUM | Update mode names, simplify approval flow, remove MODE_SWITCH |
| `repl.py` | MEDIUM | Add /proof /explore commands, Ctrl+C → cancel, background execution support |
| `planner.py` | LOW | Update mode references, fact-name reuse instructions for proof mode |
| `parallel_scheduler.py` | LOW | Add cancellation flag check between steps |

---

### Implementation Phases

#### Phase 1: Core Data Structures (`mode.py`)
1. [ ] Define `Mode` enum: `PROOF`, `EXPLORATORY`
2. [ ] Define `Phase` enum: `IDLE`, `PLANNING`, `AWAITING_APPROVAL`, `EXECUTING`, `FAILED`
3. [ ] Define `PrimaryIntent` enum: `QUERY`, `PLAN_NEW`, `PLAN_CONTINUE`, `CONTROL`
4. [ ] Define `SubIntent` enum (query: detail/provenance/summary/lookup, plan_new: compare/predict, control: mode_switch/reset/redo_cmd/help/status/exit/cancel/replan)
5. [ ] Define `TurnIntent` dataclass: primary, sub, target
6. [ ] Define `ConversationState` dataclass: mode, phase, active_plan, session_facts, failure_context
7. [ ] Rename/alias `AUDITABLE` → `PROOF` (backward compat if needed)
8. [ ] Remove `KNOWLEDGE` mode, update `suggest_mode()`

#### Phase 2: Intent Resolution (`intent.py` or new file)
9. [ ] Create `exemplars.yaml` with primary + sub-intent examples
10. [ ] Implement `IntentClassifier` class:
    - `classify(user_input, context) → TurnIntent`
    - `_classify_primary(user_input) → (PrimaryIntent, confidence)`
    - `_classify_sub(primary, user_input) → (SubIntent | None, confidence)`
    - `_extract_target(primary, user_input) → str | None`
11. [ ] Load `BAAI/bge-large-en-v1.5` embeddings (lazy load, cache)
12. [ ] Implement hierarchical matching (primary threshold: 0.80, sub threshold: 0.65)
13. [ ] Implement LLM fallback for low-confidence
14. [ ] Add message splitting on `.` `;` for multi-intent
15. [ ] Add logging: "matched primary=X at 0.87, sub=Y at 0.72"

#### Phase 3: Session Integration (`session.py`)
16. [ ] Add `self._conversation_state: ConversationState` to Session
17. [ ] Add `self._intent_classifier: IntentClassifier` to Session
18. [ ] Add `_classify_turn_intent(user_input) → TurnIntent`
19. [ ] Add `_handle_query_intent(sub_intent, context) → answer`
20. [ ] Add `_handle_plan_new_intent(problem, context) → starts planning`
21. [ ] Add `_handle_plan_continue_intent(modification, context) → replans`
22. [ ] Add `_handle_control_intent(sub_intent, context) → executes command`
23. [ ] Add `_apply_phase_transition(trigger) → updates phase`
24. [ ] Refactor `_analyze_question()` - remove intent detection, keep fact extraction
25. [ ] Update main processing to dispatch by primary intent
26. [ ] Move KNOWLEDGE logic into `_handle_query_intent()` (doc search + LLM fallback)

#### Phase 4: Execution Control (`parallel_scheduler.py`, `session.py`)
27. [ ] Add cancellation flag to execution context
28. [ ] Add check between steps: if cancelled, stop and preserve completed facts
29. [ ] Add background execution wrapper (thread or async)
30. [ ] Add intent queue for messages during execution
31. [ ] Implement queue behavior: plan_new queues (latest wins), control queues in order, query parallel

#### Phase 5: REPL & Display (`repl.py`, `feedback.py`)
32. [ ] Add `/proof` command → control.mode_switch(PROOF)
33. [ ] Add `/explore` command → control.mode_switch(EXPLORATORY)
34. [ ] Map Ctrl+C → control.cancel
35. [ ] Update mode display names in feedback.py
36. [ ] Simplify approval flow (remove MODE_SWITCH option)
37. [ ] Implement `StatusLine` class with Rich Layout
38. [ ] Add status line to REPL with fixed bottom panel
39. [ ] Wire status line updates to ConversationState changes
40. [ ] Add failure recovery prompts (retry/replan/abandon)

#### Phase 6: Testing & Validation
41. [ ] Update existing tests for new enums/structures
42. [ ] Add IntentClassifier unit tests with exemplar coverage
43. [ ] Add phase transition tests
44. [ ] Add integration tests for background execution + control
45. [ ] Add multi-intent message parsing tests
46. [ ] Add status line rendering tests

---

## Current Implementation Status

*Last updated: January 2026*

### Summary

| Category | Status | Notes |
|----------|--------|-------|
| Mode enum (3→2) | **NOT STARTED** | `ExecutionMode` still has KNOWLEDGE, EXPLORATORY, AUDITABLE |
| Phase tracking | **NOT STARTED** | No `Phase` enum or `ConversationState` exists |
| Primary/Sub Intent | **NOT STARTED** | Still using 20-value `FollowUpIntent` enum |
| Embedding classifier | **NOT STARTED** | No IntentClassifier or exemplars.yaml |
| REPL matcher | **PARTIAL** | `repl_matcher.py` exists with token-based matching (Jaccard) |
| StatusLine | **NOT STARTED** | No persistent status line UI |
| Background execution | **NOT STARTED** | Execution still blocks user input |

### What Exists Today

1. **`constat/execution/mode.py`**
   - `ExecutionMode` enum: KNOWLEDGE, EXPLORATORY, AUDITABLE (3 values)
   - `PlanApproval` enum: APPROVE, REJECT, SUGGEST, COMMAND, MODE_SWITCH
   - `suggest_mode()` function with keyword matching
   - `MODE_SYSTEM_PROMPTS` dict with all 3 modes
   - `KNOWLEDGE_KEYWORDS`, `AUDITABLE_KEYWORDS`, `EXPLORATORY_KEYWORDS` lists

2. **`constat/execution/intent.py`**
   - `FollowUpIntent` enum: 20 intents (REDO, MODIFY_FACT, STEER_PLAN, DRILL_DOWN, etc.)
   - `DetectedIntent` dataclass
   - `IntentClassification` class with confidence tracking and execution planning
   - Helper sets: `IMPLIES_REDO`, `QUICK_INTENTS`, `EXECUTION_INTENTS`
   - `from_analysis()` function to bridge session analysis to intent classification

3. **`constat/execution/repl_matcher.py`**
   - `ReplCommand` dataclass with exemplars
   - Token-based matching (Jaccard similarity + token overlap ratio)
   - Commands: `/reset`, `/mode exploratory`, `/mode auditable`, `/mode knowledge`, `/provenance`, `/redo`, `/help`
   - `match_repl_command()` with 0.80 threshold

4. **`constat/feedback.py`**
   - `LivePlanExecutionDisplay` class for step-by-step progress
   - Spinner animation frames
   - `detect_mode_switch()` function
   - Rich library integration for terminal UI

### Gap Analysis

| Plan Requirement | Current State | Gap |
|------------------|---------------|-----|
| `Mode` enum (2 values) | `ExecutionMode` (3 values) | Need to remove KNOWLEDGE, rename AUDITABLE→PROOF |
| `Phase` enum (5 values) | Does not exist | Create new enum |
| `PrimaryIntent` enum (4 values) | `FollowUpIntent` (20 values) | Create new simplified enum |
| `SubIntent` enum (~10 values) | Embedded in FollowUpIntent | Extract sub-intent structure |
| `TurnIntent` dataclass | Does not exist | Create new dataclass |
| `ConversationState` dataclass | Does not exist | Create new dataclass |
| `IntentClassifier` with embeddings | LLM-based classification only | Build embedding-based classifier |
| `exemplars.yaml` | Does not exist | Create exemplar file |
| `StatusLine` class | Does not exist | Implement persistent status UI |
| Phase transitions | Implicit in session flow | Make explicit with `_apply_phase_transition()` |
| Background execution | Blocking | Add threading/async wrapper |
| Intent queue | Does not exist | Implement queue for messages during execution |
| Cancellation hooks | Does not exist | Add to parallel_scheduler |

### Recommended Implementation Order

1. **Phase 1: Core Enums** (mode.py) - Foundation for everything else
2. **Phase 2: Intent Classifier** - Can be developed in parallel with minimal dependencies
3. **Phase 3: Session Integration** - Depends on Phase 1 & 2
4. **Phase 4: Execution Control** - Depends on Phase 3
5. **Phase 5: UI/REPL** - Can start in parallel after Phase 1
6. **Phase 6: Testing** - Ongoing throughout

### Migration Strategy

To avoid breaking existing functionality during migration:

1. **Keep backward compatibility aliases** - `ExecutionMode.AUDITABLE` can alias to new `Mode.PROOF`
2. **Add mapping functions** - `FollowUpIntent` → `PrimaryIntent` + `SubIntent` mapping
3. **Feature flag** - Add `USE_NEW_INTENT_SYSTEM` flag to gradually roll out
4. **Dual-write logging** - Log both old and new intent classifications to compare accuracy
5. **Test coverage first** - Ensure existing behavior is well-tested before refactoring
