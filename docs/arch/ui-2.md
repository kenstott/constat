# UI Phase 2 — PromptQL Feature Parity & Gaps

Gap analysis against PromptQL demo UI. Identifies what to add, what's superseded, and what's irrelevant.

## Legend

- **ADD** — Missing, worth implementing
- **SUPERSEDED** — Constat already handles this differently/better
- **IRRELEVANT** — Doesn't apply to Constat's architecture
- **PARTIAL** — Exists but incomplete

---

## 1. Conversation & Messaging

### 1a. Threaded Channels (IRRELEVANT)
PromptQL uses Slack-style channels with multiple threads per workspace. Constat uses session-per-conversation with domain scoping. Multi-user collaboration is a future concern, not a current gap — and when it arrives, session sharing is the model, not channels.

### 1b. @mention Users (ADD — Phase 3)
Users can tag collaborators into the conversation. Requires multi-user sessions first. Low priority.

### 1c. Collapsible Tool/Step Cards (PARTIAL)
PromptQL shows collapsible cards per tool call with action verb labels (`Reading [table]`, `Created [artifact]`). Constat has collapsible step groups but lacks:
- **Action verb labels** — step items show generic content, not `Reading [Settlement Lifecycle]`
- **Colored resource chips** — green for data reads, purple for created artifacts
- **Source attribution per step** — which data source/document was consulted

Plan: Add step action labels + resource chips to `BotMessageGroup` / `MessageBubble`.

### 1d. Streaming Content with Partial Results (PARTIAL)
PromptQL streams partial markdown during generation. Constat streams step events and final output but doesn't show partial markdown tokens during generation. Low priority — step-level streaming is sufficient.

---

## 2. Artifacts & Data Display

### 2a. Inline Artifact Tables (ADD)
PromptQL embeds compact tables (first 5 rows) directly in chat messages with a "View" link to expand in artifact panel. Constat shows table references as chips/links but doesn't render inline previews.

Plan: When a step produces a table artifact, render a 5-row preview inline in the message. Click opens in ArtifactPanel.

### 2b. Artifact Update Indicator (ADD)
When a table is regenerated after follow-up, PromptQL shows "(Updated)" suffix. Constat has `isSuperseded` flag on steps but doesn't surface update indicators on artifacts.

Plan: Track artifact version in `artifactStore`. Show "(Updated)" badge on re-created artifacts.

### 2c. View Changelog / Diff (ADD)
After artifact update, PromptQL shows "View changelog (7 lines)" expandable diff. Constat has no artifact diffing.

Plan: Store previous artifact content hash. On update, compute diff and offer expandable changelog link.

### 2d. Multi-Source Indicators (SUPERSEDED)
PromptQL shows which data connections contributed to an answer. Constat's domain model already scopes data sources per domain, and the proof DAG shows exact premise-to-inference lineage. The DAG is strictly more informative than a flat "sources used" badge.

### 2e. Email Composition (ADD — Low Priority)
PromptQL renders email drafts as a styled card with To/CC/Subject/Body and Send/Save buttons. Nice UX but not core to Constat's analytical mission. Add when collaboration features land.

---

## 3. Knowledge & Learning

### 3a. Wiki Pages / Knowledge Chips (SUPERSEDED)
PromptQL shows "Wiki pages: [Settlement Lifecycle] [SWIFT Message Standards]" as response header chips. These are flat text pages users manually edit.

Constat supersedes this with:
- **Domain documents** — structured per-domain, versioned, auto-discovered from config
- **Glossary** — cross-domain terms with relationships, entity resolution, aliases
- **Learnings/corrections** — tracked with merge history (corrections list shows what was updated and when)

The wiki is a manual, flat alternative to Constat's structured knowledge graph. No need to replicate.

### 3b. "Taught" Indicator (SUPERSEDED)
PromptQL shows "Pax S. taught PromptQL: the settlement_events_v3 table is the canonical source..." when a correction is applied.

Constat's corrections/learnings system already tracks who corrected what and when. Each learning has a `corrections` list showing the merge history. The equivalent UI indicator is showing the correction as accepted + merged into the learning — which the session already does inline.

### 3c. Rule/Learning Timestamps (SUPERSEDED)
PromptQL shows when a wiki annotation was last updated. Constat's learnings have a corrections list with timestamps for each correction merged into the rule. The correction trail is strictly more informative than a single "last updated" timestamp.

### 3d. @vera Teach Command (PARTIAL)
PromptQL uses `@vera <fact>` to teach. Constat uses `/learn <fact>` command. Functionally equivalent. Could alias `@vera` as syntactic sugar but `/learn` is fine.

---

## 4. Execution & Approval

### 4a. Approval Workflows (ADD)
PromptQL has explicit approve/reject flows for certain actions. Constat has `PlanApprovalDialog` and `ClarificationDialog` for plan approval and clarifications, but lacks:
- **Per-step approval** in reason-chain mode (currently auto-executes)
- **Approval history** — who approved what

Plan: In reason-chain mode, add optional step-by-step approval before execution. Track approvals in proof metadata.

### 4b. Scheduling / Recurring Queries (ADD — Phase 3)
PromptQL supports scheduled/recurring queries. Not in scope for v1 but architecturally straightforward — a cron trigger that re-runs a skill's `run_proof()`.

### 4c. Delegation to Other Users (IRRELEVANT — Phase 3)
Requires multi-user. Deferred.

---

## 5. Navigation & Layout

### 5a. Domain Context Chips in Response Header (ADD)
Show which domain documents/sources were consulted as clickable chips in the bot response header. From Phase 2 plan (item 2b in ui.md). Still worth adding.

Plan: Bot response header shows `[doc_name]` `[table_name]` chips color-coded by type. Click opens in ArtifactPanel.

### 5b. Input Bar Attach Button (ADD)
Paperclip button to attach/browse data sources. From Phase 2 plan (item 2a). Still worth adding for quick source selection.

### 5c. Fullscreen Artifact Modal (EXISTS)
Already implemented as `FullscreenArtifactModal.tsx`.

### 5d. Hamburger Menu / Settings (EXISTS)
Already implemented as `HamburgerMenu.tsx`.

---

## 6. Constat-Unique (No PromptQL Equivalent)

These features have no PromptQL counterpart and represent competitive advantages:

| Feature | Status |
|---------|--------|
| Reason-chain mode (DAG-dominant UI) | Implemented |
| Proof DAG visualization | Implemented |
| Skill creation from proof | Implemented |
| Test creation from proof | Implemented |
| Domain DAG (hierarchical domain scoping) | Implemented |
| Glossary with entity resolution + aliases | Implemented |
| Cross-domain relationship extraction | Implemented |
| Regression test panel | Implemented |
| Interactive widgets (ranking, curation, mapping, annotation, tree) | Implemented |
| Plan approval with step editing/deletion | Implemented |
| Flag/report on messages | Implemented |

---

## Implementation Priority

### P0 — Next Sprint
1. **Step action labels + resource chips** (1c) — `BotMessageGroup.tsx`, `MessageBubble.tsx`
2. **Inline artifact table previews** (2a) — `BotMessageGroup.tsx`, new `InlineTablePreview` component
3. **Domain context chips in response header** (5a) — `BotMessageGroup.tsx`

### P1 — Following Sprint
4. **Artifact update indicator** (2b) — `artifactStore.ts`, `ArtifactAccordion.tsx`
5. **View changelog** (2c) — `artifactStore.ts`, new `ArtifactDiff` component
6. **Input bar attach button** (5b) — `AutocompleteInput.tsx`

### P2 — Future
7. **Per-step approval in reason-chain** (4a) — `ReasonChainCommandStrip.tsx`, `proofStore.ts`
8. **Scheduled queries** (4b) — backend + minimal UI trigger
9. **@mention / multi-user** (1b, 4c) — requires auth + session sharing infrastructure

## Files to Modify

| Priority | File | Changes |
|----------|------|---------|
| P0 | `BotMessageGroup.tsx` | Action labels, resource chips, context chips, inline table preview |
| P0 | `MessageBubble.tsx` | Step action verb + colored chip rendering |
| P0 | `sessionStore.ts` | Track step-level resource metadata (which source was read) |
| P1 | `artifactStore.ts` | Artifact versioning, diff storage |
| P1 | `ArtifactAccordion.tsx` | "(Updated)" badge, changelog link |
| P1 | `AutocompleteInput.tsx` | Attach button, source picker |
| P2 | `ReasonChainCommandStrip.tsx` | Step approval flow |
| P2 | `proofStore.ts` | Approval metadata per node |
