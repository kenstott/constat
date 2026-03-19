# Conversation Panel — PromptQL-style UI Architecture

Reference: PromptQL demo screenshots (finance settlement use case).

## Implemented (Phase 1 — Layout Redesign)

- [x] RH artifact panel closed by default
- [x] Icon button toggle (top-right) replaces divider strip
- [x] Conversation panel always visible (no hide toggle)
- [x] All messages left-aligned with avatar + name + timestamp header
- [x] Bot messages grouped into collapsible summary per turn
- [x] In-progress: spinner + "Thinking..." with step sub-items
- [x] Completed: green checkmark summary, click to expand steps
- [x] Output renders below step summary as regular content
- [x] Centered max-width (`max-w-3xl`) conversation area
- [x] Input restyled: rounded card, shadow, circular green send button
- [x] Welcome screen: "What would you like to explore?"

## Planned (Phase 2 — Rich Interactions)

### 2a. Input bar enhancements

- **@ mention button** — left of input, opens user picker. Adds `@username` to message. PromptQL uses this for tagging collaborators into the chat (e.g., `@pax does this analysis look right for APAC?`).
- **Attach button** — paperclip icon, opens data source picker (equivalent to browsing Domains > Data Sources in the RH panel). Attaches a data source reference to the query context.
- Both buttons sit inside the input card, below the textarea, left-aligned: `@ | Attach`

### 2b. Wiki Pages / domain context chips

In the bot response header area, show linked domain resources as chips:

```
PromptQL  Mar 1, 2026, 2:14 PM
Wiki pages:  [Settlement Lifecycle]  [SWIFT Message Standards]  [Custody Reconciliation]
             [Fail Charge Policy]  [Client Risk Tiers]
```

Constat equivalent: **Domain documents and data sources** from the active domain config. These are the resources the bot consulted. Render as clickable chips that open the resource in the RH panel (deep link to document or data source).

### 2c. Step action labels

During execution, each step sub-item shows a specific action verb:

| Action | Label | Chip |
|--------|-------|------|
| Query a database table | `Reading` | green chip with table/source name |
| Read a document | `Reading` | green chip with document name |
| Query glossary | `Reading` | green chip with glossary term |
| Create a code artifact | `Created` | purple chip with filename |
| Ask LLM / generate | `Thinking...` | (no chip, animated) |
| Execute code | `Executing` | (no chip, spinner) |

Example from screenshots:
```
> Thinking...
  Reading  [Settlement Lifecycle]
  Created  [pacific_rim_fail_snapshot.py]
  Created  [regional_exposure_calc.py]
  Created  [behavior_trend_analysis.py]
```

Chip colors: green for data reads, purple for created artifacts.

### 2d. Inline artifact display

Bot responses embed artifacts directly in the message bubble:

- **Tables**: Rendered inline as a compact table (first 5 rows), with checkbox-style title and a "View" link to expand in the RH panel.
  ```
  ☑ Pacific Rim Holdings Settlement Failure Analysis — Global View
  | REGION | MARKET | FAILED TRADES | CAPITAL (SM) | AVG AGE | 6-WK TREND |
  | APAC   | Tokyo  | 47            | $142.3       | 3.2 days| ↑ 68%      |
  ...
  ```
- **Non-table artifacts** (documents, standards, policies): Rendered as a banner/card with title + "Added" / "View" actions.
  ```
  ● Settlement Fail Calculation Standard    Added    View
  ```
- **Updated tables**: Show "(Updated)" suffix when a table is regenerated after a follow-up query.

### 2e. @mention in responses

Bot can reference users with `@username` at the start of a response when replying to a specific person in multi-user sessions:

```
Good catch, Pax. I've updated the query to filter out pending_recall_flag...
```

System messages when a user is added: `@Pax has been added to the chat.`

### 2f. Glossary/knowledge teaching

When the bot learns something from a user correction, show a "taught" indicator:

```
✅ Pax S. taught PromptQL
The settlement_events_v3 table is the canonical source for fail metrics.
It must exclude records where pending_recall_flag is true...
```

Constat equivalent: When a user correction triggers a glossary update or learning, show an inline confirmation with the learned fact.

### 2g. View changelog

After an artifact update, show a "View changelog (7 lines)" link that expands to show the diff.

### 2h. Email composition

When asked to generate a communication, render it as a "New Message" modal/card with:
- To/CC fields pre-populated
- Subject line
- Formatted body
- "Save Draft" / "Send" buttons

### 2i. @vera learning (replaces /learn command)

Instead of `/learn <fact>`, users teach Vera by directly addressing her: `@vera the settlement_events_v3 table is the canonical source for fail metrics`. This triggers the same glossary/learning pipeline but feels like natural conversation rather than a command.

### 2j. Avatars

- **Bot avatar**: Vera logo — stylized "V" with balls/nodes at each endpoint (not a generic CPU chip icon). Consistent across all bot messages and grouped headers.
- **User avatar**: Circle with user's initials (e.g., "DP" for David Park, "PS" for Pax Scott). Derived from display name or email.

## File Map

| File | Purpose |
|------|---------|
| `constat-ui/src/store/uiStore.ts` | Panel visibility, deep linking |
| `constat-ui/src/components/layout/MainLayout.tsx` | Layout shell, panel toggle |
| `constat-ui/src/components/conversation/ConversationPanel.tsx` | Message grouping, message list |
| `constat-ui/src/components/conversation/BotMessageGroup.tsx` | Grouped bot turn rendering |
| `constat-ui/src/components/conversation/MessageBubble.tsx` | Individual message rendering |
| `constat-ui/src/components/conversation/AutocompleteInput.tsx` | Input bar with autocomplete |
| `constat-ui/src/store/sessionStore.ts` | Step events, execution phases |
