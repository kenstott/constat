# Reason-Chain Mode: DAG-Dominant UI

## Overview

Replace the conversation panel with the DAG diagram when entering reason-chain mode. Two hard modes with explicit transitions — no blending.

## Modes

### Exploratory (default)
- Full conversation panel + artifact panel
- Normal chat input, free-form questions
- Standard command set

### Reason-Chain
- DAG goes full-width (artifact panel expands, conversation panel collapses)
- Command strip replaces chat input — structured inputs only (add premise, run, validate)
- Existing DAG interactivity (node inspection, step details) serves as primary interface
- No free-form chat

## Transitions

| From | To | Trigger | Condition |
|------|-----|---------|-----------|
| Exploratory | Reason-chain | `/reason-chain` command or button | — |
| Reason-chain | Exploratory | **Complete** | Proof finalized, returned as artifact |
| Reason-chain | Exploratory | **Abandon** | Discards proof, warning prompt |

No casual toggling. Reason-chain is a session within a session — entered with intent, exited with a result or a conscious decision to discard. Like a transaction.

## Command Strip

Replaces the chat input in reason-chain mode. Accepts:
- Add premise (data source, document, API)
- Add inference step
- Run / re-run step
- Validate chain
- Complete (finalize and return to exploratory)
- Abandon (discard and return to exploratory)

## Panel Layout

- **Exploratory**: conversation-dominant (current layout)
- **Reason-chain**: artifact-dominant — conversation panel collapses to narrow command strip at bottom, DAG panel fills remaining space

## Implementation Notes

- Mode flag in UI store controls which layout and commands are active
- Existing DAG component and interactivity are reused as-is
- On **Complete**, the finalized proof becomes an artifact in the exploratory session
- On **Abandon**, proof state is discarded (confirm dialog)
