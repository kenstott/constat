---
name: No Zustand references
description: Never refer to the custom store as "Zustand" — it was replaced with a custom useSyncExternalStore wrapper, then largely eliminated in favor of Apollo cache
type: feedback
---

Do not call the custom store "Zustand" — the project migrated away from Zustand to a custom `createStore.ts` using `useSyncExternalStore`, and then largely eliminated the custom store in favor of Apollo Client cache. The remaining custom stores (proofStore, glossaryStore) use `createStore.ts` but are NOT Zustand.

**Why:** User corrected this misidentification. The codebase has no Zustand dependency.

**How to apply:** When discussing state management, refer to "Apollo cache" for most data, "reactive vars" (`makeVar`) for event-driven UI state, and "custom store" (not Zustand) for proofStore/glossaryStore.