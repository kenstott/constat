// Proof state store for auditable mode

import { create } from 'zustand'

type NodeStatus = 'pending' | 'planning' | 'executing' | 'resolved' | 'failed' | 'blocked'

export interface FactNode {
  id: string
  name: string
  description?: string
  status: NodeStatus
  value?: unknown
  source?: string
  confidence?: number
  tier?: number
  strategy?: string
  formula?: string
  reason?: string
  dependencies: string[]
  elapsed_ms?: number
}

interface ProofState {
  // Proof facts DAG
  facts: Map<string, FactNode>
  isProving: boolean
  isPanelOpen: boolean

  // Actions
  handleFactEvent: (eventType: string, data: Record<string, unknown>) => void
  clearFacts: () => void
  openPanel: () => void
  closePanel: () => void
  togglePanel: () => void
}

export const useProofStore = create<ProofState>((set) => ({
  facts: new Map(),
  isProving: false,
  isPanelOpen: false,

  handleFactEvent: (eventType, data) => {
    const factName = data.fact_name as string
    if (!factName) return

    set((state) => {
      const next = new Map(state.facts)
      const existing = next.get(factName) || {
        id: factName,
        name: factName,
        status: 'pending' as NodeStatus,
        dependencies: [],
      }

      switch (eventType) {
        case 'fact_start':
          next.set(factName, {
            ...existing,
            description: data.fact_description as string | undefined,
            status: 'pending',
          })
          return { facts: next, isProving: true, isPanelOpen: true }

        case 'fact_planning':
          next.set(factName, {
            ...existing,
            status: 'planning',
          })
          return { facts: next }

        case 'fact_executing':
          next.set(factName, {
            ...existing,
            status: 'executing',
            formula: data.formula as string | undefined,
          })
          return { facts: next }

        case 'fact_resolved':
          next.set(factName, {
            ...existing,
            status: 'resolved',
            value: data.value,
            source: data.source as string | undefined,
            confidence: data.confidence as number | undefined,
            tier: data.tier as number | undefined,
            strategy: data.strategy as string | undefined,
            dependencies: (data.dependencies as string[]) || existing.dependencies,
            elapsed_ms: data.elapsed_ms as number | undefined,
          })
          return { facts: next }

        case 'fact_failed':
          next.set(factName, {
            ...existing,
            status: 'failed',
            reason: data.reason as string | undefined,
          })
          return { facts: next }

        case 'proof_complete':
          return { isProving: false }

        default:
          return {}
      }
    })
  },

  clearFacts: () => set({ facts: new Map(), isProving: false }),

  openPanel: () => set({ isPanelOpen: true }),

  closePanel: () => set({ isPanelOpen: false }),

  togglePanel: () => set((state) => ({ isPanelOpen: !state.isPanelOpen })),
}))
