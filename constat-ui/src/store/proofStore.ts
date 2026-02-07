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
  isPlanningComplete: boolean  // True after all fact_start events received and execution begins
  isPanelOpen: boolean
  proofSummary: string | null  // LLM-generated summary when available

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
  isPlanningComplete: false,
  isPanelOpen: false,
  proofSummary: null,

  handleFactEvent: (eventType, data) => {
    const factName = data.fact_name as string
    console.log(`[proofStore] ${eventType}:`, factName, data.dependencies)

    // Handle events that don't require fact_name
    if (eventType === 'proof_start') {
      // Clear facts and start proving, but don't open panel until DAG is complete
      set({ facts: new Map(), isProving: true, isPlanningComplete: false, proofSummary: null })
      return
    }
    if (eventType === 'dag_execution_start') {
      // All fact_start events have been received - now open the panel
      console.log('[proofStore] dag_execution_start - all nodes known, opening panel')
      set({ isPlanningComplete: true, isPanelOpen: true })
      return
    }
    if (eventType === 'proof_complete') {
      set({ isProving: false })
      return
    }
    if (eventType === 'proof_summary_ready') {
      // LLM-generated summary is available
      const summary = data.summary as string
      console.log('[proofStore] proof_summary_ready')
      set({ proofSummary: summary })
      return
    }

    // Remaining events require fact_name
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
            dependencies: (data.dependencies as string[]) || existing.dependencies,
            status: 'pending',
          })
          // Don't open panel during planning - wait for DAG to be complete
          return { facts: next, isProving: true }

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

        default:
          return {}
      }
    })
  },

  clearFacts: () => set({ facts: new Map(), isProving: false, isPlanningComplete: false, proofSummary: null }),

  openPanel: () => set({ isPanelOpen: true }),

  closePanel: () => set({ isPanelOpen: false }),

  togglePanel: () => set((state) => ({ isPanelOpen: !state.isPanelOpen })),
}))
