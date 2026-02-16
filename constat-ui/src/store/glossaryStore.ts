import { create } from 'zustand'
import type { GlossaryTerm, GlossaryFilter } from '@/types/api'
import * as sessionsApi from '@/api/sessions'

interface GlossaryState {
  terms: GlossaryTerm[]
  deprecatedTerms: Array<{
    name: string
    display_name: string
    definition: string
    domain?: string
    status: string
    provenance: string
  }>
  selectedName: string | null
  viewMode: 'tree' | 'list'
  filters: GlossaryFilter
  totalDefined: number
  totalSelfDescribing: number
  loading: boolean
  error: string | null

  fetchTerms: (sessionId: string) => Promise<void>
  fetchDeprecated: (sessionId: string) => Promise<void>
  addDefinition: (sessionId: string, name: string, definition: string) => Promise<void>
  updateTerm: (sessionId: string, name: string, updates: Record<string, unknown>) => Promise<void>
  deleteTerm: (sessionId: string, name: string) => Promise<void>
  refineTerm: (sessionId: string, name: string) => Promise<{ before: string; after: string } | null>
  generateGlossary: (sessionId: string) => Promise<void>
  selectTerm: (name: string | null) => void
  setFilter: (filter: Partial<GlossaryFilter>) => void
  setViewMode: (mode: 'tree' | 'list') => void
}

export const useGlossaryStore = create<GlossaryState>((set, get) => ({
  terms: [],
  deprecatedTerms: [],
  selectedName: null,
  viewMode: 'list',
  filters: { scope: 'all' },
  totalDefined: 0,
  totalSelfDescribing: 0,
  loading: false,
  error: null,

  fetchTerms: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const { filters } = get()
      const response = await sessionsApi.getGlossary(sessionId, filters.scope || 'all')
      set({
        terms: response.terms,
        totalDefined: response.total_defined,
        totalSelfDescribing: response.total_self_describing,
        loading: false,
      })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchDeprecated: async (sessionId) => {
    try {
      const response = await fetch(`/api/sessions/${sessionId}/glossary/deprecated`)
      const data = await response.json()
      set({ deprecatedTerms: data.terms || [] })
    } catch {
      // Silent — deprecated is secondary
    }
  },

  addDefinition: async (sessionId, name, definition) => {
    await sessionsApi.addDefinition(sessionId, name, definition)
    await get().fetchTerms(sessionId)
  },

  updateTerm: async (sessionId, name, updates) => {
    await sessionsApi.updateGlossaryTerm(sessionId, name, updates)
    await get().fetchTerms(sessionId)
  },

  deleteTerm: async (sessionId, name) => {
    await sessionsApi.deleteGlossaryTerm(sessionId, name)
    await get().fetchTerms(sessionId)
  },

  refineTerm: async (sessionId, name) => {
    try {
      const result = await sessionsApi.refineGlossaryTerm(sessionId, name)
      await get().fetchTerms(sessionId)
      return { before: result.before, after: result.after }
    } catch {
      return null
    }
  },

  generateGlossary: async (sessionId) => {
    await sessionsApi.generateGlossary(sessionId)
  },

  selectTerm: (name) => set({ selectedName: name }),

  setFilter: (filter) =>
    set((state) => ({ filters: { ...state.filters, ...filter } })),

  setViewMode: (mode) => set({ viewMode: mode }),
}))
