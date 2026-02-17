import { create } from 'zustand'
import type { GlossaryTerm, GlossaryFilter } from '@/types/api'
import * as sessionsApi from '@/api/sessions'

interface TaxonomySuggestion {
  child: string
  parent: string
  confidence: string
  reason: string
}

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
  taxonomySuggestions: TaxonomySuggestion[]
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
  suggestTaxonomy: (sessionId: string) => Promise<void>
  acceptTaxonomySuggestion: (sessionId: string, suggestion: TaxonomySuggestion) => Promise<void>
  dismissTaxonomySuggestion: (suggestion: TaxonomySuggestion) => void
  bulkUpdateStatus: (sessionId: string, names: string[], status: string) => Promise<void>
  selectTerm: (name: string | null) => void
  setFilter: (filter: Partial<GlossaryFilter>) => void
  setViewMode: (mode: 'tree' | 'list') => void
}

export const useGlossaryStore = create<GlossaryState>((set, get) => ({
  terms: [],
  deprecatedTerms: [],
  taxonomySuggestions: [],
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

  suggestTaxonomy: async (sessionId) => {
    try {
      const result = await sessionsApi.suggestTaxonomy(sessionId)
      set({ taxonomySuggestions: result.suggestions })
    } catch {
      set({ taxonomySuggestions: [] })
    }
  },

  acceptTaxonomySuggestion: async (sessionId, suggestion) => {
    // Find parent term to get its ID
    const { terms } = get()
    const parentTerm = terms.find(t => t.name.toLowerCase() === suggestion.parent.toLowerCase())
    if (!parentTerm) return

    // Get the parent's entity_id or glossary ID to use as parent_id
    const parentId = parentTerm.entity_id || suggestion.parent
    await sessionsApi.updateGlossaryTerm(sessionId, suggestion.child, { parent_id: parentId })

    // Remove from suggestions
    set((state) => ({
      taxonomySuggestions: state.taxonomySuggestions.filter(
        s => !(s.child === suggestion.child && s.parent === suggestion.parent)
      ),
    }))
    await get().fetchTerms(sessionId)
  },

  dismissTaxonomySuggestion: (suggestion) => {
    set((state) => ({
      taxonomySuggestions: state.taxonomySuggestions.filter(
        s => !(s.child === suggestion.child && s.parent === suggestion.parent)
      ),
    }))
  },

  bulkUpdateStatus: async (sessionId, names, status) => {
    await sessionsApi.bulkUpdateStatus(sessionId, names, status)
    await get().fetchTerms(sessionId)
  },

  selectTerm: (name) => set({ selectedName: name }),

  setFilter: (filter) =>
    set((state) => ({ filters: { ...state.filters, ...filter } })),

  setViewMode: (mode) => set({ viewMode: mode }),
}))
