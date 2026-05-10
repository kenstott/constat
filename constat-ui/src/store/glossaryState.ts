// Copyright (c) 2025 Kenneth Stott
// Canary: c3f1b2a0-9d4e-4f7b-8c6a-1e2f3a4b5c6d
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the LICENSE file in the root directory of this source tree.

import { makeVar, useReactiveVar } from '@apollo/client'
import type { GlossaryTerm, GlossaryFilter } from '@/types/api'
import * as sessionsApi from '@/api/sessions'
import { getCachedEntry } from './entityCache'
import { inflateToGlossaryTerms } from './entityCacheKeys'
import {
  glossaryGeneratingVar,
  glossaryGenerationStageVar,
  glossaryGenerationPercentVar,
  glossaryTaxonomySuggestionsVar,
  setGlossaryGenerating,
  setGlossaryProgress,
} from '@/graphql/ui-state'

// ---------- Types ----------

interface TaxonomySuggestion {
  child: string
  parent: string
  confidence: string
  reason: string
}

export interface DeprecatedTerm {
  name: string
  display_name: string
  definition: string
  domain?: string
  status: string
  provenance: string
}

// ---------- Reactive vars ----------

export const glossaryTermsVar = makeVar<GlossaryTerm[]>([])
export const glossaryDeprecatedTermsVar = makeVar<DeprecatedTerm[]>([])
export const glossarySelectedNameVar = makeVar<string | null>(null)
export const glossaryViewModeVar = makeVar<'tree' | 'list' | 'tags'>('list')
export const glossaryFiltersVar = makeVar<GlossaryFilter>({ scope: 'all' })
export const glossaryTotalDefinedVar = makeVar<number>(0)
export const glossaryTotalSelfDescribingVar = makeVar<number>(0)
export const glossaryLoadingVar = makeVar<boolean>(true)
export const glossaryEntityRebuildingVar = makeVar<boolean>(false)
export const glossaryRefreshKeyVar = makeVar<number>(0)
export const glossaryErrorVar = makeVar<string | null>(null)
export const glossaryExpandedItemsVar = makeVar<Record<string, boolean>>({})

// ---------- Actions ----------

export async function fetchTerms(sessionId: string): Promise<void> {
  const isInitialLoad = glossaryTermsVar().length === 0
  if (isInitialLoad) {
    glossaryLoadingVar(true)
    glossaryErrorVar(null)
  } else {
    glossaryErrorVar(null)
  }
  try {
    const filters = glossaryFiltersVar()
    const response = await sessionsApi.getGlossary(sessionId, filters.scope || 'all', filters.domain)
    glossaryTermsVar(response.terms)
    glossaryTotalDefinedVar(response.total_defined)
    glossaryTotalSelfDescribingVar(response.total_self_describing)
    glossaryLoadingVar(false)
  } catch (error) {
    glossaryErrorVar(String(error))
    glossaryLoadingVar(false)
  }
}

export async function fetchDeprecated(sessionId: string): Promise<void> {
  try {
    const data = await sessionsApi.getDeprecatedTerms(sessionId)
    glossaryDeprecatedTermsVar(data.terms || [])
  } catch {
    // Silent — deprecated is secondary
  }
}

export async function addDefinition(sessionId: string, name: string, definition: string): Promise<void> {
  await sessionsApi.addDefinition(sessionId, name, definition)
  await fetchTerms(sessionId)
}

export async function updateTerm(sessionId: string, name: string, updates: Record<string, unknown>): Promise<void> {
  await sessionsApi.updateGlossaryTerm(sessionId, name, updates)
  await fetchTerms(sessionId)
}

export async function deleteTerm(sessionId: string, name: string): Promise<void> {
  await sessionsApi.deleteGlossaryTerm(sessionId, name)
  await fetchTerms(sessionId)
  await fetchDeprecated(sessionId)
}

export async function renameTerm(sessionId: string, name: string, newName: string): Promise<void> {
  await sessionsApi.renameTerm(sessionId, name, newName)
  await fetchTerms(sessionId)
}

export async function reconnectTerm(sessionId: string, name: string, updates: { parent_id?: string; domain?: string }): Promise<void> {
  await sessionsApi.reconnectTerm(sessionId, name, updates)
  await fetchTerms(sessionId)
  await fetchDeprecated(sessionId)
}

export async function deleteDrafts(sessionId: string): Promise<number> {
  const result = await sessionsApi.deleteGlossaryByStatus(sessionId, 'draft')
  // Optimistically remove draft terms
  glossaryTermsVar(glossaryTermsVar().filter((t) => t.status !== 'draft'))
  await fetchTerms(sessionId)
  return result.count
}

export async function refineTerm(sessionId: string, name: string): Promise<{ before: string; after: string } | null> {
  try {
    const result = await sessionsApi.refineGlossaryTerm(sessionId, name)
    await fetchTerms(sessionId)
    return { before: result.before, after: result.after }
  } catch {
    return null
  }
}

export async function generateGlossary(sessionId: string, phases?: Record<string, boolean>): Promise<void> {
  glossaryGeneratingVar(true)
  await sessionsApi.generateGlossary(sessionId, phases)
  // generating stays true until glossary_rebuild_complete WS event
}

export function addTerms(terms: GlossaryTerm[]): void {
  const current = glossaryTermsVar()
  const byName = new Map(current.map(t => [t.name, t]))
  let newCount = 0
  for (const term of terms) {
    if (!byName.has(term.name)) {
      newCount++
    }
    byName.set(term.name, term)
  }
  glossaryTermsVar(Array.from(byName.values()))
  glossaryTotalDefinedVar(glossaryTotalDefinedVar() + newCount)
}

export function setTermsFromState(terms: GlossaryTerm[], totalDefined: number, totalSelfDescribing: number): void {
  glossaryTermsVar(terms)
  glossaryTotalDefinedVar(totalDefined)
  glossaryTotalSelfDescribingVar(totalSelfDescribing)
  glossaryLoadingVar(false)
}

export async function suggestTaxonomy(sessionId: string): Promise<void> {
  try {
    const result = await sessionsApi.suggestTaxonomy(sessionId)
    glossaryTaxonomySuggestionsVar(result.suggestions)
  } catch {
    glossaryTaxonomySuggestionsVar([])
  }
}

export async function acceptTaxonomySuggestion(sessionId: string, suggestion: TaxonomySuggestion): Promise<void> {
  const terms = glossaryTermsVar()
  const parentTerm = terms.find(t => t.name.toLowerCase() === suggestion.parent.toLowerCase())
  if (!parentTerm) return

  const parentId = parentTerm.glossary_id || parentTerm.entity_id || suggestion.parent
  await sessionsApi.updateGlossaryTerm(sessionId, suggestion.child, { parent_id: parentId })

  glossaryTaxonomySuggestionsVar(
    glossaryTaxonomySuggestionsVar().filter(
      s => !(s.child === suggestion.child && s.parent === suggestion.parent)
    )
  )
  await fetchTerms(sessionId)
}

export function dismissTaxonomySuggestion(suggestion: TaxonomySuggestion): void {
  glossaryTaxonomySuggestionsVar(
    glossaryTaxonomySuggestionsVar().filter(
      s => !(s.child === suggestion.child && s.parent === suggestion.parent)
    )
  )
}

export async function bulkUpdateStatus(sessionId: string, names: string[], status: string): Promise<void> {
  await sessionsApi.bulkUpdateStatus(sessionId, names, status)
  await fetchTerms(sessionId)
}

export function selectTerm(name: string | null): void {
  glossarySelectedNameVar(name)
}

export function setFilter(filter: Partial<GlossaryFilter>): void {
  glossaryFiltersVar({ ...glossaryFiltersVar(), ...filter })
}

export function setViewMode(mode: 'tree' | 'list' | 'tags'): void {
  glossaryViewModeVar(mode)
}

export function setEntityRebuilding(rebuilding: boolean): void {
  glossaryEntityRebuildingVar(rebuilding)
}

export function setGenerating(generating: boolean): void {
  setGlossaryGenerating(generating)
  if (!generating) {
    glossaryRefreshKeyVar(glossaryRefreshKeyVar() + 1)
  }
}

export function setProgress(stage: string, percent: number): void {
  setGlossaryProgress(stage, percent)
}

export function bumpRefreshKey(): void {
  glossaryRefreshKeyVar(glossaryRefreshKeyVar() + 1)
}

export async function loadFromCache(sessionId: string): Promise<void> {
  const entry = await getCachedEntry(sessionId)
  if (!entry) return
  const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(entry.state)
  glossaryTermsVar(terms)
  glossaryTotalDefinedVar(totalDefined)
  glossaryTotalSelfDescribingVar(totalSelfDescribing)
  glossaryLoadingVar(false)
}

export function toggleExpanded(key: string, defaultOpen: boolean): void {
  const current = glossaryExpandedItemsVar()
  glossaryExpandedItemsVar({
    ...current,
    [key]: !(current[key] ?? defaultOpen),
  })
}

// ---------- Composite hook (replaces useGlossaryStore()) ----------

export function useGlossaryState() {
  const terms = useReactiveVar(glossaryTermsVar)
  const deprecatedTerms = useReactiveVar(glossaryDeprecatedTermsVar)
  const taxonomySuggestions = useReactiveVar(glossaryTaxonomySuggestionsVar)
  const selectedName = useReactiveVar(glossarySelectedNameVar)
  const viewMode = useReactiveVar(glossaryViewModeVar)
  const filters = useReactiveVar(glossaryFiltersVar)
  const totalDefined = useReactiveVar(glossaryTotalDefinedVar)
  const totalSelfDescribing = useReactiveVar(glossaryTotalSelfDescribingVar)
  const loading = useReactiveVar(glossaryLoadingVar)
  const generating = useReactiveVar(glossaryGeneratingVar)
  const generationStage = useReactiveVar(glossaryGenerationStageVar)
  const generationPercent = useReactiveVar(glossaryGenerationPercentVar)
  const entityRebuilding = useReactiveVar(glossaryEntityRebuildingVar)
  const refreshKey = useReactiveVar(glossaryRefreshKeyVar)
  const error = useReactiveVar(glossaryErrorVar)
  const expandedItems = useReactiveVar(glossaryExpandedItemsVar)

  return {
    terms,
    deprecatedTerms,
    taxonomySuggestions,
    selectedName,
    viewMode,
    filters,
    totalDefined,
    totalSelfDescribing,
    loading,
    generating,
    generationStage,
    generationPercent,
    entityRebuilding,
    refreshKey,
    error,
    expandedItems,
    fetchTerms,
    fetchDeprecated,
    addDefinition,
    updateTerm,
    deleteTerm,
    deleteDrafts,
    refineTerm,
    generateGlossary,
    suggestTaxonomy,
    acceptTaxonomySuggestion,
    dismissTaxonomySuggestion,
    renameTerm,
    reconnectTerm,
    addTerms,
    setTermsFromState,
    bulkUpdateStatus,
    selectTerm,
    setFilter,
    setViewMode,
    setGenerating,
    setProgress,
    setEntityRebuilding,
    loadFromCache,
    toggleExpanded,
  }
}

// ---------- Selector hook (replaces useGlossaryStore(selector)) ----------

export function useGlossaryVar<T>(
  selector: (state: ReturnType<typeof useGlossaryState>) => T
): T {
  const terms = useReactiveVar(glossaryTermsVar)
  const deprecatedTerms = useReactiveVar(glossaryDeprecatedTermsVar)
  const taxonomySuggestions = useReactiveVar(glossaryTaxonomySuggestionsVar)
  const selectedName = useReactiveVar(glossarySelectedNameVar)
  const viewMode = useReactiveVar(glossaryViewModeVar)
  const filters = useReactiveVar(glossaryFiltersVar)
  const totalDefined = useReactiveVar(glossaryTotalDefinedVar)
  const totalSelfDescribing = useReactiveVar(glossaryTotalSelfDescribingVar)
  const loading = useReactiveVar(glossaryLoadingVar)
  const generating = useReactiveVar(glossaryGeneratingVar)
  const generationStage = useReactiveVar(glossaryGenerationStageVar)
  const generationPercent = useReactiveVar(glossaryGenerationPercentVar)
  const entityRebuilding = useReactiveVar(glossaryEntityRebuildingVar)
  const refreshKey = useReactiveVar(glossaryRefreshKeyVar)
  const error = useReactiveVar(glossaryErrorVar)
  const expandedItems = useReactiveVar(glossaryExpandedItemsVar)

  return selector({
    terms,
    deprecatedTerms,
    taxonomySuggestions,
    selectedName,
    viewMode,
    filters,
    totalDefined,
    totalSelfDescribing,
    loading,
    generating,
    generationStage,
    generationPercent,
    entityRebuilding,
    refreshKey,
    error,
    expandedItems,
    fetchTerms,
    fetchDeprecated,
    addDefinition,
    updateTerm,
    deleteTerm,
    deleteDrafts,
    refineTerm,
    generateGlossary,
    suggestTaxonomy,
    acceptTaxonomySuggestion,
    dismissTaxonomySuggestion,
    renameTerm,
    reconnectTerm,
    addTerms,
    setTermsFromState,
    bulkUpdateStatus,
    selectTerm,
    setFilter,
    setViewMode,
    setGenerating,
    setProgress,
    setEntityRebuilding,
    loadFromCache,
    toggleExpanded,
  })
}
