// Copyright (c) 2025 Kenneth Stott
// Canary: 774acff4-3004-45a0-a7c0-74093823c7e4
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { makeVar } from '@apollo/client'

// ---------- Panel visibility ----------

export const showArtifactPanelVar = makeVar<boolean>(true)
export const showProofPanelVar = makeVar<boolean>(false)
export const showGlossaryPanelVar = makeVar<boolean>(false)
export const menuOpenVar = makeVar<boolean>(false)
export const conversationPanelHiddenVar = makeVar<boolean>(false)
export const artifactPanelHiddenVar = makeVar<boolean>(
  JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.artifactPanelHidden ?? true
)
export const artifactPanelWidthVar = makeVar<number>(
  JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.artifactPanelWidth ?? 400
)

export function toggleArtifactPanel() {
  const hidden = !artifactPanelHiddenVar()
  artifactPanelHiddenVar(hidden)
  if (hidden) conversationPanelHiddenVar(false)
  _persistUI()
}

export function showArtifactPanel() {
  if (artifactPanelHiddenVar()) {
    artifactPanelHiddenVar(false)
    _persistUI()
  }
}

export function toggleConversationPanel() {
  const hidden = !conversationPanelHiddenVar()
  conversationPanelHiddenVar(hidden)
  if (hidden) artifactPanelHiddenVar(false)
  _persistUI()
}

export function setArtifactPanelWidth(width: number) {
  artifactPanelWidthVar(width)
  _persistUI()
}

// ---------- Expanded accordion sections ----------

const storedSections = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.expandedArtifactSections
export const expandedSectionsVar = makeVar<string[]>(storedSections ?? ['charts', 'tables'])

export function toggleSection(section: string) {
  const current = expandedSectionsVar()
  expandedSectionsVar(
    current.includes(section)
      ? current.filter(s => s !== section)
      : [...current, section]
  )
  _persistUI()
}

export function expandSection(section: string) {
  const current = expandedSectionsVar()
  if (!current.includes(section)) {
    expandedSectionsVar([...current, section])
    _persistUI()
  }
}

export function expandSections(sections: string[]) {
  const current = expandedSectionsVar()
  const newOnes = sections.filter(s => !current.includes(s))
  if (newOnes.length) {
    expandedSectionsVar([...current, ...newOnes])
    _persistUI()
  }
}

// ---------- Deep linking ----------

export interface DeepLink {
  type: 'table' | 'document' | 'api' | 'glossary_term' | 'session'
  dbName?: string
  tableName?: string
  documentName?: string
  apiName?: string
  termName?: string
  sessionId?: string
  _ts?: number
}

/** Parse a URL path into a DeepLink, or null. */
export function pathToDeepLink(pathname: string): DeepLink | null {
  const parts = pathname.split('/').filter(Boolean).map(decodeURIComponent)
  if (parts.length === 0) return null
  switch (parts[0]) {
    case 'db':
      if (parts.length >= 3) return { type: 'table', dbName: parts[1], tableName: parts[2] }
      break
    case 'apis':
      if (parts.length >= 2) return { type: 'api', apiName: parts[1] }
      break
    case 'doc':
      if (parts.length >= 2) return { type: 'document', documentName: parts[1] }
      break
    case 'glossary':
      if (parts.length >= 2) return { type: 'glossary_term', termName: parts[1] }
      break
    case 's':
      if (parts.length >= 2) return { type: 'session', sessionId: parts[1] }
      break
  }
  return null
}

/** Apply a deep link — expand sections, select term, navigate. */
export function applyDeepLink(link: DeepLink) {
  // Session deep links: check if this is a public session
  if (link.type === 'session' && link.sessionId) {
    import('@/graphql/client').then(({ apolloClient }) => {
      import('@/graphql/operations/public').then(({ PUBLIC_SESSION_QUERY }) => {
        apolloClient.query({
          query: PUBLIC_SESSION_QUERY,
          variables: { sessionId: link.sessionId },
        })
          .then(() => {
            publicSessionIdVar(link.sessionId!)
          })
          .catch(() => {
            import('@/api/session-id').then(({ storeSessionId }) => {
              storeSessionId(link.sessionId!)
              window.location.href = '/'
            })
          })
      })
    })
    return
  }

  // Show artifact panel
  artifactPanelHiddenVar(false)

  // Expand the relevant accordion section
  const sectionMap: Record<string, string> = {
    table: 'databases',
    document: 'documents',
    api: 'apis',
    glossary_term: 'glossary',
  }
  const section = sectionMap[link.type]
  if (section) expandSection(section)

  // For glossary terms, select the term
  if (link.type === 'glossary_term' && link.termName) {
    import('@/store/glossaryStore').then(({ useGlossaryStore }) => {
      useGlossaryStore.getState().selectTerm(link.termName!)
    })
  }

  // Set the deep link for ArtifactPanel to handle
  setDeepLink(link)
}

export const activeDeepLinkVar = makeVar<DeepLink | null>(null)

export function setDeepLink(link: DeepLink) {
  activeDeepLinkVar({ ...link, _ts: Date.now() })
}

export function consumeDeepLink(): DeepLink | null {
  const link = activeDeepLinkVar()
  if (link) activeDeepLinkVar(null)
  return link
}

// ---------- Fullscreen artifact ----------

export interface FullscreenArtifact {
  type: 'artifact' | 'table' | 'proof_value' | 'database_table'
  id?: number
  name?: string
  content?: string
  dbName?: string
  tableName?: string
}

export const fullscreenArtifactVar = makeVar<FullscreenArtifact | null>(null)

export function openFullscreenArtifact(artifact: FullscreenArtifact) {
  fullscreenArtifactVar(artifact)
}

export function closeFullscreenArtifact() {
  fullscreenArtifactVar(null)
}

// ---------- Public session ----------

export const publicSessionIdVar = makeVar<string | null>(null)

export function clearPublicSession() {
  publicSessionIdVar(null)
}

// ---------- Theme ----------

type Theme = 'light' | 'dark' | 'system'
const storedTheme = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.theme
export const themeVar = makeVar<Theme>(storedTheme ?? 'system')

export function setTheme(theme: Theme) {
  themeVar(theme)
  if (theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark')
  } else {
    document.documentElement.classList.remove('dark')
  }
  _persistUI()
}

// ---------- Brief mode ----------

const storedBrief = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.briefMode
export const briefModeVar = makeVar<boolean>(storedBrief ?? false)

export function toggleBriefMode() {
  const next = !briefModeVar()
  briefModeVar(next)
  _persistUI()
}

// ---------- UI mode ----------

type UIMode = 'exploratory' | 'reason-chain'
const storedMode = JSON.parse(localStorage.getItem('constat-ui-storage') || '{}')?.state?.uiMode
export const uiModeVar = makeVar<UIMode>(storedMode ?? 'exploratory')

export function enterReasonChainMode() {
  uiModeVar('reason-chain')
  conversationPanelHiddenVar(false)
}

export function exitReasonChainMode() {
  uiModeVar('exploratory')
}

// ---------- Results filter ----------

export const resultsShowPublishedOnlyVar = makeVar<boolean>(
  localStorage.getItem('constat-results-filter') !== 'all'
)

export function setResultsShowPublishedOnly(v: boolean) {
  resultsShowPublishedOnlyVar(v)
  localStorage.setItem('constat-results-filter', v ? 'published' : 'all')
}

// ---------- Glossary generation progress ----------

export const glossaryGeneratingVar = makeVar<boolean>(false)
export const glossaryGenerationStageVar = makeVar<string | null>(null)
export const glossaryGenerationPercentVar = makeVar<number>(0)

export function setGlossaryGenerating(generating: boolean) {
  glossaryGeneratingVar(generating)
  if (!generating) {
    glossaryGenerationStageVar(null)
    glossaryGenerationPercentVar(0)
  }
}

export const glossaryTaxonomySuggestionsVar = makeVar<Array<{ child: string; parent: string; confidence: string; reason: string }>>([])

export function setGlossaryProgress(stage: string, percent: number) {
  glossaryGenerationStageVar(stage)
  glossaryGenerationPercentVar(percent)
}

// ---------- Collapsed result steps ----------

export const collapsedResultStepsVar = makeVar<Set<number>>(new Set())

export function toggleResultStep(stepNumber: number) {
  const next = new Set(collapsedResultStepsVar())
  if (next.has(stepNumber)) next.delete(stepNumber)
  else next.add(stepNumber)
  collapsedResultStepsVar(next)
}

export function expandResultStep(stepNumber: number) {
  const current = collapsedResultStepsVar()
  if (current.has(stepNumber)) {
    const next = new Set(current)
    next.delete(stepNumber)
    collapsedResultStepsVar(next)
  }
}

// ---------- Toast notifications ----------

export interface Toast {
  id: string
  message: string
  type: 'success' | 'error' | 'info'
}

export const toastsVar = makeVar<Toast[]>([])

let toastCounter = 0

export function addToast(message: string, type: Toast['type'] = 'success') {
  const id = String(++toastCounter)
  toastsVar([...toastsVar(), { id, message, type }])
  setTimeout(() => dismissToast(id), 3000)
}

export function dismissToast(id: string) {
  toastsVar(toastsVar().filter(t => t.id !== id))
}

// ---------- localStorage persistence ----------

/** Fetch user preferences from server and apply to reactive vars */
export async function initPreferences() {
  try {
    const resp = await fetch('/api/users/me/preferences', {
      headers: await import('@/config/auth-helpers').then(m => m.getAuthHeaders()),
    })
    if (resp.ok) {
      const prefs = await resp.json()
      if (prefs.brief_mode !== undefined) {
        briefModeVar(prefs.brief_mode)
      }
    }
  } catch {
    // Server may not be available yet
  }
}

function _persistUI() {
  const state = {
    theme: themeVar(),
    briefMode: briefModeVar(),
    uiMode: uiModeVar(),
    conversationPanelHidden: conversationPanelHiddenVar(),
    artifactPanelHidden: artifactPanelHiddenVar(),
    artifactPanelWidth: artifactPanelWidthVar(),
    expandedArtifactSections: expandedSectionsVar(),
  }
  localStorage.setItem('constat-ui-storage', JSON.stringify({ state }))
}
