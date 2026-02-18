// UI state store

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type Theme = 'light' | 'dark' | 'system'

// Artifact reference for fullscreen display
interface FullscreenArtifact {
  type: 'artifact' | 'table' | 'proof_value' | 'database_table'
  id?: number  // For artifacts
  name?: string  // For tables or proof node names
  content?: string  // For proof values (markdown tables)
  dbName?: string  // For database_table type
  tableName?: string  // For database_table type
}

// Deep link — encoded as URL path
// /db/{dbName}/{tableName}
// /apis/{apiName}
// /doc/{documentName}
// /glossary/{termName}
export interface DeepLink {
  type: 'table' | 'document' | 'api' | 'glossary_term'
  dbName?: string      // For table
  tableName?: string   // For table
  documentName?: string // For document
  apiName?: string     // For api
  termName?: string    // For glossary_term
  _ts?: number         // Nonce to force re-trigger on repeated clicks
}

/** Encode a DeepLink to a URL path. */
export function deepLinkToPath(link: DeepLink): string {
  switch (link.type) {
    case 'table':
      return `/db/${encodeURIComponent(link.dbName!)}/${encodeURIComponent(link.tableName!)}`
    case 'api':
      return `/apis/${encodeURIComponent(link.apiName!)}`
    case 'document':
      return `/doc/${encodeURIComponent(link.documentName!)}`
    case 'glossary_term':
      return `/glossary/${encodeURIComponent(link.termName!)}`
  }
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
  }
  return null
}

/** Apply a deep link — expand sections, select term, set pending for ArtifactPanel. */
export function applyDeepLink(link: DeepLink) {
  const store = useUIStore.getState()

  // Expand the relevant accordion section
  const sectionMap: Record<string, string> = {
    table: 'databases',
    document: 'documents',
    api: 'apis',
    glossary_term: 'glossary',
  }
  const section = sectionMap[link.type]
  if (section) store.expandArtifactSection(section)

  // For glossary terms, also tell the glossary store to select the term
  if (link.type === 'glossary_term' && link.termName) {
    import('@/store/glossaryStore').then(({ useGlossaryStore }) => {
      useGlossaryStore.getState().selectTerm(link.termName!)
    })
  }

  console.log('[deep-link] applyDeepLink: expanding section', section, ', setting pendingDeepLink')
  // Spread to create a new object reference so Zustand triggers re-render on repeated clicks
  useUIStore.setState({ pendingDeepLink: { ...link, _ts: Date.now() } })
}

interface UIState {
  // Theme
  theme: Theme
  setTheme: (theme: Theme) => void

  // Panels
  menuOpen: boolean
  artifactPanelWidth: number
  expandedArtifactSections: string[]

  // Fullscreen artifact modal
  fullscreenArtifact: FullscreenArtifact | null
  openFullscreenArtifact: (artifact: FullscreenArtifact) => void
  closeFullscreenArtifact: () => void

  // Deep linking (consumed by ArtifactPanel)
  pendingDeepLink: DeepLink | null
  navigateTo: (link: DeepLink) => void
  consumeDeepLink: () => DeepLink | null

  // Actions
  toggleMenu: () => void
  setMenuOpen: (open: boolean) => void
  setArtifactPanelWidth: (width: number) => void
  toggleArtifactSection: (section: string) => void
  expandArtifactSection: (section: string) => void
  expandArtifactSections: (sections: string[]) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      theme: 'system' as Theme,
      menuOpen: false,
      artifactPanelWidth: 400,
      expandedArtifactSections: ['charts', 'tables'],
      fullscreenArtifact: null,
      pendingDeepLink: null,

      openFullscreenArtifact: (artifact: FullscreenArtifact) => set({ fullscreenArtifact: artifact }),
      closeFullscreenArtifact: () => set({ fullscreenArtifact: null }),

      navigateTo: (link: DeepLink) => {
        const path = deepLinkToPath(link)
        console.log('[deep-link] navigateTo:', link.type, path, link)
        window.history.pushState({ deepLink: true }, '', path)
        applyDeepLink(link)
      },

      consumeDeepLink: () => {
        const link = get().pendingDeepLink
        if (link) set({ pendingDeepLink: null })
        return link
      },

      setTheme: (theme) => {
        set({ theme })
        // Apply theme to document
        if (theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
          document.documentElement.classList.add('dark')
        } else {
          document.documentElement.classList.remove('dark')
        }
      },

      toggleMenu: () => set((state) => ({ menuOpen: !state.menuOpen })),
      setMenuOpen: (open) => set({ menuOpen: open }),
      setArtifactPanelWidth: (width) => set({ artifactPanelWidth: width }),

      toggleArtifactSection: (section) =>
        set((state) => ({
          expandedArtifactSections: state.expandedArtifactSections.includes(section)
            ? state.expandedArtifactSections.filter((s) => s !== section)
            : [...state.expandedArtifactSections, section],
        })),

      expandArtifactSection: (section) =>
        set((state) => ({
          expandedArtifactSections: state.expandedArtifactSections.includes(section)
            ? state.expandedArtifactSections
            : [...state.expandedArtifactSections, section],
        })),

      expandArtifactSections: (sections) =>
        set((state) => {
          const newSections = sections.filter(
            (s) => !state.expandedArtifactSections.includes(s)
          )
          return {
            expandedArtifactSections: [...state.expandedArtifactSections, ...newSections],
          }
        }),
    }),
    {
      name: 'constat-ui-storage',
      partialize: (state) => ({
        theme: state.theme,
        artifactPanelWidth: state.artifactPanelWidth,
        expandedArtifactSections: state.expandedArtifactSections,
      }),
    }
  )
)

// Initialize theme on load
if (typeof window !== 'undefined') {
  const stored = localStorage.getItem('constat-ui-storage')
  const theme = stored ? JSON.parse(stored).state?.theme : 'system'
  if (theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark')
  }
}
