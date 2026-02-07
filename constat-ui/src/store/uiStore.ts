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
    (set) => ({
      theme: 'system',
      menuOpen: false,
      artifactPanelWidth: 400,
      expandedArtifactSections: ['charts', 'tables'],
      fullscreenArtifact: null,

      openFullscreenArtifact: (artifact) => set({ fullscreenArtifact: artifact }),
      closeFullscreenArtifact: () => set({ fullscreenArtifact: null }),

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