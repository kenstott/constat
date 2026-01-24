// UI state store

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type Theme = 'light' | 'dark' | 'system'

interface UIState {
  // Theme
  theme: Theme
  setTheme: (theme: Theme) => void

  // Panels
  menuOpen: boolean
  artifactPanelWidth: number
  expandedArtifactSections: string[]

  // Actions
  toggleMenu: () => void
  setMenuOpen: (open: boolean) => void
  setArtifactPanelWidth: (width: number) => void
  toggleArtifactSection: (section: string) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      theme: 'system',
      menuOpen: false,
      artifactPanelWidth: 400,
      expandedArtifactSections: ['charts', 'tables'],

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