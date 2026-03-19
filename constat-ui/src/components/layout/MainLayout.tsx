// Main Layout component

import { ReactNode, useState, useCallback, useEffect } from 'react'
import { StatusBar } from './StatusBar'
import { HamburgerMenu } from './HamburgerMenu'
import { useUIStore } from '@/store/uiStore'
import { Squares2X2Icon } from '@heroicons/react/24/outline'

interface MainLayoutProps {
  conversationPanel: ReactNode
  artifactPanel: ReactNode
}

const MIN_PANEL_WIDTH = 150
const DEFAULT_PANEL_WIDTH = 384 // w-96
const PANEL_WIDTH_KEY = 'constat-panel-width'

// Load saved panel width from localStorage
function getSavedPanelWidth(): number {
  try {
    const saved = localStorage.getItem(PANEL_WIDTH_KEY)
    if (saved) {
      const width = parseInt(saved, 10)
      if (!isNaN(width) && width >= MIN_PANEL_WIDTH) {
        return width
      }
    }
  } catch {
    // localStorage not available
  }
  return DEFAULT_PANEL_WIDTH
}

export function MainLayout({
  conversationPanel,
  artifactPanel,
}: MainLayoutProps) {
  const [panelWidth, setPanelWidth] = useState(getSavedPanelWidth)
  const [isResizing, setIsResizing] = useState(false)
  const artifactPanelHidden = useUIStore((s) => s.artifactPanelHidden)
  const toggleArtifactPanel = useUIStore((s) => s.toggleArtifactPanel)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsResizing(true)
  }, [])

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing) return
      const newWidth = window.innerWidth - e.clientX
      const maxWidth = window.innerWidth - MIN_PANEL_WIDTH
      const clampedWidth = Math.min(Math.max(newWidth, MIN_PANEL_WIDTH), maxWidth)
      setPanelWidth(clampedWidth)
    },
    [isResizing]
  )

  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
    try {
      localStorage.setItem(PANEL_WIDTH_KEY, panelWidth.toString())
    } catch {
      // localStorage not available
    }
  }, [panelWidth])

  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
  }, [isResizing, handleMouseMove, handleMouseUp])

  return (
    <div className="h-full flex flex-col">
      {/* Status Bar */}
      <StatusBar />

      {/* Hamburger Menu (drawer) */}
      <HamburgerMenu />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Conversation Panel — always visible */}
        <main className="flex-1 flex flex-col overflow-hidden relative">
          {conversationPanel}
          <button
            onClick={toggleArtifactPanel}
            className="absolute top-3 right-3 p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-500 z-10"
            title={artifactPanelHidden ? 'Show details panel' : 'Hide details panel'}
          >
            <Squares2X2Icon className="w-5 h-5" />
          </button>
        </main>

        {artifactPanelHidden ? null : (
          <>
            {/* Resize Handle */}
            <div
              className={`w-1 cursor-col-resize hover:bg-primary-400 transition-colors ${
                isResizing ? 'bg-primary-500' : 'bg-gray-200 dark:bg-gray-700'
              }`}
              onMouseDown={handleMouseDown}
            />

            {/* Artifact Panel */}
            <aside
              className="border-l border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900"
              style={{ width: panelWidth }}
            >
              {artifactPanel}
            </aside>
          </>
        )}
      </div>
    </div>
  )
}
