// Main Layout component

import { ReactNode, useState, useCallback, useEffect } from 'react'
import { StatusBar } from './StatusBar'
import { HamburgerMenu } from './HamburgerMenu'
import { Toolbar } from './Toolbar'
import { useUIStore } from '@/store/uiStore'

interface MainLayoutProps {
  conversationPanel: ReactNode
  artifactPanel: ReactNode
  onNewQuery?: () => void
  onShowProof?: () => void
  onShowHelp?: () => void
  isCreatingNewSession?: boolean
}

const MIN_PANEL_WIDTH = 200
const MAX_PANEL_WIDTH = 800
const DEFAULT_PANEL_WIDTH = 384 // w-96
const PANEL_WIDTH_KEY = 'constat-panel-width'

// Load saved panel width from localStorage
function getSavedPanelWidth(): number {
  try {
    const saved = localStorage.getItem(PANEL_WIDTH_KEY)
    if (saved) {
      const width = parseInt(saved, 10)
      if (!isNaN(width) && width >= MIN_PANEL_WIDTH && width <= MAX_PANEL_WIDTH) {
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
  onNewQuery,
  onShowProof,
  onShowHelp,
  isCreatingNewSession,
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

      // Calculate new width (panel is on the right, so we measure from right edge)
      const newWidth = window.innerWidth - e.clientX
      const clampedWidth = Math.min(Math.max(newWidth, MIN_PANEL_WIDTH), MAX_PANEL_WIDTH)
      setPanelWidth(clampedWidth)
    },
    [isResizing]
  )

  const handleMouseUp = useCallback(() => {
    setIsResizing(false)
    // Save to localStorage when resizing ends
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
        {/* Conversation Panel */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {conversationPanel}
        </main>

        {artifactPanelHidden ? (
          /* Show panel button when hidden */
          <button
            onClick={toggleArtifactPanel}
            className="flex items-center justify-center w-6 bg-gray-100 dark:bg-gray-800 border-l border-gray-200 dark:border-gray-700 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-gray-500 dark:text-gray-400"
            title="Show artifact panel"
          >
            <span className="text-xs font-bold">&laquo;</span>
          </button>
        ) : (
          <>
            {/* Resize Handle with hide button */}
            <div className="relative flex flex-col items-center">
              <button
                onClick={toggleArtifactPanel}
                className="absolute -left-3 top-2 z-10 w-6 h-6 flex items-center justify-center rounded-full bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-500 dark:text-gray-400 text-xs font-bold shadow-sm transition-colors"
                title="Hide artifact panel"
              >
                &raquo;
              </button>
              <div
                className={`w-1 h-full cursor-col-resize hover:bg-primary-400 transition-colors ${
                  isResizing ? 'bg-primary-500' : 'bg-gray-200 dark:bg-gray-700'
                }`}
                onMouseDown={handleMouseDown}
              />
            </div>

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

      {/* Toolbar */}
      <Toolbar onNewQuery={onNewQuery} onShowProof={onShowProof} onShowHelp={onShowHelp} isCreatingNewSession={isCreatingNewSession} />
    </div>
  )
}
