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
  onNewQuery,
  onShowProof,
  onShowHelp,
  isCreatingNewSession,
}: MainLayoutProps) {
  const [panelWidth, setPanelWidth] = useState(getSavedPanelWidth)
  const [isResizing, setIsResizing] = useState(false)
  const conversationPanelHidden = useUIStore((s) => s.conversationPanelHidden)
  const toggleConversationPanel = useUIStore((s) => s.toggleConversationPanel)
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
      // Clamp so neither panel goes below MIN_PANEL_WIDTH
      const newWidth = window.innerWidth - e.clientX
      const maxWidth = window.innerWidth - MIN_PANEL_WIDTH
      const clampedWidth = Math.min(Math.max(newWidth, MIN_PANEL_WIDTH), maxWidth)
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
        {conversationPanelHidden ? null : (
          <main className="flex-1 flex flex-col overflow-hidden relative">
            {conversationPanel}
          </main>
        )}

        {/* Vertical panel toggle strip */}
        <div className="flex flex-col items-center justify-center gap-1 px-0.5 bg-gray-100 dark:bg-gray-800 border-x border-gray-200 dark:border-gray-700">
          <button
            onClick={toggleConversationPanel}
            className="w-5 h-5 flex items-center justify-center rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-500 dark:text-gray-400 text-xs font-bold transition-colors"
            title={conversationPanelHidden ? 'Show conversation panel' : 'Hide conversation panel'}
          >
            {conversationPanelHidden ? <span>&raquo;</span> : <span>&laquo;</span>}
          </button>
          <button
            onClick={toggleArtifactPanel}
            className="w-5 h-5 flex items-center justify-center rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-500 dark:text-gray-400 text-xs font-bold transition-colors"
            title={artifactPanelHidden ? 'Show artifact panel' : 'Hide artifact panel'}
          >
            {artifactPanelHidden ? <span>&laquo;</span> : <span>&raquo;</span>}
          </button>
        </div>

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
              className={`border-l border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900 ${conversationPanelHidden ? 'flex-1' : ''}`}
              style={conversationPanelHidden ? undefined : { width: panelWidth }}
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
