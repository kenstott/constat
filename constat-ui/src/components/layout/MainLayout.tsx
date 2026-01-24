// Main Layout component

import { ReactNode } from 'react'
import { StatusBar } from './StatusBar'
import { HamburgerMenu } from './HamburgerMenu'
import { Toolbar } from './Toolbar'

interface MainLayoutProps {
  conversationPanel: ReactNode
  artifactPanel: ReactNode
  onCommand?: (command: string) => void
  onNewQuery?: () => void
}

export function MainLayout({
  conversationPanel,
  artifactPanel,
  onCommand,
  onNewQuery,
}: MainLayoutProps) {
  return (
    <div className="h-full flex flex-col">
      {/* Status Bar */}
      <StatusBar />

      {/* Hamburger Menu (drawer) */}
      <HamburgerMenu onCommand={onCommand} />

      {/* Main content area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Conversation Panel */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {conversationPanel}
        </main>

        {/* Artifact Panel */}
        <aside className="w-96 border-l border-gray-200 dark:border-gray-700 flex flex-col overflow-hidden bg-gray-50 dark:bg-gray-900">
          {artifactPanel}
        </aside>
      </div>

      {/* Toolbar */}
      <Toolbar onNewQuery={onNewQuery} />
    </div>
  )
}