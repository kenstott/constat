// Main App component

import { useEffect, useRef } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { useSessionStore } from '@/store/sessionStore'

function App() {
  const { session, createSession } = useSessionStore()
  const queryInputRef = useRef<HTMLTextAreaElement>(null)

  // Create session on mount if none exists
  useEffect(() => {
    if (!session) {
      createSession()
    }
  }, [session, createSession])

  const handleCommand = (command: string) => {
    // Handle menu commands
    console.log('Command:', command)
    // Could dispatch to a command handler or inject into query input
  }

  const handleNewQuery = () => {
    // Focus query input
    queryInputRef.current?.focus()
  }

  return (
    <>
      <MainLayout
        conversationPanel={<ConversationPanel />}
        artifactPanel={<ArtifactPanel />}
        onCommand={handleCommand}
        onNewQuery={handleNewQuery}
      />
      <ClarificationDialog />
      <PlanApprovalDialog />
    </>
  )
}

export default App