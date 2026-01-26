// Main App component

import { useEffect, useRef } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { FullscreenArtifactModal } from '@/components/artifacts/FullscreenArtifactModal'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'

const SESSION_STORAGE_KEY = 'constat-session-id'

function ConnectingOverlay() {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="flex flex-col items-center gap-4">
        {/* Spinner */}
        <div className="relative">
          <div className="w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full" />
          <div className="absolute top-0 left-0 w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
        {/* Text */}
        <div className="text-center">
          <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
            Connecting to Constat
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Initializing session...
          </p>
        </div>
      </div>
    </div>
  )
}

function App() {
  const { session, wsConnected, createSession, messages } = useSessionStore()
  const queryInputRef = useRef<HTMLTextAreaElement>(null)
  const initializingRef = useRef(false)

  // Debounce timer ref for message persistence
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastSavedRef = useRef<string>('')

  // Persist messages to server whenever they change (debounced)
  useEffect(() => {
    if (!session || messages.length === 0) return

    // Filter out transient messages (live, pending) and save stable ones
    const stableMessages = messages.filter(m => !m.isLive && !m.isPending)
    if (stableMessages.length === 0) return

    // Serialize for comparison
    const serialized = JSON.stringify(stableMessages.map(m => ({
      id: m.id,
      type: m.type,
      content: m.content,
      timestamp: m.timestamp.toISOString(),
      stepNumber: m.stepNumber,
      isFinalInsight: m.isFinalInsight,
    })))

    // Skip if nothing changed
    if (serialized === lastSavedRef.current) return

    // Debounce saves to avoid hammering the server
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
    }

    saveTimerRef.current = setTimeout(() => {
      const messagesToSave = stableMessages.map(m => ({
        id: m.id,
        type: m.type,
        content: m.content,
        timestamp: m.timestamp.toISOString(),
        stepNumber: m.stepNumber,
        isFinalInsight: m.isFinalInsight,
      }))
      sessionsApi.saveMessages(session.session_id, messagesToSave)
        .then(() => {
          lastSavedRef.current = serialized
        })
        .catch(err => console.error('Failed to save messages:', err))
    }, 1000) // Save after 1 second of inactivity

    return () => {
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
    }
  }, [session, messages])

  // Create or restore session on mount
  useEffect(() => {
    // Guard against double initialization (React Strict Mode, race conditions)
    if (session || initializingRef.current) {
      return
    }
    initializingRef.current = true

    // Try to restore from localStorage (persists across browser refreshes)
    const savedSessionId = localStorage.getItem(SESSION_STORAGE_KEY)
    if (savedSessionId) {
      // Try to reconnect to existing session
      sessionsApi.getSession(savedSessionId)
        .then(async (restoredSession) => {
          useSessionStore.getState().setSession(restoredSession)
          // Restore conversation messages from server
          try {
            const { messages: serverMessages } = await sessionsApi.getMessages(savedSessionId)
            if (serverMessages && serverMessages.length > 0) {
              // Restore timestamps as Date objects
              const restoredMessages = serverMessages.map(m => ({
                ...m,
                timestamp: new Date(m.timestamp),
              }))
              useSessionStore.setState({ messages: restoredMessages })
            }
          } catch (e) {
            console.error('Failed to restore messages from server:', e)
          }
        })
        .catch(() => {
          // Session no longer exists on server, create new one
          localStorage.removeItem(SESSION_STORAGE_KEY)
          createSession().then(() => {
            const newSession = useSessionStore.getState().session
            if (newSession) {
              localStorage.setItem(SESSION_STORAGE_KEY, newSession.session_id)
            }
          })
        })
        .finally(() => {
          initializingRef.current = false
        })
    } else {
      // No saved session, create new one
      createSession().then(() => {
        const newSession = useSessionStore.getState().session
        if (newSession) {
          localStorage.setItem(SESSION_STORAGE_KEY, newSession.session_id)
        }
      }).finally(() => {
        initializingRef.current = false
      })
    }
  }, [session, createSession])

  const handleNewQuery = async () => {
    // Clear artifact store and create a new session (equivalent to /reset)
    useArtifactStore.getState().clear()
    localStorage.removeItem(SESSION_STORAGE_KEY)
    lastSavedRef.current = '' // Reset saved state for new session
    await createSession()
    const newSession = useSessionStore.getState().session
    if (newSession) {
      localStorage.setItem(SESSION_STORAGE_KEY, newSession.session_id)
    }
    queryInputRef.current?.focus()
  }

  const handleShowProof = () => {
    // TODO: Open proof tree dialog
    console.log('Show proof tree')
  }

  // Show connecting overlay until session exists and WebSocket is connected
  if (!session || !wsConnected) {
    return <ConnectingOverlay />
  }

  return (
    <>
      <MainLayout
        conversationPanel={<ConversationPanel />}
        artifactPanel={<ArtifactPanel />}
        onNewQuery={handleNewQuery}
        onShowProof={handleShowProof}
      />
      <ClarificationDialog />
      <PlanApprovalDialog />
      <FullscreenArtifactModal />
    </>
  )
}

export default App