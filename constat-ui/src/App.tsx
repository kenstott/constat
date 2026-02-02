// Main App component

import { useEffect, useRef, useState } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { FullscreenArtifactModal } from '@/components/artifacts/FullscreenArtifactModal'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { LoginPage } from '@/components/auth/LoginPage'
import { ProofDAGPanel } from '@/components/proof/ProofDAGPanel'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import { useProofStore } from '@/store/proofStore'
import * as sessionsApi from '@/api/sessions'

const SESSION_STORAGE_KEY = 'constat-session-id'
const SPLASH_MIN_DURATION = 1500 // Minimum splash screen duration in ms

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

function SplashScreen() {
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-gray-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-900 dark:to-blue-950">
      {/* Logo */}
      <div className="relative mb-6">
        {/* Animated glow ring */}
        <div className="absolute inset-0 w-24 h-24 rounded-full bg-blue-500/20 dark:bg-blue-400/20 animate-pulse" />
        {/* Logo container */}
        <div className="relative w-24 h-24 flex items-center justify-center rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 shadow-xl">
          {/* Stylized "V" logo with data nodes */}
          <svg
            viewBox="0 0 64 64"
            className="w-14 h-14 text-white"
            fill="none"
            stroke="currentColor"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Main "V" shape */}
            <path d="M12 12L32 52L52 12" />
            {/* Data connection dots at vertices */}
            <circle cx="12" cy="12" r="4" fill="currentColor" stroke="none" className="animate-pulse" />
            <circle cx="52" cy="12" r="4" fill="currentColor" stroke="none" className="animate-pulse" style={{ animationDelay: '0.2s' }} />
            <circle cx="32" cy="52" r="5" fill="currentColor" stroke="none" className="animate-pulse" style={{ animationDelay: '0.4s' }} />
          </svg>
        </div>
      </div>

      {/* App name */}
      <h1 className="text-4xl font-bold text-gray-900 dark:text-white tracking-tight mb-2">
        Vera
      </h1>

      {/* Tagline */}
      <p className="text-lg text-gray-500 dark:text-gray-400 mb-8">
        Powered by Constat AI Reasoning Engine
      </p>

      {/* Loading indicator */}
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
        <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
      </div>

      {/* Copyright */}
      <p className="absolute bottom-6 text-xs text-gray-400 dark:text-gray-500">
        2025 Kenneth Stott
      </p>
    </div>
  )
}

function MainApp() {
  const { session, wsConnected, createSession, messages } = useSessionStore()
  const { userId } = useAuthStore()
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

    // Include userId in storage key for user-specific session restoration
    const storageKey = isAuthDisabled ? SESSION_STORAGE_KEY : `${SESSION_STORAGE_KEY}-${userId}`

    // Try to restore from localStorage (persists across browser refreshes)
    const savedSessionId = localStorage.getItem(storageKey)
    if (savedSessionId) {
      // Try to reconnect to existing session - fetch session and messages in parallel
      Promise.all([
        sessionsApi.getSession(savedSessionId),
        sessionsApi.getMessages(savedSessionId).catch(() => ({ messages: [] })),
      ])
        .then(([restoredSession, messagesResult]) => {
          // Restore messages BEFORE connecting WebSocket (prevents welcome message overwrite)
          if (messagesResult.messages && messagesResult.messages.length > 0) {
            const restoredMessages = messagesResult.messages.map(m => ({
              ...m,
              timestamp: new Date(m.timestamp),
            }))
            useSessionStore.setState({ messages: restoredMessages, suggestions: [], plan: null })
          }
          // Set session with preserveMessages to avoid clearing restored messages
          useSessionStore.getState().setSession(restoredSession, { preserveMessages: true })
        })
        .catch(() => {
          // Session no longer exists on server, create new one
          localStorage.removeItem(storageKey)
          createSession(userId).then(() => {
            const newSession = useSessionStore.getState().session
            if (newSession) {
              localStorage.setItem(storageKey, newSession.session_id)
            }
          })
        })
        .finally(() => {
          initializingRef.current = false
        })
    } else {
      // No saved session, create new one
      createSession(userId).then(() => {
        const newSession = useSessionStore.getState().session
        if (newSession) {
          localStorage.setItem(storageKey, newSession.session_id)
        }
      }).finally(() => {
        initializingRef.current = false
      })
    }
  }, [session, createSession, userId])

  const handleNewQuery = async () => {
    // Clear conversation state but keep the same session (preserves DBs, projects, entities, learnings)
    useProofStore.getState().clearFacts()

    // Clear conversation-related state only
    useSessionStore.setState({
      messages: [],
      suggestions: [],
      plan: null,
      queuedMessages: [],
      clarification: null,
      status: 'idle',
      executionPhase: 'idle',
      currentStepNumber: 0,
      stepAttempt: 1,
      stepMessageIds: {},
      liveMessageId: null,
      thinkingMessageId: null,
      lastQueryStartStep: 0,
      queryContext: null,
    })

    // Clear query-produced artifacts (tables, artifacts, facts, step codes) but NOT data sources/entities
    const artifactStore = useArtifactStore.getState()
    artifactStore.clearQueryResults()

    // Reset saved messages state for this session
    lastSavedRef.current = ''

    // Clear persisted messages on server for this session
    if (session) {
      sessionsApi.saveMessages(session.session_id, []).catch(err =>
        console.error('Failed to clear messages:', err)
      )
    }

    queryInputRef.current?.focus()
  }

  // Proof panel state
  const { facts: proofFacts, isPanelOpen: isProofPanelOpen, openPanel: openProofPanel, closePanel: closeProofPanel } = useProofStore()

  const handleShowProof = () => {
    openProofPanel()
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
      <ProofDAGPanel
        isOpen={isProofPanelOpen}
        onClose={closeProofPanel}
        facts={proofFacts}
      />
    </>
  )
}

function App() {
  const { initialize, loading, initialized } = useAuthStore()
  const isAuthenticated = useAuthStore((state) => {
    if (isAuthDisabled) return true
    return state.user !== null
  })

  // Track splash screen timing
  const [splashStartTime] = useState(() => Date.now())
  const [minTimeElapsed, setMinTimeElapsed] = useState(false)

  // Initialize auth on mount
  useEffect(() => {
    initialize()
  }, [initialize])

  // Ensure splash screen shows for minimum duration
  useEffect(() => {
    const elapsed = Date.now() - splashStartTime
    const remaining = SPLASH_MIN_DURATION - elapsed

    if (remaining <= 0) {
      setMinTimeElapsed(true)
    } else {
      const timer = setTimeout(() => setMinTimeElapsed(true), remaining)
      return () => clearTimeout(timer)
    }
  }, [splashStartTime])

  // Show splash screen while auth is initializing OR minimum time hasn't elapsed
  if (!initialized || loading || !minTimeElapsed) {
    return <SplashScreen />
  }

  // Show login page if not authenticated (and auth is enabled)
  if (!isAuthDisabled && !isAuthenticated) {
    return <LoginPage />
  }

  // User is authenticated (or auth disabled), show main app
  return <MainApp />
}

export default App