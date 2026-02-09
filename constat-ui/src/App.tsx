// Main App component

import { useCallback, useEffect, useRef, useState } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { FullscreenArtifactModal } from '@/components/artifacts/FullscreenArtifactModal'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { LoginPage } from '@/components/auth/LoginPage'
import { ProofDAGPanel } from '@/components/proof/ProofDAGPanel'
import { HelpModal } from '@/components/help/HelpModal'
import { useSessionStore } from '@/store/sessionStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import { useProofStore } from '@/store/proofStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'

const SPLASH_MIN_DURATION = 1500 // Minimum splash screen duration in ms

// Initialization phases for granular status display
type InitPhase =
  | 'creating_session'
  | 'connecting_websocket'
  | 'ready'

const INIT_PHASE_MESSAGES: Record<InitPhase, { title: string; detail: string }> = {
  creating_session: { title: 'Connecting to Constat', detail: 'Starting session...' },
  connecting_websocket: { title: 'Connecting', detail: 'Establishing real-time connection...' },
  ready: { title: 'Ready', detail: '' },
}

function ConnectingOverlay({ phase }: { phase: InitPhase }) {
  const { title, detail } = INIT_PHASE_MESSAGES[phase]
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
            {title}
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            {detail}
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
  const { fetchAllSkills } = useArtifactStore()
  const queryInputRef = useRef<HTMLTextAreaElement>(null)
  const initializingRef = useRef(false)
  const [initPhase, setInitPhase] = useState<InitPhase>('creating_session')

  // Debounce timer ref for message persistence
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastSavedRef = useRef<string>('')
  const pendingMessagesRef = useRef<{ sessionId: string; messages: object[] } | null>(null)

  // Helper to save messages immediately (used by debounce and beforeunload)
  const saveMessagesNow = useCallback(() => {
    const pending = pendingMessagesRef.current
    if (!pending) return

    // Use sendBeacon for reliability during page unload
    const blob = new Blob([JSON.stringify({ messages: pending.messages })], { type: 'application/json' })
    navigator.sendBeacon(`/api/sessions/${pending.sessionId}/messages`, blob)
    pendingMessagesRef.current = null
  }, [])

  // Save messages immediately on page unload (refresh, close, navigate away)
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Cancel debounced save and save immediately
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current)
      }
      saveMessagesNow()
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    return () => window.removeEventListener('beforeunload', handleBeforeUnload)
  }, [saveMessagesNow])

  // Persist messages to server whenever they change (debounced)
  useEffect(() => {
    if (!session || messages.length === 0) return

    // Filter out transient messages (live, pending) and save stable ones
    const stableMessages = messages.filter(m => !m.isLive && !m.isPending)
    if (stableMessages.length === 0) return

    // Serialize for comparison
    const messagesToSave = stableMessages.map(m => ({
      id: m.id,
      type: m.type,
      content: m.content,
      timestamp: m.timestamp.toISOString(),
      stepNumber: m.stepNumber,
      isFinalInsight: m.isFinalInsight,
    }))
    const serialized = JSON.stringify(messagesToSave)

    // Skip if nothing changed
    if (serialized === lastSavedRef.current) return

    // Store pending messages for beforeunload handler
    pendingMessagesRef.current = { sessionId: session.session_id, messages: messagesToSave }

    // Debounce saves to avoid hammering the server
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current)
    }

    saveTimerRef.current = setTimeout(() => {
      sessionsApi.saveMessages(session.session_id, messagesToSave)
        .then(() => {
          lastSavedRef.current = serialized
          pendingMessagesRef.current = null
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
  // Session ID is managed by sessions.ts (localStorage) and server handles reconnection
  useEffect(() => {
    // Guard against double initialization (React Strict Mode, race conditions)
    if (session || initializingRef.current) {
      return
    }
    initializingRef.current = true
    setInitPhase('creating_session')

    // createSession handles:
    // 1. Getting/creating session ID from localStorage (per user)
    // 2. Sending to server (which reconnects if exists, or creates new)
    // 3. Connecting WebSocket
    createSession(userId)
      .then(() => {
        setInitPhase('connecting_websocket')
      })
      .finally(() => {
        initializingRef.current = false
      })
  }, [session, createSession, userId])

  const handleNewQuery = async () => {
    setIsCreatingNewSession(true)
    try {
      // Create a brand new session (preserves old session in history)
      useProofStore.getState().clearFacts()
      lastSavedRef.current = ''

      // Create new session - this preserves the old session in history
      await createSession(userId, true) // forceNew = true

      queryInputRef.current?.focus()
    } finally {
      setIsCreatingNewSession(false)
    }
  }

  // Proof panel state
  const { facts: proofFacts, isPanelOpen: isProofPanelOpen, isPlanningComplete, proofSummary, isSummaryGenerating, openPanel: openProofPanel, closePanel: closeProofPanel, clearFacts } = useProofStore()
  const { submitQuery } = useSessionStore()

  const handleShowProof = () => {
    // Clear previous proof state and open panel
    clearFacts()
    openProofPanel()
    // Submit /prove command to trigger proof execution
    submitQuery('/prove', true)
  }

  // Help modal state
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const handleShowHelp = () => setIsHelpOpen(true)

  // New query loading state
  const [isCreatingNewSession, setIsCreatingNewSession] = useState(false)

  // Show connecting overlay until session exists and WebSocket is connected
  if (!session || !wsConnected) {
    return <ConnectingOverlay phase={initPhase} />
  }

  return (
    <>
      <MainLayout
        conversationPanel={<ConversationPanel />}
        artifactPanel={<ArtifactPanel />}
        onNewQuery={handleNewQuery}
        onShowProof={handleShowProof}
        onShowHelp={handleShowHelp}
        isCreatingNewSession={isCreatingNewSession}
      />
      <ClarificationDialog />
      <PlanApprovalDialog />
      <FullscreenArtifactModal />
      <ProofDAGPanel
        isOpen={isProofPanelOpen}
        onClose={closeProofPanel}
        facts={proofFacts}
        isPlanningComplete={isPlanningComplete}
        summary={proofSummary}
        isSummaryGenerating={isSummaryGenerating}
        sessionId={session?.session_id}
        onSkillCreated={() => fetchAllSkills()}
        onRedo={() => {
          clearFacts()
          submitQuery('/prove', true)
        }}
      />
      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
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