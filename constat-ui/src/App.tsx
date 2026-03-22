// Main App component

import { useCallback, useEffect, useRef, useState } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { FullscreenArtifactModal } from '@/components/artifacts/FullscreenArtifactModal'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { LoginPage } from '@/components/auth/LoginPage'
import { ProofDAGPanel, type ProofDAGActions } from '@/components/proof/ProofDAGPanel'
import { ReasonChainCommandStrip } from '@/components/proof/ReasonChainCommandStrip'
import { HelpModal } from '@/components/help/HelpModal'
import { useSessionStore } from '@/store/sessionStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import { useProofStore } from '@/store/proofStore'
import { useArtifactStore } from '@/store/artifactStore'
import { pathToDeepLink, applyDeepLink, useUIStore } from '@/store/uiStore'
import { ToastContainer } from '@/components/common/ToastContainer'
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

/** Handles URL-based deep links on initial load and browser back/forward. */
function NavigationSync() {
  const location = useLocation()
  const navigate = useNavigate()
  const initialRef = useRef(true)

  // On mount and location change (popstate / direct visit), apply deep link from URL
  useEffect(() => {
    const link = pathToDeepLink(location.pathname)
    if (link) {
      applyDeepLink(link)
      // Replace to / so the deep link URL doesn't stick
      navigate('/', { replace: true })
    }
    initialRef.current = false
  }, [location.pathname, navigate])

  // Listen for popstate (browser back/forward) — pushState in navigateTo
  // doesn't trigger React Router's location update, so we need this listener
  useEffect(() => {
    const handlePopState = () => {
      const link = pathToDeepLink(window.location.pathname)
      if (link) {
        applyDeepLink(link)
        window.history.replaceState(null, '', '/')
      }
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  return null
}

/** Read-only viewer for publicly shared sessions. */
function PublicViewerApp({ sessionId }: { sessionId: string }) {
  const { clearPublicSession } = useUIStore()
  const [summary, setSummary] = useState<string | null>(null)
  const [messages, setMessages] = useState<Array<{ id: string; type: string; content: string; timestamp: string }>>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const [sessionResp, messagesResp] = await Promise.all([
          sessionsApi.publicGetSession(sessionId),
          sessionsApi.publicGetMessages(sessionId),
        ])
        if (cancelled) return
        setSummary(sessionResp.summary)
        setMessages(messagesResp.messages || [])
      } catch {
        if (!cancelled) setError('This session is not available or not publicly shared.')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [sessionId])

  if (loading) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full animate-spin border-t-blue-500" />
          <p className="text-gray-500 dark:text-gray-400">Loading shared session...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="text-center max-w-md">
          <p className="text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">Session Not Available</p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">{error}</p>
          <button
            onClick={clearPublicSession}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go to App
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 flex flex-col bg-gray-50 dark:bg-gray-900">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 flex items-center justify-center rounded-lg bg-blue-500 text-white font-bold text-sm">V</div>
          <div>
            <h1 className="text-sm font-semibold text-gray-800 dark:text-gray-200">Shared Session</h1>
            {summary && <p className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-md">{summary}</p>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded">
            Read-only
          </span>
          <button
            onClick={clearPublicSession}
            className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
          >
            Go to App
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full">
        {messages.length === 0 ? (
          <p className="text-center text-gray-500 dark:text-gray-400 py-8">No messages in this session.</p>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`rounded-lg px-4 py-3 ${
                msg.type === 'user'
                  ? 'bg-blue-50 dark:bg-blue-900/20 ml-8'
                  : 'bg-white dark:bg-gray-800 mr-8 border border-gray-200 dark:border-gray-700'
              }`}
            >
              <div className="text-xs text-gray-400 dark:text-gray-500 mb-1">
                {msg.type === 'user' ? 'User' : msg.type === 'output' ? 'Result' : msg.type}
              </div>
              <div className="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">
                {msg.content}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function MainApp() {
  const { session, wsConnected, createSession, messages } = useSessionStore()
  const { userId } = useAuthStore()
  const { fetchAllSkills } = useArtifactStore()
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
      stepDurationMs: m.stepDurationMs,
      role: m.role,
      skills: m.skills,
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
    useUIStore.getState().initPreferences()
    createSession(userId)
      .then(() => {
        setInitPhase('connecting_websocket')
      })
      .finally(() => {
        initializingRef.current = false
      })
  }, [session, createSession, userId])

  // Proof panel state
  const { facts: proofFacts, isPanelOpen: isProofPanelOpen, isPlanningComplete, proofSummary, isSummaryGenerating, closePanel: closeProofPanel, clearFacts, isProving, hasCompletedProof } = useProofStore()
  const { submitQuery } = useSessionStore()
  const uiMode = useUIStore((s) => s.uiMode)
  const exitReasonChainMode = useUIStore((s) => s.exitReasonChainMode)

  // Help modal state
  const [isHelpOpen, setIsHelpOpen] = useState(false)
  const proofActionsRef = useRef<ProofDAGActions>(null)

  // Show connecting overlay until session exists and WebSocket is connected
  if (!session || !wsConnected) {
    return <ConnectingOverlay phase={initPhase} />
  }

  const isReasonChain = uiMode === 'reason-chain'
  const proofComplete = !isProving && hasCompletedProof

  const handleRedo = (guidance?: string) => {
    clearFacts()
    submitQuery(guidance ? `/reason ${guidance}` : '/reason', true)
  }

  // Explore: exit reason-chain mode. If proof incomplete, abandon (clear facts).
  const handleExplore = () => {
    if (!proofComplete) clearFacts()
    closeProofPanel()
    exitReasonChainMode()
  }

  // Check if result node exists for "Final" button
  const hasResultNode = (() => {
    if (proofFacts.size === 0) return false
    const entries = Array.from(proofFacts.values())
    // Final node = no other node depends on it
    const depsOf = new Set(entries.flatMap(n => n.dependencies))
    const roots = entries.filter(n => !depsOf.has(n.id))
    return roots.length > 0 && roots[0].status === 'resolved'
  })()

  const reasonChainPanel = (
    <div className="flex-1 flex flex-col overflow-hidden">
      <ProofDAGPanel
        embedded
        isOpen
        onClose={() => {}}
        facts={proofFacts}
        isPlanningComplete={isPlanningComplete}
        summary={proofSummary}
        isSummaryGenerating={isSummaryGenerating}
        sessionId={session?.session_id}
        onSkillCreated={() => fetchAllSkills()}
        onRedo={handleRedo}
        actionsRef={proofActionsRef}
      />
      <ReasonChainCommandStrip
        onExplore={handleExplore}
        isProofComplete={proofComplete}
        isSummaryGenerating={isSummaryGenerating}
        hasSummary={!!proofSummary}
        hasSessionId={!!session?.session_id}
        hasResultNode={hasResultNode}
        proofActions={proofActionsRef}
      />
    </div>
  )

  return (
    <>
      <NavigationSync />
      <MainLayout
        conversationPanel={isReasonChain ? reasonChainPanel : <ConversationPanel />}
        artifactPanel={<ArtifactPanel />}
      />
      <ClarificationDialog />
      <PlanApprovalDialog />
      <FullscreenArtifactModal />
      {!isReasonChain && (
        <ProofDAGPanel
          isOpen={isProofPanelOpen}
          onClose={closeProofPanel}
          facts={proofFacts}
          isPlanningComplete={isPlanningComplete}
          summary={proofSummary}
          isSummaryGenerating={isSummaryGenerating}
          sessionId={session?.session_id}
          onSkillCreated={() => fetchAllSkills()}
          onRedo={handleRedo}
        />
      )}
      <HelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
      />
    </>
  )
}

function App() {
  const { initialize, loading, initialized } = useAuthStore()
  const publicSessionId = useUIStore((state) => state.publicSessionId)
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

  // Public session viewer (no auth required)
  if (publicSessionId) {
    return <PublicViewerApp sessionId={publicSessionId} />
  }

  // Show splash screen while auth is initializing OR minimum time hasn't elapsed
  if (!initialized || loading || !minTimeElapsed) {
    return <SplashScreen />
  }

  // Show login page if not authenticated (and auth is enabled)
  if (!isAuthDisabled && !isAuthenticated) {
    return <LoginPage />
  }

  // All routes render the same MainApp — deep links are handled by NavigationSync
  return (
    <>
      <Routes>
        <Route path="/*" element={<MainApp />} />
      </Routes>
      <ToastContainer />
    </>
  )
}

export default App
