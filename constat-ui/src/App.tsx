// Main App component

import { useEffect, useRef, useState } from 'react'
import { MainLayout } from '@/components/layout/MainLayout'
import { ConversationPanel } from '@/components/conversation/ConversationPanel'
import { ArtifactPanel } from '@/components/artifacts/ArtifactPanel'
import { FullscreenArtifactModal } from '@/components/artifacts/FullscreenArtifactModal'
import { ClarificationDialog } from '@/components/conversation/ClarificationDialog'
import { PlanApprovalDialog } from '@/components/conversation/PlanApprovalDialog'
import { LoginPage } from '@/components/auth/LoginPage'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
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
          {/* Stylized "C" logo with data nodes */}
          <svg
            viewBox="0 0 64 64"
            className="w-14 h-14 text-white"
            fill="none"
            stroke="currentColor"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            {/* Main "C" shape */}
            <path d="M44 18C40 12 33 8 26 8C14 8 8 18 8 32C8 46 14 56 26 56C33 56 40 52 44 46" />
            {/* Data connection dots */}
            <circle cx="48" cy="20" r="4" fill="currentColor" stroke="none" className="animate-pulse" />
            <circle cx="52" cy="32" r="4" fill="currentColor" stroke="none" className="animate-pulse" style={{ animationDelay: '0.2s' }} />
            <circle cx="48" cy="44" r="4" fill="currentColor" stroke="none" className="animate-pulse" style={{ animationDelay: '0.4s' }} />
            {/* Connection lines */}
            <path d="M44 18L48 20M44 32L52 32M44 46L48 44" strokeWidth="2" className="opacity-60" />
          </svg>
        </div>
      </div>

      {/* App name */}
      <h1 className="text-4xl font-bold text-gray-900 dark:text-white tracking-tight mb-2">
        Constat
      </h1>

      {/* Tagline */}
      <p className="text-lg text-gray-500 dark:text-gray-400 mb-8">
        AI-powered data analysis
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
    // Clear artifact store and create a new session (equivalent to /reset)
    useArtifactStore.getState().clear()
    const storageKey = isAuthDisabled ? SESSION_STORAGE_KEY : `${SESSION_STORAGE_KEY}-${userId}`
    localStorage.removeItem(storageKey)
    lastSavedRef.current = '' // Reset saved state for new session
    await createSession(userId)
    const newSession = useSessionStore.getState().session
    if (newSession) {
      localStorage.setItem(storageKey, newSession.session_id)
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