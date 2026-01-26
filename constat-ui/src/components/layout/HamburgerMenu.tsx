// Hamburger Menu (drawer) component

import { Fragment, useEffect, useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import {
  Cog6ToothIcon,
  ChatBubbleLeftRightIcon,
  PlusIcon,
} from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'
import type { Session } from '@/types/api'

interface HamburgerMenuProps {
  onNewSession?: () => void
}

export function HamburgerMenu({ onNewSession }: HamburgerMenuProps) {
  const { menuOpen, setMenuOpen, theme, setTheme } = useUIStore()
  const { session: currentSession, setSession, createSession } = useSessionStore()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)

  // Fetch sessions when menu opens
  useEffect(() => {
    if (menuOpen) {
      setLoadingSessions(true)
      sessionsApi.listSessions()
        .then((response) => {
          // Sort by last_activity descending (most recent first)
          // Exclude current session - it will be shown separately at the top
          // Exclude empty sessions (no tables and no query executed)
          const sorted = [...response.sessions]
            .filter(s => s.session_id !== currentSession?.session_id)
            .filter(s => s.tables_count > 0 || s.current_query)
            .sort(
              (a, b) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime()
            )
          setSessions(sorted)
        })
        .catch(console.error)
        .finally(() => setLoadingSessions(false))
    }
  }, [menuOpen, currentSession?.session_id])

  const handleSwitchSession = async (sessionId: string) => {
    if (sessionId === currentSession?.session_id) {
      setMenuOpen(false)
      return
    }
    try {
      const session = await sessionsApi.getSession(sessionId)

      // Clear current state and set new session
      useArtifactStore.getState().clear()
      useSessionStore.getState().clearMessages()
      setSession(session)

      // Update localStorage with new session ID
      localStorage.setItem('constat-session-id', sessionId)

      // Fetch all session data to restore state
      const artifactStore = useArtifactStore.getState()
      await Promise.all([
        artifactStore.fetchTables(sessionId),
        artifactStore.fetchArtifacts(sessionId),
        artifactStore.fetchFacts(sessionId),
        artifactStore.fetchEntities(sessionId),
        artifactStore.fetchDataSources(sessionId),
        artifactStore.fetchStepCodes(sessionId),
      ])

      // Restore conversation messages from server
      try {
        const { messages } = await sessionsApi.getMessages(sessionId)
        if (messages && messages.length > 0) {
          const restoredMessages = messages.map(m => ({
            ...m,
            timestamp: new Date(m.timestamp),
          }))
          useSessionStore.setState({ messages: restoredMessages })
        }
      } catch (e) {
        console.error('Failed to restore messages:', e)
      }

      setMenuOpen(false)
    } catch (error) {
      console.error('Failed to switch session:', error)
    }
  }

  const handleNewSession = async () => {
    useArtifactStore.getState().clear()
    await createSession()
    setMenuOpen(false)
    onNewSession?.()
  }

  // Format session title: use summary if available, otherwise current_query, otherwise session_id
  const getSessionTitle = (session: Session) => {
    if (session.summary) return session.summary
    if (session.current_query) {
      const query = session.current_query
      return query.length > 40 ? query.slice(0, 40) + '...' : query
    }
    return `Session ${session.session_id.slice(0, 8)}`
  }

  // Format relative time
  const formatRelativeTime = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  return (
    <Transition.Root show={menuOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={setMenuOpen}>
        <Transition.Child
          as={Fragment}
          enter="ease-in-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in-out duration-300"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-gray-500/75 dark:bg-gray-900/75 transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 left-0 flex max-w-full pr-10">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-300"
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-300"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-xs">
                  <div className="flex h-full flex-col overflow-y-auto bg-white dark:bg-gray-800 shadow-xl">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-4 border-b border-gray-200 dark:border-gray-700">
                      <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                        Menu
                      </Dialog.Title>
                      <button
                        onClick={() => setMenuOpen(false)}
                        className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                      >
                        <XMarkIcon className="w-5 h-5 text-gray-500" />
                      </button>
                    </div>

                    {/* Sessions section */}
                    <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                          Sessions
                        </h3>
                        <button
                          onClick={handleNewSession}
                          className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                          title="New session"
                        >
                          <PlusIcon className="w-4 h-4" />
                        </button>
                      </div>
                      <div className="space-y-1 max-h-64 overflow-y-auto">
                        {/* Current session - always shown first */}
                        {currentSession && (
                          <button
                            key={currentSession.session_id}
                            onClick={() => setMenuOpen(false)}
                            className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300"
                          >
                            <ChatBubbleLeftRightIcon className="w-4 h-4 flex-shrink-0 text-primary-500" />
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-medium truncate">
                                {getSessionTitle(currentSession)}
                              </p>
                              <p className="text-xs opacity-75">
                                Current session
                                {currentSession.tables_count > 0 && ` · ${currentSession.tables_count} tables`}
                              </p>
                            </div>
                          </button>
                        )}
                        {/* Historical sessions */}
                        {loadingSessions ? (
                          <p className="text-xs text-gray-400 py-2">Loading sessions...</p>
                        ) : sessions.length === 0 && !currentSession ? (
                          <p className="text-xs text-gray-400 py-2">No sessions yet</p>
                        ) : (
                          sessions.map((session) => (
                            <button
                              key={session.session_id}
                              onClick={() => handleSwitchSession(session.session_id)}
                              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors hover:bg-gray-100 dark:hover:bg-gray-700"
                            >
                              <ChatBubbleLeftRightIcon className="w-4 h-4 flex-shrink-0 text-gray-400" />
                              <div className="flex-1 min-w-0">
                                <p className="text-sm font-medium truncate text-gray-900 dark:text-gray-100">
                                  {getSessionTitle(session)}
                                </p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">
                                  {formatRelativeTime(session.last_activity)}
                                  {session.tables_count > 0 && ` · ${session.tables_count} tables`}
                                </p>
                              </div>
                            </button>
                          ))
                        )}
                      </div>
                    </div>

                    {/* Settings */}
                    <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Cog6ToothIcon className="w-5 h-5 text-gray-500" />
                          <span className="text-sm text-gray-700 dark:text-gray-300">
                            Theme
                          </span>
                        </div>
                        <select
                          value={theme}
                          onChange={(e) => setTheme(e.target.value as 'light' | 'dark' | 'system')}
                          className="text-sm bg-gray-100 dark:bg-gray-700 border-0 rounded-md px-2 py-1"
                        >
                          <option value="light">Light</option>
                          <option value="dark">Dark</option>
                          <option value="system">System</option>
                        </select>
                      </div>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  )
}