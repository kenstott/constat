// Copyright (c) 2025 Kenneth Stott
// Canary: b267904f-97dc-4e27-b330-21cb8ef1b130
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Hamburger Menu (drawer) component

import { Fragment, useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import {
  Cog6ToothIcon,
  ChatBubbleLeftRightIcon,
  PlusIcon,
  ArrowRightOnRectangleIcon,
  UserCircleIcon,
  ChevronRightIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline'
import { useQuery } from '@apollo/client'
import { useReactiveVar } from '@apollo/client'
import { menuOpenVar, themeVar, setTheme } from '@/graphql/ui-state'
import { useSessionContext } from '@/contexts/SessionContext'
import { useAuth } from '@/contexts/AuthContext'
import type { Session } from '@/types/api'
import DomainPanel from '@/components/artifacts/DomainPanel'
import { SESSIONS_QUERY, toSession } from '@/graphql/operations/sessions'

interface HamburgerMenuProps {
  onNewSession?: () => void
}

function AccountSection({ onClose }: { onClose: () => void }) {
  const { user, logout } = useAuth()
  const [imageError, setImageError] = useState(false)

  const handleLogout = async () => {
    await logout()
    onClose()
    // Clear session storage
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('constat-session-id')) {
        localStorage.removeItem(key)
      }
    })
  }

  if (!user) return null

  return (
    <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
      <div className="flex items-center gap-3 mb-3">
        {user.photoURL && !imageError ? (
          <img
            src={user.photoURL}
            alt={user.displayName || 'User'}
            className="w-8 h-8 rounded-full"
            onError={() => setImageError(true)}
          />
        ) : (
          <UserCircleIcon className="w-8 h-8 text-gray-400" />
        )}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
            {user.displayName || 'User'}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
            {user.email}
          </p>
        </div>
      </div>
      <button
        onClick={handleLogout}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
      >
        <ArrowRightOnRectangleIcon className="w-5 h-5" />
        Sign out
      </button>
    </div>
  )
}

export function HamburgerMenu({ onNewSession }: HamburgerMenuProps) {
  const menuOpen = useReactiveVar(menuOpenVar)
  const setMenuOpen = (v: boolean) => menuOpenVar(v)
  const theme = useReactiveVar(themeVar)
  const { session: currentSession, createSession, switchSession } = useSessionContext()
  const { isAuthDisabled } = useAuth()
  // Loading state for session operations
  const [switchingSessionId, setSwitchingSessionId] = useState<string | null>(null)
  const [creatingSession, setCreatingSession] = useState(false)
  const [domainCollapsed, setDomainCollapsed] = useState(false)

  // Fetch sessions via GraphQL when menu opens
  const { data: sessionsData, loading: loadingSessions } = useQuery(SESSIONS_QUERY, {
    skip: !menuOpen,
    fetchPolicy: 'network-only',
  })

  // Sort by last_activity descending, exclude current session and empty sessions
  const sessions: Session[] = (sessionsData?.sessions?.sessions || [])
    .map(toSession)
    .filter((s: Session) => s.session_id !== currentSession?.session_id)
    .filter((s: Session) => s.tables_count > 0 || s.current_query)
    .sort(
      (a: Session, b: Session) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime()
    )

  const handleSwitchSession = async (sessionId: string) => {
    if (sessionId === currentSession?.session_id) {
      setMenuOpen(false)
      return
    }

    setSwitchingSessionId(sessionId)
    try {
      await switchSession(sessionId)
      setMenuOpen(false)
    } catch (error) {
      console.error('Failed to switch session:', error)
    } finally {
      setSwitchingSessionId(null)
    }
  }

  const handleNewSession = async () => {
    setCreatingSession(true)
    try {
      // createSession with forceNew=true generates a new session ID
      await createSession(undefined, true)
      setMenuOpen(false)
      onNewSession?.()
    } finally {
      setCreatingSession(false)
    }
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
    <>
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
                          Query History
                        </h3>
                        <button
                          onClick={handleNewSession}
                          disabled={creatingSession || switchingSessionId !== null}
                          className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          title="New query"
                        >
                          {creatingSession ? (
                            <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                            </svg>
                          ) : (
                            <PlusIcon className="w-4 h-4" />
                          )}
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
                          sessions.map((session) => {
                            const isLoading = switchingSessionId === session.session_id
                            return (
                              <button
                                key={session.session_id}
                                onClick={() => handleSwitchSession(session.session_id)}
                                disabled={switchingSessionId !== null || creatingSession}
                                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors ${
                                  isLoading
                                    ? 'bg-primary-50 dark:bg-primary-900/20'
                                    : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                                } disabled:opacity-50 disabled:cursor-not-allowed`}
                              >
                                {isLoading ? (
                                  <svg className="w-4 h-4 flex-shrink-0 text-primary-500 animate-spin" viewBox="0 0 24 24" fill="none">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                  </svg>
                                ) : (
                                  <ChatBubbleLeftRightIcon className="w-4 h-4 flex-shrink-0 text-gray-400" />
                                )}
                                <div className="flex-1 min-w-0">
                                  <p className="text-sm font-medium truncate text-gray-900 dark:text-gray-100">
                                    {getSessionTitle(session)}
                                  </p>
                                  <p className="text-xs text-gray-500 dark:text-gray-400">
                                    {isLoading ? 'Restoring...' : formatRelativeTime(session.last_activity)}
                                    {!isLoading && session.tables_count > 0 && ` · ${session.tables_count} tables`}
                                  </p>
                                </div>
                              </button>
                            )
                          })
                        )}
                      </div>
                    </div>

                    {/* Domain */}
                    <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                      <button
                        onClick={() => setDomainCollapsed(!domainCollapsed)}
                        className="w-full flex items-center justify-between mb-2"
                      >
                        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide flex items-center gap-1.5">
                          <GlobeAltIcon className="w-4 h-4" />
                          Domain
                        </h3>
                        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${domainCollapsed ? '' : 'rotate-90'}`} />
                      </button>
                      {!domainCollapsed && <DomainPanel />}
                    </div>

                    {/* Settings */}
                    <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-4 space-y-4">
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

                      {/* User account section (only when auth enabled) */}
                      {!isAuthDisabled && (
                        <AccountSection onClose={() => setMenuOpen(false)} />
                      )}
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition.Root>

    </>
  )
}