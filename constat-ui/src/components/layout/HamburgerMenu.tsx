// Hamburger Menu (drawer) component

import { Fragment, useEffect, useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import {
  TableCellsIcon,
  CodeBracketIcon,
  CircleStackIcon,
  DocumentTextIcon,
  ClockIcon,
  BookOpenIcon,
  ArrowUpTrayIcon,
  LinkIcon,
  ServerIcon,
  Cog6ToothIcon,
  ChatBubbleLeftRightIcon,
  PlusIcon,
} from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'
import type { Session } from '@/types/api'

interface MenuItem {
  name: string
  icon: React.ComponentType<{ className?: string }>
  command: string
  description: string
}

const menuItems: MenuItem[] = [
  {
    name: 'Tables',
    icon: TableCellsIcon,
    command: '/tables',
    description: 'View session tables',
  },
  {
    name: 'Code',
    icon: CodeBracketIcon,
    command: '/code',
    description: 'Show generated code',
  },
  {
    name: 'Query',
    icon: CircleStackIcon,
    command: '/query',
    description: 'Run SQL query',
  },
  {
    name: 'Facts',
    icon: DocumentTextIcon,
    command: '/facts',
    description: 'View resolved facts',
  },
  {
    name: 'History',
    icon: ClockIcon,
    command: '/history',
    description: 'Session history',
  },
  {
    name: 'Learnings',
    icon: BookOpenIcon,
    command: '/learnings',
    description: 'View learnings',
  },
  {
    name: 'Upload File',
    icon: ArrowUpTrayIcon,
    command: '/add',
    description: 'Upload a file',
  },
  {
    name: 'Add File Ref',
    icon: LinkIcon,
    command: '/file',
    description: 'Add file reference',
  },
  {
    name: 'Database',
    icon: ServerIcon,
    command: '/database',
    description: 'Manage databases',
  },
]

interface HamburgerMenuProps {
  onCommand?: (command: string) => void
  onNewSession?: () => void
}

export function HamburgerMenu({ onCommand, onNewSession }: HamburgerMenuProps) {
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
          const sorted = [...response.sessions].sort(
            (a, b) => new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime()
          )
          setSessions(sorted)
        })
        .catch(console.error)
        .finally(() => setLoadingSessions(false))
    }
  }, [menuOpen])

  const handleCommand = (command: string) => {
    onCommand?.(command)
    setMenuOpen(false)
  }

  const handleSwitchSession = async (sessionId: string) => {
    if (sessionId === currentSession?.session_id) {
      setMenuOpen(false)
      return
    }
    try {
      const session = await sessionsApi.getSession(sessionId)

      // Clear current state and set new session
      useArtifactStore.getState().clear()
      setSession(session)

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
                        {loadingSessions ? (
                          <p className="text-xs text-gray-400 py-2">Loading sessions...</p>
                        ) : sessions.length === 0 ? (
                          <p className="text-xs text-gray-400 py-2">No sessions yet</p>
                        ) : (
                          sessions.map((session) => (
                            <button
                              key={session.session_id}
                              onClick={() => handleSwitchSession(session.session_id)}
                              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-colors ${
                                session.session_id === currentSession?.session_id
                                  ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                                  : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                              }`}
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

                    {/* Commands section */}
                    <div className="px-4 pt-3 pb-1">
                      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                        Commands
                      </h3>
                    </div>
                    <nav className="flex-1 px-2 pb-4 space-y-1">
                      {menuItems.map((item) => (
                        <button
                          key={item.command}
                          onClick={() => handleCommand(item.command)}
                          className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        >
                          <item.icon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                              {item.name}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                              {item.description}
                            </p>
                          </div>
                          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono">
                            {item.command}
                          </span>
                        </button>
                      ))}
                    </nav>

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