// Hamburger Menu (drawer) component

import { Fragment, useEffect, useState } from 'react'
import { Dialog, Transition } from '@headlessui/react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import {
  Cog6ToothIcon,
  ChatBubbleLeftRightIcon,
  PlusIcon,
  PencilIcon,
  ArrowRightOnRectangleIcon,
  UserCircleIcon,
} from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import * as sessionsApi from '@/api/sessions'
import type { Session } from '@/types/api'
import type { ProjectInfo } from '@/api/sessions'

interface HamburgerMenuProps {
  onNewSession?: () => void
}

function AccountSection({ onClose }: { onClose: () => void }) {
  const { user, logout } = useAuthStore()
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
  const { menuOpen, setMenuOpen, theme, setTheme } = useUIStore()
  const { session: currentSession, setSession, updateSession, createSession } = useSessionStore()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loadingSessions, setLoadingSessions] = useState(false)
  const [projects, setProjects] = useState<ProjectInfo[]>([])
  const [loadingProjects, setLoadingProjects] = useState(false)

  // Loading state for session operations
  const [switchingSessionId, setSwitchingSessionId] = useState<string | null>(null)
  const [creatingSession, setCreatingSession] = useState(false)

  // Project editor modal state
  const [editingProject, setEditingProject] = useState<string | null>(null)
  const [projectContent, setProjectContent] = useState('')
  const [projectPath, setProjectPath] = useState('')
  const [savingProject, setSavingProject] = useState(false)
  const [projectSaveError, setProjectSaveError] = useState<string | null>(null)

  // Fetch sessions, projects, and skills when menu opens
  useEffect(() => {
    if (menuOpen) {
      // Fetch sessions
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

      // Fetch projects
      setLoadingProjects(true)
      sessionsApi.listProjects()
        .then((response) => {
          setProjects(response.projects)
        })
        .catch(console.error)
        .finally(() => setLoadingProjects(false))
    }
  }, [menuOpen, currentSession?.session_id])

  const handleSwitchSession = async (sessionId: string) => {
    if (sessionId === currentSession?.session_id) {
      setMenuOpen(false)
      return
    }

    setSwitchingSessionId(sessionId)
    try {
      console.log('[switchSession] Switching to session:', sessionId)
      // Use createSession to restore/reconnect - it handles both in-memory and historical sessions
      // getSession only looks in memory and returns 404 for historical sessions
      const [session, messagesResult] = await Promise.all([
        sessionsApi.createSession(currentSession?.user_id || 'default', sessionId),
        sessionsApi.getMessages(sessionId).catch((err) => {
          console.warn('[switchSession] Failed to fetch messages:', err)
          return { messages: [] }
        }),
      ])
      console.log('[switchSession] Restored session:', session.session_id)
      console.log('[switchSession] Fetched messages:', messagesResult.messages?.length || 0)

      // Clear current state
      useArtifactStore.getState().clear()

      // Restore messages BEFORE connecting WebSocket (prevents welcome message overwrite)
      if (messagesResult.messages && messagesResult.messages.length > 0) {
        const restoredMessages = messagesResult.messages.map(m => ({
          id: m.id,
          type: m.type as 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking',
          content: m.content,
          timestamp: new Date(m.timestamp),
          stepNumber: m.stepNumber,
          isFinalInsight: m.isFinalInsight,
        }))
        console.log('[switchSession] Restored messages:', restoredMessages.length)
        useSessionStore.setState({ messages: restoredMessages, suggestions: [], plan: null })
      } else {
        console.log('[switchSession] No messages to restore, clearing')
        useSessionStore.getState().clearMessages()
      }

      // Set session with preserveMessages to avoid clearing restored messages
      setSession(session, { preserveMessages: true })

      // Update localStorage with user-specific session key
      sessionsApi.storeSessionId(sessionId, session.user_id)

      // Fetch all session data to restore state (parallel for speed)
      const artifactStore = useArtifactStore.getState()
      await Promise.all([
        artifactStore.fetchTables(sessionId),
        artifactStore.fetchArtifacts(sessionId),
        artifactStore.fetchFacts(sessionId),
        artifactStore.fetchEntities(sessionId),
        artifactStore.fetchDataSources(sessionId),
        artifactStore.fetchStepCodes(sessionId),
        artifactStore.fetchInferenceCodes(sessionId),
        artifactStore.fetchLearnings(),
        artifactStore.fetchAllRoles(sessionId),
        artifactStore.fetchPromptContext(sessionId),
      ])

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

  const [projectError, setProjectError] = useState<string | null>(null)

  const handleToggleProject = async (projectFilename: string) => {
    if (!currentSession) return

    setProjectError(null)
    const currentProjects = currentSession.active_projects || []
    const isSelected = currentProjects.includes(projectFilename)

    // Toggle: remove if selected, add if not
    const newProjects = isSelected
      ? currentProjects.filter(p => p !== projectFilename)
      : [...currentProjects, projectFilename]

    try {
      await sessionsApi.setActiveProjects(currentSession.session_id, newProjects)
      // Update local session state without clearing messages or reconnecting WebSocket
      updateSession({ active_projects: newProjects })
      // Refresh data sources and entities to show merged sources
      const artifactStore = useArtifactStore.getState()
      await artifactStore.fetchDataSources(currentSession.session_id)
      // Entities refresh via entity_rebuild_complete WS event
    } catch (error: unknown) {
      console.error('Failed to set projects:', error)
      // Handle conflict errors
      if (error && typeof error === 'object' && 'message' in error) {
        const errObj = error as { message?: string; response?: { data?: { detail?: { message?: string; conflicts?: string[] } } } }
        const detail = errObj.response?.data?.detail
        if (detail && typeof detail === 'object' && detail.conflicts) {
          setProjectError(detail.conflicts.join('\n'))
        } else {
          setProjectError(errObj.message || 'Failed to update projects')
        }
      }
    }
  }

  // Open project editor
  const handleEditProject = async (filename: string) => {
    try {
      setProjectSaveError(null)
      const result = await sessionsApi.getProjectContent(filename)
      setProjectContent(result.content)
      setProjectPath(result.path)
      setEditingProject(filename)
    } catch (error) {
      console.error('Failed to load project:', error)
    }
  }

  // Save project content
  const handleSaveProject = async () => {
    if (!editingProject) return

    setSavingProject(true)
    setProjectSaveError(null)
    try {
      await sessionsApi.updateProjectContent(editingProject, projectContent)
      // Refresh projects list
      const response = await sessionsApi.listProjects()
      setProjects(response.projects)
      setEditingProject(null)
    } catch (error: unknown) {
      console.error('Failed to save project:', error)
      if (error && typeof error === 'object' && 'message' in error) {
        setProjectSaveError((error as { message: string }).message)
      } else {
        setProjectSaveError('Failed to save project')
      }
    } finally {
      setSavingProject(false)
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
                          Sessions
                        </h3>
                        <button
                          onClick={handleNewSession}
                          disabled={creatingSession || switchingSessionId !== null}
                          className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          title="New session"
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

                    {/* Projects section */}
                    {projects.length > 0 && (
                      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                          Projects
                        </h3>
                        {projectError && (
                          <div className="mb-2 p-2 text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 rounded-md">
                            {projectError}
                          </div>
                        )}
                        <div className="space-y-1">
                          {loadingProjects ? (
                            <p className="text-xs text-gray-400 py-2">Loading projects...</p>
                          ) : (
                            <>
                              {/* Project options with checkboxes */}
                              {projects.map((project) => {
                                const isSelected = currentSession?.active_projects?.includes(project.filename) ?? false
                                return (
                                  <div
                                    key={project.filename}
                                    className={`flex items-center gap-2 px-2 py-1.5 rounded-md transition-colors ${
                                      isSelected
                                        ? 'bg-primary-100 dark:bg-primary-900/30'
                                        : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                                    }`}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={isSelected}
                                      onChange={() => handleToggleProject(project.filename)}
                                      className="w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500 dark:bg-gray-700"
                                    />
                                    <button
                                      onClick={() => handleToggleProject(project.filename)}
                                      className="flex-1 min-w-0 text-left"
                                    >
                                      <p className={`text-sm font-medium truncate ${
                                        isSelected
                                          ? 'text-primary-700 dark:text-primary-300'
                                          : 'text-gray-900 dark:text-gray-100'
                                      }`}>
                                        {project.name}
                                      </p>
                                      {project.description && (
                                        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                                          {project.description}
                                        </p>
                                      )}
                                    </button>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        handleEditProject(project.filename)
                                      }}
                                      className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-600"
                                      title="Edit project YAML"
                                    >
                                      <PencilIcon className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
                                    </button>
                                  </div>
                                )
                              })}
                            </>
                          )}
                        </div>
                      </div>
                    )}

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

      {/* Project Editor Modal */}
      <Transition.Root show={editingProject !== null} as={Fragment}>
        <Dialog as="div" className="relative z-[60]" onClose={() => setEditingProject(null)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-500/75 dark:bg-gray-900/75 transition-opacity" />
          </Transition.Child>

          <div className="fixed inset-0 z-10 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-3xl transform overflow-hidden rounded-lg bg-white dark:bg-gray-800 shadow-xl transition-all">
                  <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                    <Dialog.Title className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                      Edit Project: {editingProject}
                    </Dialog.Title>
                    <button
                      onClick={() => setEditingProject(null)}
                      className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
                    >
                      <XMarkIcon className="w-5 h-5 text-gray-500" />
                    </button>
                  </div>

                  <div className="p-4">
                    <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                      {projectPath}
                    </p>
                    <textarea
                      value={projectContent}
                      onChange={(e) => setProjectContent(e.target.value)}
                      className="w-full h-96 font-mono text-sm p-3 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      spellCheck={false}
                    />
                    {projectSaveError && (
                      <p className="mt-2 text-sm text-red-600 dark:text-red-400">
                        {projectSaveError}
                      </p>
                    )}
                  </div>

                  <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-2">
                    <button
                      onClick={() => setEditingProject(null)}
                      className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSaveProject}
                      disabled={savingProject}
                      className="px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors"
                    >
                      {savingProject ? 'Saving...' : 'Save'}
                    </button>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition.Root>
    </>
  )
}