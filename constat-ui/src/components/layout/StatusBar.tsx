// Status Bar component

import { useEffect, useState, useRef } from 'react'
import { useSessionStore } from '@/store/sessionStore'
import { useUIStore } from '@/store/uiStore'
import {
  Bars3Icon,
  SignalIcon,
  SignalSlashIcon,
  SunIcon,
  MoonIcon,
  UserCircleIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline'

const statusColors: Record<string, string> = {
  idle: 'bg-gray-400',
  planning: 'bg-yellow-400 animate-pulse',
  awaiting_approval: 'bg-orange-400',
  executing: 'bg-blue-400 animate-pulse',
  completed: 'bg-green-400',
  error: 'bg-red-400',
  cancelled: 'bg-gray-400',
}

export function StatusBar() {
  const { session, status, wsConnected, roles, currentRole, fetchRoles, setRole } = useSessionStore()
  const { theme, setTheme, toggleMenu } = useUIStore()
  const [roleMenuOpen, setRoleMenuOpen] = useState(false)
  const roleMenuRef = useRef<HTMLDivElement>(null)

  // Fetch roles when session changes
  useEffect(() => {
    if (session) {
      fetchRoles()
    }
  }, [session, fetchRoles])

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (roleMenuRef.current && !roleMenuRef.current.contains(event.target as Node)) {
        setRoleMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
    setTheme(next)
  }

  const handleRoleSelect = (roleName: string | null) => {
    setRole(roleName)
    setRoleMenuOpen(false)
  }

  return (
    <header className="h-12 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center px-4 gap-4">
      {/* Menu button */}
      <button
        onClick={toggleMenu}
        className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
        aria-label="Toggle menu"
      >
        <Bars3Icon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
      </button>

      {/* Logo/Title */}
      <div className="flex items-center gap-2">
        <span className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          Constat
        </span>
      </div>

      {/* Session info */}
      {session && (
        <div className="flex items-center gap-3 ml-4">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${statusColors[status] || statusColors.idle}`}
            />
            <span className="text-sm text-gray-600 dark:text-gray-400 capitalize">
              {status.replace('_', ' ')}
            </span>
          </div>
          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono">
            {session.session_id.slice(0, 8)}...
          </span>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Role selector */}
      {session && roles.length > 0 && (
        <div className="relative" ref={roleMenuRef}>
          <button
            onClick={() => setRoleMenuOpen(!roleMenuOpen)}
            className="flex items-center gap-1.5 px-2 py-1 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-sm"
          >
            <UserCircleIcon className="w-4 h-4 text-gray-500 dark:text-gray-400" />
            <span className={currentRole ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400'}>
              {currentRole || 'No role'}
            </span>
            <ChevronDownIcon className="w-3 h-3 text-gray-400" />
          </button>

          {roleMenuOpen && (
            <div className="absolute right-0 mt-1 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50">
              <div className="py-1">
                <button
                  onClick={() => handleRoleSelect(null)}
                  className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 ${
                    !currentRole ? 'text-primary-600 dark:text-primary-400 font-medium' : 'text-gray-700 dark:text-gray-300'
                  }`}
                >
                  No role
                </button>
                {roles.map((role) => (
                  <button
                    key={role.name}
                    onClick={() => handleRoleSelect(role.name)}
                    className={`w-full text-left px-3 py-1.5 text-sm hover:bg-gray-100 dark:hover:bg-gray-700 ${
                      role.name === currentRole ? 'text-primary-600 dark:text-primary-400 font-medium' : 'text-gray-700 dark:text-gray-300'
                    }`}
                    title={role.prompt}
                  >
                    {role.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Connection status */}
      <div className="flex items-center gap-1.5">
        {wsConnected ? (
          <SignalIcon className="w-4 h-4 text-green-500" />
        ) : (
          <SignalSlashIcon className="w-4 h-4 text-red-500" />
        )}
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {wsConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* Theme toggle */}
      <button
        onClick={toggleTheme}
        className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700"
        aria-label="Toggle theme"
      >
        {theme === 'dark' ? (
          <MoonIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        ) : (
          <SunIcon className="w-5 h-5 text-gray-600 dark:text-gray-400" />
        )}
      </button>
    </header>
  )
}