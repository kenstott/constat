// Status Bar component

import { useSessionStore } from '@/store/sessionStore'
import { useUIStore } from '@/store/uiStore'
import { useArtifactStore } from '@/store/artifactStore'
import {
  Bars3Icon,
  SignalIcon,
  SignalSlashIcon,
  SunIcon,
  MoonIcon,
  CircleStackIcon,
  LightBulbIcon,
  ChartBarIcon,
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
  const { session, status, wsConnected } = useSessionStore()
  const { theme, setTheme, toggleMenu } = useUIStore()
  const { databases, apis, documents, facts, artifacts } = useArtifactStore()

  // Count datasources (databases + APIs + documents)
  const datasourceCount = databases.length + apis.length + documents.length

  // Count visualizations (charts, diagrams, images)
  const visualizationTypes = ['chart', 'diagram', 'image', 'html', 'vega']
  const visualizationCount = artifacts.filter(a =>
    visualizationTypes.some(t => a.type?.toLowerCase().includes(t))
  ).length

  const toggleTheme = () => {
    const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
    setTheme(next)
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

      {/* Counts */}
      {session && (
        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center gap-1" title="Data Sources">
            <CircleStackIcon className="w-4 h-4" />
            <span>{datasourceCount}</span>
          </div>
          <div className="flex items-center gap-1" title="Facts">
            <LightBulbIcon className="w-4 h-4" />
            <span>{facts.length}</span>
          </div>
          <div className="flex items-center gap-1" title="Visualizations">
            <ChartBarIcon className="w-4 h-4" />
            <span>{visualizationCount}</span>
          </div>
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