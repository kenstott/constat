// Toolbar component

import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useUIStore } from '@/store/uiStore'
import {
  PlusIcon,
  StopIcon,
  QuestionMarkCircleIcon,
  CheckBadgeIcon,
  BoltIcon,
} from '@heroicons/react/24/outline'

interface ToolbarProps {
  onNewQuery?: () => void
  onShowProof?: () => void
  onShowHelp?: () => void
  isCreatingNewSession?: boolean
}

export function Toolbar({ onNewQuery, onShowProof, onShowHelp, isCreatingNewSession }: ToolbarProps) {
  const { status, cancelExecution } = useSessionStore()
  const { tables, stepCodes } = useArtifactStore()
  const { briefMode, toggleBriefMode } = useUIStore()

  const isExecuting = status === 'planning' || status === 'executing'

  // Proof button is only enabled after at least one plan has been executed
  const hasExecutedPlan = stepCodes.length > 0 || tables.length > 0

  return (
    <footer className="h-12 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 flex items-center px-4 gap-4">
      {/* Quick actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onNewQuery}
          disabled={isExecuting || isCreatingNewSession}
          className="btn-secondary text-xs disabled:opacity-50"
        >
          {isCreatingNewSession ? (
            <svg className="w-4 h-4 mr-1 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          ) : (
            <PlusIcon className="w-4 h-4 mr-1" />
          )}
          {isCreatingNewSession ? 'Creating...' : 'New Query'}
        </button>

        <button
          onClick={onShowProof}
          disabled={isExecuting || !hasExecutedPlan}
          className="btn-secondary text-xs disabled:opacity-50"
          title={hasExecutedPlan ? "Show proof tree / reasoning chain" : "Execute a query first to view proof"}
        >
          <CheckBadgeIcon className="w-4 h-4 mr-1" />
          Proof
        </button>

        <button
          onClick={toggleBriefMode}
          className={`btn-secondary text-xs ${briefMode ? 'bg-primary-100 dark:bg-primary-900/30 border-primary-400 dark:border-primary-600 text-primary-700 dark:text-primary-300' : ''}`}
          title={briefMode ? "Brief mode ON — skipping insight synthesis" : "Brief mode OFF — full insight synthesis"}
        >
          <BoltIcon className="w-4 h-4 mr-1" />
          Brief
        </button>

        {isExecuting && (
          <button
            onClick={cancelExecution}
            className="btn-secondary text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 animate-pulse border-red-300 dark:border-red-700"
          >
            <StopIcon className="w-4 h-4 mr-1 animate-spin" style={{ animationDuration: '3s' }} />
            Cancel
          </button>
        )}

        {/* Input hints */}
        <span className="text-xs text-gray-400 dark:text-gray-500 ml-2">
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-[10px]">Enter</kbd> send
          <span className="mx-1">·</span>
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-[10px]">/</kbd> commands
        </span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Help */}
      <button onClick={onShowHelp} className="btn-ghost text-xs">
        <QuestionMarkCircleIcon className="w-4 h-4 mr-1" />
        Help
      </button>
    </footer>
  )
}