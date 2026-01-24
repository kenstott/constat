// Toolbar component

import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import {
  PlusIcon,
  StopIcon,
  TrashIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline'

interface ToolbarProps {
  onNewQuery?: () => void
}

export function Toolbar({ onNewQuery }: ToolbarProps) {
  const { session, status, cancelExecution, clearMessages } = useSessionStore()
  const { tables, artifacts } = useArtifactStore()

  const isExecuting = status === 'planning' || status === 'executing'

  return (
    <footer className="h-12 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 flex items-center px-4 gap-4">
      {/* Quick actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onNewQuery}
          disabled={isExecuting}
          className="btn-secondary text-xs disabled:opacity-50"
        >
          <PlusIcon className="w-4 h-4 mr-1" />
          New Query
        </button>

        {isExecuting && (
          <button
            onClick={cancelExecution}
            className="btn-secondary text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
          >
            <StopIcon className="w-4 h-4 mr-1" />
            Cancel
          </button>
        )}

        <button
          onClick={clearMessages}
          disabled={isExecuting}
          className="btn-ghost text-xs disabled:opacity-50"
        >
          <TrashIcon className="w-4 h-4 mr-1" />
          Clear
        </button>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Stats */}
      {session && (
        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span>{tables.length} tables</span>
          <span>{artifacts.length} artifacts</span>
        </div>
      )}

      {/* Help */}
      <button className="btn-ghost text-xs">
        <QuestionMarkCircleIcon className="w-4 h-4 mr-1" />
        Help
      </button>
    </footer>
  )
}