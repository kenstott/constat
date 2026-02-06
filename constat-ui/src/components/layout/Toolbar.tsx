// Toolbar component

import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import {
  PlusIcon,
  StopIcon,
  QuestionMarkCircleIcon,
  CircleStackIcon,
  LightBulbIcon,
  DocumentIcon,
  CheckBadgeIcon,
} from '@heroicons/react/24/outline'

interface ToolbarProps {
  onNewQuery?: () => void
  onShowProof?: () => void
  onShowHelp?: () => void
  isCreatingNewSession?: boolean
}

export function Toolbar({ onNewQuery, onShowProof, onShowHelp, isCreatingNewSession }: ToolbarProps) {
  const { session, status, cancelExecution } = useSessionStore()
  const { databases, apis, documents, facts, artifacts, tables, stepCodes } = useArtifactStore()

  // Count datasources (databases + APIs + documents)
  const datasourceCount = databases.length + apis.length + documents.length

  // Count results (tables + non-code/error/output artifacts) - matches Results accordion
  const excludedArtifactTypes = new Set(['code', 'error', 'output', 'table'])
  const resultArtifacts = artifacts.filter((a) => !excludedArtifactTypes.has(a.artifact_type))
  const resultsCount = tables.length + resultArtifacts.length

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

        {isExecuting && (
          <button
            onClick={cancelExecution}
            className="btn-secondary text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20"
          >
            <StopIcon className="w-4 h-4 mr-1" />
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

      {/* Stats */}
      {session && (
        <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
          <div className="flex items-center gap-1" title="Data Sources: Connected databases, APIs, and documents available for querying">
            <CircleStackIcon className="w-4 h-4" />
            <span>{datasourceCount}</span>
          </div>
          <div className="flex items-center gap-1" title="Results: Tables, charts, and other analysis outputs">
            <DocumentIcon className="w-4 h-4" />
            <span>{resultsCount}</span>
          </div>
          <div className="flex items-center gap-1" title="Facts: Discovered insights and computed values stored for reference">
            <LightBulbIcon className="w-4 h-4" />
            <span>{facts.length}</span>
          </div>
        </div>
      )}

      {/* Help */}
      <button onClick={onShowHelp} className="btn-ghost text-xs">
        <QuestionMarkCircleIcon className="w-4 h-4 mr-1" />
        Help
      </button>
    </footer>
  )
}