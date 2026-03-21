// Bottom command strip for reason-chain mode — replaces chat input

import type { ProofDAGActions } from './ProofDAGPanel'

interface ReasonChainCommandStripProps {
  onExplore: () => void
  isProofComplete: boolean
  isSummaryGenerating?: boolean
  hasSummary?: boolean
  hasSessionId?: boolean
  hasResultNode?: boolean
  proofActions?: React.RefObject<ProofDAGActions | null>
}

export function ReasonChainCommandStrip({
  onExplore,
  isProofComplete,
  isSummaryGenerating,
  hasSummary,
  hasSessionId,
  hasResultNode,
  proofActions,
}: ReasonChainCommandStripProps) {
  return (
    <div className="px-4 py-2.5 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 flex items-center justify-between">
      <button
        onClick={onExplore}
        className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
      >
        Explore
      </button>
      {isProofComplete && (
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => proofActions?.current?.showRedoForm()}
            className="px-3 py-1.5 text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition-colors"
          >
            Redo
          </button>
          {hasResultNode && (
            <button
              onClick={() => proofActions?.current?.showFinalResult()}
              className="px-3 py-1.5 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
            >
              Final
            </button>
          )}
          <button
            onClick={() => proofActions?.current?.showSummary()}
            disabled={!hasSummary}
            className={`px-3 py-1.5 text-sm font-medium rounded-lg transition-colors flex items-center gap-1.5 ${
              hasSummary
                ? 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20'
                : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
            }`}
            title={hasSummary ? 'View summary' : isSummaryGenerating ? 'Generating...' : 'Not available'}
          >
            {isSummaryGenerating && (
              <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            )}
            Summary
          </button>
          <div className="w-px h-4 bg-gray-200 dark:bg-gray-700 mx-0.5" />
          {hasSessionId && (
            <>
              <button
                onClick={() => proofActions?.current?.showSkillForm()}
                className="px-3 py-1.5 text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 rounded-lg transition-colors"
              >
                + Skill
              </button>
              <button
                onClick={() => proofActions?.current?.showTestForm()}
                className="px-3 py-1.5 text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition-colors"
              >
                + Test
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
