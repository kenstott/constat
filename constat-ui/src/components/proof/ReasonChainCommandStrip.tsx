// Copyright (c) 2025 Kenneth Stott
// Canary: 70b4ef55-3e29-4a3d-9a75-28f63ff5d323
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Bottom command strip for reason-chain mode — replaces chat input

import { useState } from 'react'
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

const Spinner = () => (
  <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24" fill="none">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
  </svg>
)

export function ReasonChainCommandStrip({
  onExplore,
  isProofComplete,
  isSummaryGenerating,
  hasSummary,
  hasSessionId,
  hasResultNode,
  proofActions,
}: ReasonChainCommandStripProps) {
  const [isSavingTest, setIsSavingTest] = useState(false)
  const [isSavingSkill, setIsSavingSkill] = useState(false)

  const handleTestClick = async () => {
    if (isSavingTest) return
    setIsSavingTest(true)
    try {
      await proofActions?.current?.showTestForm()
    } finally {
      setIsSavingTest(false)
    }
  }

  const handleSkillClick = () => {
    if (isSavingSkill) return
    setIsSavingSkill(true)
    proofActions?.current?.showSkillForm()
    // Skill form is a modal — reset after a short delay since showSkillForm is sync
    setTimeout(() => setIsSavingSkill(false), 300)
  }

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
            {isSummaryGenerating && <Spinner />}
            Summary
          </button>
          <div className="w-px h-4 bg-gray-200 dark:bg-gray-700 mx-0.5" />
          {hasSessionId && (
            <>
              <button
                onClick={handleSkillClick}
                disabled={isSavingSkill}
                className="px-3 py-1.5 text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                {isSavingSkill && <Spinner />}
                + Skill
              </button>
              <button
                onClick={handleTestClick}
                disabled={isSavingTest}
                className="px-3 py-1.5 text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
              >
                {isSavingTest && <Spinner />}
                + Test
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
