// Copyright (c) 2025 Kenneth Stott
// Canary: c43e32a7-5427-4481-b931-b5ef333307b1
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState } from 'react'
import {
  ChevronRightIcon,
  LightBulbIcon,
  QuestionMarkCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline'
import type { ObjectivesEntry } from '@/types/api'
import { AccordionSection } from '../ArtifactAccordion'

interface ObjectivesSectionProps {
  objectives: ObjectivesEntry[]
}

export function ObjectivesSection({ objectives }: ObjectivesSectionProps) {
  const [objectivesCollapsed, setObjectivesCollapsed] = useState(() =>
    localStorage.getItem('constat-objectives-collapsed') === 'true'
  )

  if (objectives.length === 0) return null

  return (
    <>
      {/* --- Objectives sub-group --- */}
      <button
        onClick={() => {
          const newVal = !objectivesCollapsed
          setObjectivesCollapsed(newVal)
          localStorage.setItem('constat-objectives-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Objectives
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${objectivesCollapsed ? '' : 'rotate-90'}`} />
      </button>

      {!objectivesCollapsed && (
      <>

      {/* Question */}
      {objectives.filter(o => o.type === 'question').length > 0 && (
        <AccordionSection
          id="objectives-question"
          title="Question"
          icon={<LightBulbIcon className="w-4 h-4" />}
        >
          {objectives.filter(o => o.type === 'question').map((o, i) => (
            <p key={i} className="text-sm text-gray-700 dark:text-gray-300" style={{ whiteSpace: 'pre-line' }}>{o.text}</p>
          ))}
        </AccordionSection>
      )}

      {/* Clarifications */}
      {objectives.filter(o => o.type === 'clarification').length > 0 && (
        <AccordionSection
          id="objectives-clarifications"
          title="Clarifications"
          count={objectives.filter(o => o.type === 'clarification').length}
          icon={<QuestionMarkCircleIcon className="w-4 h-4" />}
        >
          <dl className="space-y-2">
            {objectives.filter(o => o.type === 'clarification').map((o, i) => (
              <div key={i} className="text-sm">
                <dt className="font-medium text-gray-700 dark:text-gray-300">Q: {o.question}</dt>
                <dd className="ml-4 text-gray-500 dark:text-gray-400">A: {o.answer}</dd>
              </div>
            ))}
          </dl>
        </AccordionSection>
      )}

      {/* Redo History */}
      {objectives.filter(o => o.type === 'redo').length > 0 && (
        <AccordionSection
          id="objectives-redos"
          title="Redos"
          count={objectives.filter(o => o.type === 'redo').length}
          icon={<ArrowPathIcon className="w-4 h-4" />}
        >
          <div className="space-y-2">
            {objectives.filter(o => o.type === 'redo').map((o, i) => (
              <div key={i} className="text-sm flex items-start gap-2">
                <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                  o.mode === 'auditable'
                    ? 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300'
                    : 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                }`}>{o.mode}</span>
                <span className="text-gray-700 dark:text-gray-300">{o.guidance || '(no guidance)'}</span>
              </div>
            ))}
          </div>
        </AccordionSection>
      )}

      </>
      )}
    </>
  )
}
