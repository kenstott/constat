// Copyright (c) 2025 Kenneth Stott
// Canary: 4a90e03c-4243-4723-a491-540ba50cb6db
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useEffect, useState, useCallback } from 'react'
import { useReactiveVar } from '@apollo/client'
import {
  ChevronRightIcon,
  ChevronDownIcon,
  CircleStackIcon,
  StarIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline'
import {
  expandedSectionsVar,
  collapsedResultStepsVar,
  toggleResultStep as toggleResultStepFn,
  toggleSection,
} from '@/graphql/ui-state'
import { useTables } from '@/hooks/useTables'
import { useArtifacts } from '@/hooks/useArtifacts'
import { toTableInfo, toArtifact } from '@/graphql/operations/data'
import { TableAccordion } from '../TableAccordion'
import { ArtifactItemAccordion } from '../ArtifactItemAccordion'
import { CodeViewer } from '../CodeViewer'
import type { Message } from '@/events/types'

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const remainSec = Math.round(seconds % 60)
  return `${minutes}m ${remainSec}s`
}

/** Collapsible step code list for the Exploratory Code section */
function StepCodeList({ stepCodes, supersededStepNumbers, tables, artifacts }: {
  stepCodes: Array<{ step_number: number; goal: string; code: string; prompt?: string; model?: string }>
  supersededStepNumbers: Set<number>
  tables: Array<{ name: string; step_number: number }>
  artifacts: Array<{ id: number; name: string; step_number: number; artifact_type: string }>
}) {
  const [collapsedSteps, setCollapsedSteps] = useState<Set<number>>(() => new Set(stepCodes.map((s) => s.step_number)))
  const [showPrompt, setShowPrompt] = useState<Set<number>>(new Set())

  // Build step → outputs lookup
  const stepOutputs = new Map<number, Array<{ type: 'table' | 'artifact'; name: string; id: string }>>()
  const excludedTypes = new Set(['code', 'error', 'output', 'table'])
  for (const t of tables) {
    if (!stepOutputs.has(t.step_number)) stepOutputs.set(t.step_number, [])
    stepOutputs.get(t.step_number)!.push({ type: 'table', name: t.name, id: `table-${t.name}` })
  }
  for (const a of artifacts) {
    if (excludedTypes.has(a.artifact_type)) continue
    if (!stepOutputs.has(a.step_number)) stepOutputs.set(a.step_number, [])
    stepOutputs.get(a.step_number)!.push({ type: 'artifact', name: a.name, id: `artifact-${a.id}` })
  }

  // Collapse newly arriving step codes by default
  useEffect(() => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      for (const s of stepCodes) {
        next.add(s.step_number)
      }
      return next.size === prev.size ? prev : next
    })
  }, [stepCodes])

  const toggleStep = useCallback((stepNumber: number) => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }, [])

  const togglePrompt = useCallback((stepNumber: number) => {
    setShowPrompt((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }, [])

  return (
    <div className="space-y-1">
      {stepCodes.map((step) => {
        const isCollapsed = collapsedSteps.has(step.step_number)
        return (
          <div key={step.step_number} className={supersededStepNumbers.has(step.step_number) ? 'opacity-40' : ''}>
            <div className="flex items-center gap-1">
              <button
                onClick={() => toggleStep(step.step_number)}
                className="flex items-center gap-1 flex-1 text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
              >
                {isCollapsed ? (
                  <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
                ) : (
                  <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
                )}
                <span>Step {step.step_number}</span>
                {step.model && (
                  <span className="text-[10px] text-gray-400 dark:text-gray-500 font-mono ml-1">{step.model}</span>
                )}
              </button>
              {step.prompt && (
                <button
                  onClick={(e) => { e.stopPropagation(); togglePrompt(step.step_number) }}
                  className="ml-auto p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                  title="View generation prompt"
                >
                  <QuestionMarkCircleIcon className={`w-4 h-4 ${showPrompt.has(step.step_number) ? 'text-primary-500' : 'text-gray-400 hover:text-primary-500'}`} />
                </button>
              )}
            </div>
            {!isCollapsed && (
              <>
                {showPrompt.has(step.step_number) && step.prompt && (
                  <pre className="text-[10px] leading-tight text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded overflow-auto max-h-60 mb-1 whitespace-pre-wrap">
                    {step.prompt}
                  </pre>
                )}
                <CodeViewer
                  code={step.code}
                  language="python"
                />
                {stepOutputs.has(step.step_number) && (
                  <div className="flex flex-wrap gap-1 mt-1 px-1">
                    {stepOutputs.get(step.step_number)!.map((output) => (
                      <button
                        key={output.id}
                        onClick={() => {
                          const el = document.getElementById(`section-results`)
                          el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                          // After scroll, highlight the specific item
                          requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                              const item = document.getElementById(output.id)
                              if (item) {
                                item.scrollIntoView({ behavior: 'smooth', block: 'center' })
                                item.classList.add('ring-2', 'ring-primary-400')
                                setTimeout(() => item.classList.remove('ring-2', 'ring-primary-400'), 2000)
                              }
                            })
                          })
                        }}
                        className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-900/20 dark:text-primary-400 dark:hover:bg-primary-900/40 transition-colors"
                        title={`Jump to ${output.name}`}
                      >
                        {output.type === 'table' ? <CircleStackIcon className="w-3 h-3" /> : <StarIcon className="w-3 h-3" />}
                        {output.name}
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )
      })}
    </div>
  )
}

interface ResultsSectionProps {
  messages: Message[]
  stepCodes: Array<{ step_number: number; goal: string; code: string; prompt?: string; model?: string }>
  supersededStepNumbers: Set<number>
}

export function ResultsSection({ messages, stepCodes, supersededStepNumbers: _supersededStepNumbers }: ResultsSectionProps) {
  void _supersededStepNumbers // Used by StepCodeList (exported separately)
  const { tables } = useTables()
  const { artifacts } = useArtifacts()

  const expandedArtifactSections = useReactiveVar(expandedSectionsVar)
  const collapsedResultSteps = useReactiveVar(collapsedResultStepsVar)
  const toggleResultStep = toggleResultStepFn

  const [resultsCollapsed, setResultsCollapsed] = useState(() => {
    return localStorage.getItem('constat-results-collapsed') === 'true'
  })

  useEffect(() => {
    if (expandedArtifactSections.includes('results') && resultsCollapsed) {
      setResultsCollapsed(false)
      localStorage.setItem('constat-results-collapsed', 'false')
      // Consume the signal so collapsing works again
      toggleSection('results')
    }
  }, [expandedArtifactSections, resultsCollapsed])

  // Unified Results: combine tables and artifacts into a flat list
  type ResultItem =
    | { type: 'table'; data: typeof tables[0]; created_at: string; is_published: boolean }
    | { type: 'artifact'; data: typeof artifacts[0]; created_at: string; is_published: boolean }

  // Types to exclude when showing all (non-result artifacts)
  // Note: 'table' is excluded because tables are already shown via the tables array
  const excludedArtifactTypes = new Set(['code', 'error', 'output', 'table'])

  // Build unified results list (filter out code, error, output artifacts)
  const allResults: ResultItem[] = [
    ...tables.map((t: ReturnType<typeof toTableInfo>) => ({
      type: 'table' as const,
      data: t,
      created_at: '', // Tables don't have created_at
      is_published: t.is_starred || false,
    })),
    ...artifacts
      .filter((a: ReturnType<typeof toArtifact>) => !excludedArtifactTypes.has(a.artifact_type))
      .map((a: ReturnType<typeof toArtifact>) => ({
        type: 'artifact' as const,
        data: a,
        created_at: a.created_at || '',
        is_published: a.is_starred || a.is_key_result || false,
      })),
  ]

  // Sort: inferences (negative step_number) descending first (-1, -2, …),
  // then regular steps ascending (1, 2, 3…). Within same step, by name.
  allResults.sort((a, b) => {
    const aStep = a.data.step_number || 0
    const bStep = b.data.step_number || 0
    const aIsInf = aStep < 0
    const bIsInf = bStep < 0
    if (aIsInf && bIsInf) return bStep - aStep // -1 before -2 before -3
    if (aIsInf !== bIsInf) return aIsInf ? -1 : 1 // inferences before steps
    if (aStep !== bStep) return aStep - bStep
    return a.data.name.localeCompare(b.data.name)
  })

  // Show only starred (published) results — intermediates live in Debug
  const displayedResults = allResults.filter((r) => r.is_published)

  // Group results by step_number
  const resultsByStep: { stepNumber: number; goal: string | undefined; items: ResultItem[] }[] = []
  const stepMap = new Map<number, ResultItem[]>()
  for (const r of displayedResults) {
    const sn = r.data.step_number || 0
    if (!stepMap.has(sn)) stepMap.set(sn, [])
    stepMap.get(sn)!.push(r)
  }
  // Build ordered groups, look up goal from stepCodes
  const stepGoalMap = new Map(stepCodes.map((s) => [s.step_number, s.goal]))
  for (const [stepNumber, items] of stepMap) {
    resultsByStep.push({ stepNumber, goal: stepGoalMap.get(stepNumber), items })
  }

  const totalCount = displayedResults.length

  // Auto-expand: find best item to expand
  const isResultsSectionExpanded = expandedArtifactSections.includes('results')
  let bestResultId: string | null = null

  if (isResultsSectionExpanded && displayedResults.length > 0) {
    // Prefer published items with priority keywords
    const hasPriorityKeyword = (name?: string, title?: string): boolean => {
      const text = `${name || ''} ${title || ''}`.toLowerCase()
      return ['final', 'recommended', 'answer', 'result', 'conclusion'].some(kw => text.includes(kw))
    }

    const withKeyword = displayedResults.find((r) => {
      const title = r.type === 'artifact' ? r.data.title : undefined
      return hasPriorityKeyword(r.data.name, title)
    })
    const best = withKeyword || displayedResults[0]
    bestResultId = best.type === 'table' ? `table-${best.data.name}` : `artifact-${best.data.id}`
  }

  if (totalCount === 0) return null

  const finalInsight = messages.find((m) => m.isFinalInsight && m.stepDurationMs != null)
  const totalElapsedMs = finalInsight?.stepDurationMs ?? null

  return (
    <>
      <button
        onClick={() => {
          const newVal = !resultsCollapsed
          setResultsCollapsed(newVal)
          localStorage.setItem('constat-results-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Results ({totalCount})
        </span>
        <div className="flex items-center gap-1.5">
          {totalElapsedMs != null && totalElapsedMs > 0 && (
            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400" title="Total elapsed time">
              {formatMs(totalElapsedMs)}
            </span>
          )}
          <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${resultsCollapsed ? '' : 'rotate-90'}`} />
        </div>
      </button>

      {!resultsCollapsed && (
      <div id="section-results" className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 bg-white dark:bg-gray-800">
        {displayedResults.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No results yet. Star a table in Debug to promote it here.
          </p>
        ) : (
          <div className="space-y-3">
            {resultsByStep.map(({ stepNumber, goal, items }) => {
              const isStepCollapsed = collapsedResultSteps.has(stepNumber)
              const tableCount = items.filter(i => i.type === 'table').length
              const artifactCount = items.filter(i => i.type === 'artifact').length
              const tooltipParts = [
                goal,
                `${tableCount} table(s), ${artifactCount} artifact(s)`,
              ].filter(Boolean).join('\n')
              return (
              <div key={`step-group-${stepNumber}`}>
                {stepNumber !== 0 && (
                  <button
                    onClick={() => toggleResultStep(stepNumber)}
                    className="flex items-center gap-1 text-[10px] font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-1 px-1 hover:text-gray-600 dark:hover:text-gray-300 transition-colors w-full text-left"
                    title={tooltipParts}
                  >
                    <ChevronDownIcon className={`w-3 h-3 transition-transform ${isStepCollapsed ? '-rotate-90' : ''}`} />
                    {stepNumber < 0 ? `Inference ${Math.abs(stepNumber)}` : `Step ${stepNumber}`}
                    <span className="text-gray-300 dark:text-gray-600 ml-auto normal-case">{items.length}</span>
                  </button>
                )}
                {(!isStepCollapsed || stepNumber === 0) && (
                <div className="space-y-2">
                  {items.map((result) =>
                    result.type === 'table' ? (
                      <div key={`table-${result.data.name}`} id={`table-${result.data.name}`}>
                        <TableAccordion
                          table={result.data}
                          initiallyOpen={bestResultId === `table-${result.data.name}`}
                        />
                      </div>
                    ) : (
                      <div key={`artifact-${result.data.id}`} id={`artifact-${result.data.id}`}>
                        <ArtifactItemAccordion
                          artifact={result.data}
                          initiallyOpen={bestResultId === `artifact-${result.data.id}`}
                        />
                      </div>
                    )
                  )}
                </div>
                )}
              </div>
              )
            })}
          </div>
        )}
      </div>
      )}
    </>
  )
}

export { StepCodeList }
