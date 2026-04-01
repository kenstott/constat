// Copyright (c) 2025 Kenneth Stott
// Canary: 76e18916-ff3f-474a-b235-f4c59b4a0f0f
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState, useEffect, useCallback } from 'react'
import {
  ChevronRightIcon,
  ChevronDownIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon,
  CodeBracketIcon,
  CircleStackIcon,
  DocumentTextIcon,
  StarIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline'
import { AccordionSection } from '../ArtifactAccordion'
import { CodeViewer } from '../CodeViewer'
import { useTables } from '@/hooks/useTables'
import { getAuthHeaders } from '@/config/auth-helpers'

interface StepCode {
  step_number: number
  goal: string
  code: string
  prompt?: string
  model?: string
}

interface InferenceCode {
  inference_id: string
  name: string
  operation: string
  code: string
  attempt: number
  prompt?: string
  model?: string
}

interface ScratchpadEntry {
  step_number: number
  goal: string
  narrative: string
  tables_created: string[]
  code: string
  user_query: string
  objective_index: number | null
}

interface CodeLogSectionProps {
  stepCodes: StepCode[]
  inferenceCodes: InferenceCode[]
  scratchpadEntries: ScratchpadEntry[]
  sessionDDL: string
  supersededStepNumbers: Set<number>
  onRefreshDDL: () => void
  session: { session_id: string } | null
  canSeeSection: (section: string) => boolean
  artifacts: Array<{ id: number; name: string; step_number: number; artifact_type: string }>
}

/** Collapsible scratchpad list for the Scratchpad section */
function ScratchpadList({ entries, supersededStepNumbers, tables }: {
  entries: Array<{ step_number: number; goal: string; narrative: string; tables_created: string[]; code: string; user_query: string; objective_index: number | null }>
  supersededStepNumbers: Set<number>
  tables: Array<{ name: string }>
}) {
  const [collapsedSteps, setCollapsedSteps] = useState<Set<number>>(() => new Set(entries.map((e) => e.step_number)))

  // Collapse newly arriving entries by default
  useEffect(() => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      for (const e of entries) {
        next.add(e.step_number)
      }
      return next.size === prev.size ? prev : next
    })
  }, [entries])

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

  // Build set of existing table names for linking
  const tableNames = new Set(tables.map((t) => t.name))

  return (
    <div className="space-y-1">
      {entries.map((entry) => {
        const isCollapsed = collapsedSteps.has(entry.step_number)
        return (
          <div key={entry.step_number} className={supersededStepNumbers.has(entry.step_number) ? 'opacity-40' : ''}>
            <button
              onClick={() => toggleStep(entry.step_number)}
              className="flex items-center gap-1 w-full text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
            >
              {isCollapsed ? (
                <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
              ) : (
                <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
              )}
              <span className="font-medium">Step {entry.step_number}:</span>
              <span>{entry.goal}</span>
            </button>
            {!isCollapsed && (
              <div className="pl-5 pb-2 space-y-1.5">
                <p className="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {entry.narrative}
                </p>
                {entry.tables_created.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {entry.tables_created.map((tbl) => (
                      <button
                        key={tbl}
                        onClick={() => {
                          if (!tableNames.has(tbl)) return
                          const el = document.getElementById('section-results')
                          el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                          requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                              const item = document.getElementById(`table-${tbl}`)
                              if (item) {
                                item.scrollIntoView({ behavior: 'smooth', block: 'center' })
                                item.classList.add('ring-2', 'ring-primary-400')
                                setTimeout(() => item.classList.remove('ring-2', 'ring-primary-400'), 2000)
                              }
                            })
                          })
                        }}
                        className={`inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                          tableNames.has(tbl)
                            ? 'bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-900/20 dark:text-primary-400 dark:hover:bg-primary-900/40 cursor-pointer'
                            : 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-500 cursor-default'
                        }`}
                        title={tableNames.has(tbl) ? `Jump to ${tbl}` : tbl}
                      >
                        <CircleStackIcon className="w-3 h-3" />
                        {tbl}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

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

export function CodeLogSection({
  stepCodes,
  inferenceCodes,
  scratchpadEntries,
  sessionDDL,
  supersededStepNumbers,
  onRefreshDDL,
  session,
  canSeeSection,
  artifacts,
}: CodeLogSectionProps) {
  const { tables } = useTables()
  const [codeLogCollapsed, setCodeLogCollapsed] = useState(() => localStorage.getItem('constat-codelog-collapsed') === 'true')
  const [collapsedInferences, setCollapsedInferences] = useState<Set<string>>(new Set<string>())
  const [showInferencePrompt, setShowInferencePrompt] = useState<Set<string>>(new Set())

  // Default inference codes to collapsed when they load
  useEffect(() => {
    if (inferenceCodes.length > 0) {
      setCollapsedInferences(new Set(inferenceCodes.map((inf) => inf.inference_id)))
    }
  }, [inferenceCodes])

  const codeLogVisible = canSeeSection('code') || canSeeSection('inference_code') || scratchpadEntries.length > 0

  return (
    <>
      {/* --- Code Log sub-group --- */}
      {codeLogVisible && (
      <button
        onClick={() => {
          const newVal = !codeLogCollapsed
          setCodeLogCollapsed(newVal)
          localStorage.setItem('constat-codelog-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Code Log
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${codeLogCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {codeLogVisible && !codeLogCollapsed && (
      <>

      {/* Scratchpad - execution narrative per step */}
      {scratchpadEntries.length > 0 && (
        <AccordionSection
          id="scratchpad"
          title="Scratchpad"
          count={scratchpadEntries.length}
          icon={<DocumentTextIcon className="w-4 h-4" />}
        >
          <ScratchpadList entries={scratchpadEntries} supersededStepNumbers={supersededStepNumbers} tables={tables} />
        </AccordionSection>
      )}

      {/* Code - only show when there's code and user can see it */}
      {canSeeSection('code') && stepCodes.length > 0 && (
        <AccordionSection
          id="code"
          title="Exploratory Code"
          count={stepCodes.length}
          icon={<CodeBracketIcon className="w-4 h-4" />}
          command="/code"
          action={
            <button
              onClick={async () => {
                if (!session) return
                try {
                  const headers = await getAuthHeaders()

                  const response = await fetch(
                    `/api/sessions/${session.session_id}/download-code`,
                    { headers, credentials: 'include' }
                  )
                  if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}))
                    const message = errorData.detail || 'Failed to download code'
                    alert(message)
                    return
                  }
                  const blob = await response.blob()
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `session_${session.session_id.slice(0, 8)}_code.py`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                } catch (err) {
                  console.error('Download failed:', err)
                  alert('Failed to download code. Please try again.')
                }
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Download as Python script"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
            </button>
          }
        >
          <StepCodeList stepCodes={stepCodes} supersededStepNumbers={supersededStepNumbers} tables={tables} artifacts={artifacts} />
        </AccordionSection>
      )}

      {/* Inference Code - auditable mode (separate from step code) */}
      {canSeeSection('inference_code') && inferenceCodes.length > 0 && (
        <AccordionSection
          id="inference-code"
          title="Inference Code"
          count={inferenceCodes.length}
          icon={<CodeBracketIcon className="w-4 h-4" />}
          action={
            <button
              onClick={async () => {
                if (!session) return
                try {
                  const headers = await getAuthHeaders()
                  const response = await fetch(
                    `/api/sessions/${session.session_id}/download-inference-code`,
                    { headers, credentials: 'include' }
                  )
                  if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}))
                    alert(errorData.detail || 'Failed to download inference code')
                    return
                  }
                  const blob = await response.blob()
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `session_${session.session_id.slice(0, 8)}_inference.py`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                } catch (err) {
                  console.error('Download failed:', err)
                  alert('Failed to download inference code. Please try again.')
                }
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Download as Python script"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
            </button>
          }
        >
          <div className="space-y-1">
            {inferenceCodes.map((inf) => {
              const isCollapsed = collapsedInferences.has(inf.inference_id)
              return (
                <div key={inf.inference_id}>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setCollapsedInferences((prev) => {
                        const next = new Set(prev)
                        if (next.has(inf.inference_id)) next.delete(inf.inference_id)
                        else next.add(inf.inference_id)
                        return next
                      })}
                      className="flex items-start gap-1 flex-1 text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
                    >
                      {isCollapsed ? (
                        <ChevronRightIcon className="w-3 h-3 flex-shrink-0 mt-0.5" />
                      ) : (
                        <ChevronDownIcon className="w-3 h-3 flex-shrink-0 mt-0.5" />
                      )}
                      <span className="flex flex-col">
                        <span>{inf.inference_id}: {inf.name} = {inf.operation}</span>
                        {inf.model && (
                          <span className="text-[10px] text-gray-400 dark:text-gray-500 font-mono">{inf.model}</span>
                        )}
                      </span>
                    </button>
                    {inf.prompt && (
                      <button
                        onClick={() => setShowInferencePrompt((prev) => {
                          const next = new Set(prev)
                          if (next.has(inf.inference_id)) next.delete(inf.inference_id)
                          else next.add(inf.inference_id)
                          return next
                        })}
                        className="text-[10px] text-gray-400 hover:text-primary-500 ml-auto px-1"
                      >
                        {showInferencePrompt.has(inf.inference_id) ? 'Hide Prompt' : 'Prompt'}
                      </button>
                    )}
                  </div>
                  {!isCollapsed && (
                    <>
                      {showInferencePrompt.has(inf.inference_id) && inf.prompt && (
                        <pre className="text-[10px] leading-tight text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded overflow-auto max-h-60 mb-1 whitespace-pre-wrap">
                          {inf.prompt}
                        </pre>
                      )}
                      <CodeViewer
                        code={inf.code}
                        language="python"
                      />
                    </>
                  )}
                </div>
              )
            })}
          </div>
        </AccordionSection>
      )}

      {/* Session Store DDL */}
      {sessionDDL && (
        <AccordionSection
          id="session-ddl"
          title="Session Store"
          icon={<CircleStackIcon className="w-4 h-4" />}
          action={
            <button
              onClick={onRefreshDDL}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Refresh DDL"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>
          }
        >
          <pre className="text-[11px] leading-relaxed text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 p-3 rounded overflow-auto max-h-96 whitespace-pre font-mono">
            {sessionDDL}
          </pre>
        </AccordionSection>
      )}

      </>
      )}
    </>
  )
}
