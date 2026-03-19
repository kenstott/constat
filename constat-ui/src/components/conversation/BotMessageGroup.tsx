// Bot message group — collapses thinking/plan/steps into a summary with expandable detail

import { useState, useMemo, useEffect } from 'react'
import { MessageBubble, StepDisplayMode } from './MessageBubble'
import {
  CheckCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  TableCellsIcon,
  DocumentTextIcon,
} from '@heroicons/react/24/outline'
import { VeraIcon } from './VeraIcon'
import { useArtifactStore } from '@/store/artifactStore'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import * as api from '@/api/client'

type StoreMessage = {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: Date
  stepNumber?: number
  isLive?: boolean
  isPending?: boolean
  defaultExpanded?: boolean
  isFinalInsight?: boolean
  role?: string
  skills?: string[]
  stepStartedAt?: number
  stepDurationMs?: number
  stepAttempts?: number
  isSuperseded?: boolean
}

interface BotMessageGroupProps {
  messages: StoreMessage[]
  stepOverride?: { mode: StepDisplayMode; version: number }
  stepOutputsMap: Map<number, Array<{ type: 'table' | 'artifact'; name: string; id: string }>>
  onOutputClick: (stepNumber: number | undefined, output: { type: 'table' | 'artifact'; name: string; id: string }) => void
  onRoleClick: (role: string) => void
  onStepEdit: (stepNumber: number, newGoal: string) => void
  onStepDelete: (stepNumber: number) => void
  openProofPanel: () => void
  allMessages: StoreMessage[]
}

export function BotMessageGroup({
  messages,
  stepOverride,
  stepOutputsMap,
  onOutputClick,
  onRoleClick,
  onStepEdit,
  onStepDelete,
  openProofPanel,
  allMessages,
}: BotMessageGroupProps) {
  const [expanded, setExpanded] = useState(false)

  // Separate steps/thinking from output messages
  const stepMessages = messages.filter((m) => m.type === 'step' || m.type === 'thinking' || m.type === 'plan' || m.type === 'system')
  const outputMessages = messages.filter((m) => m.type === 'output')

  // Check if any step is still in progress
  const isInProgress = messages.some((m) => m.isLive || m.isPending)
  const allComplete = stepMessages.length > 0 && stepMessages.every((m) => !m.isLive && !m.isPending)

  // Find most recent user query for flagging
  const queryText = useMemo(() => {
    const firstMsg = messages[0]
    const idx = allMessages.indexOf(firstMsg)
    for (let i = idx - 1; i >= 0; i--) {
      if (allMessages[i].type === 'user') return allMessages[i].content
    }
    return undefined
  }, [messages, allMessages])

  // Count completed steps
  const completedSteps = stepMessages.filter((m) => m.type === 'step' && m.stepDurationMs !== undefined).length
  const totalSteps = stepMessages.filter((m) => m.type === 'step').length

  const timestamp = messages[0]?.timestamp

  return (
    <div>
      {/* Avatar header */}
      <div className="flex items-center gap-3 mb-2">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-800 ${isInProgress ? 'animate-pulse' : ''}`}>
          <VeraIcon className="w-5 h-5" />
        </div>
        <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">Vera</span>
        {timestamp && !isInProgress && (
          <span className="text-xs text-gray-400 dark:text-gray-500">{timestamp.toLocaleTimeString()}</span>
        )}
      </div>

      {/* Steps: show bubbles during execution, collapsible summary when complete */}
      {stepMessages.length > 0 && (
        <div className="ml-11 mb-2">
          {isInProgress ? (
            // In-progress: show actual step bubbles
            <div className="space-y-2">
              {stepMessages.map((message) => (
                <MessageBubble
                  key={message.id}
                  type={message.type}
                  content={message.content}
                  timestamp={message.timestamp}
                  stepNumber={message.stepNumber}
                  isLive={message.isLive}
                  isPending={message.isPending}
                  defaultExpanded={message.defaultExpanded}
                  isFinalInsight={message.isFinalInsight}
                  role={message.role}
                  skills={message.skills}
                  stepStartedAt={message.stepStartedAt}
                  stepDurationMs={message.stepDurationMs}
                  stepAttempts={message.stepAttempts}
                  stepDisplayMode={message.type === 'step' ? stepOverride?.mode : undefined}
                  stepDisplayModeVersion={stepOverride?.version}
                  queryText={queryText}
                  isSuperseded={message.isSuperseded}
                  onStepEdit={onStepEdit}
                  onStepDelete={onStepDelete}
                  stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
                  onOutputClick={(output) => onOutputClick(message.stepNumber, output)}
                  onRoleClick={onRoleClick}
                  hideHeader
                />
              ))}
            </div>
          ) : allComplete ? (
            // Completed: collapsible summary
            <div>
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
              >
                <CheckCircleIcon className="w-4 h-4 text-green-500 flex-shrink-0" />
                <span>
                  {totalSteps > 0
                    ? `Completed ${completedSteps} step${completedSteps !== 1 ? 's' : ''}`
                    : 'Analyzed data and generated results'}
                </span>
                {expanded ? (
                  <ChevronUpIcon className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDownIcon className="w-3.5 h-3.5" />
                )}
              </button>
              {expanded && (
                <div className="mt-2 space-y-2 border-l-2 border-gray-200 dark:border-gray-700 pl-3 ml-1">
                  {stepMessages.map((message) => (
                    <MessageBubble
                      key={message.id}
                      type={message.type}
                      content={message.content}
                      timestamp={message.timestamp}
                      stepNumber={message.stepNumber}
                      isLive={message.isLive}
                      isPending={message.isPending}
                      defaultExpanded={message.defaultExpanded}
                      isFinalInsight={message.isFinalInsight}
                      role={message.role}
                      skills={message.skills}
                      stepStartedAt={message.stepStartedAt}
                      stepDurationMs={message.stepDurationMs}
                      stepAttempts={message.stepAttempts}
                      stepDisplayMode={message.type === 'step' ? stepOverride?.mode : undefined}
                      stepDisplayModeVersion={stepOverride?.version}
                      queryText={queryText}
                      isSuperseded={message.isSuperseded}
                      onStepEdit={onStepEdit}
                      onStepDelete={onStepDelete}
                      stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
                      onOutputClick={(output) => onOutputClick(message.stepNumber, output)}
                      onRoleClick={onRoleClick}
                      hideHeader
                    />
                  ))}
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* Output messages render as regular content below the step summary */}
      {outputMessages.map((message) => (
        <div key={message.id} className="ml-11">
          <MessageBubble
            hideHeader
            type={message.type}
            content={message.content}
            timestamp={message.timestamp}
            stepNumber={message.stepNumber}
            isLive={message.isLive}
            isPending={message.isPending}
            defaultExpanded={message.defaultExpanded}
            isFinalInsight={message.isFinalInsight}
            onViewResult={message.isFinalInsight && message.content?.toLowerCase().includes('proof')
              ? openProofPanel : undefined}
            role={message.role}
            skills={message.skills}
            stepStartedAt={message.stepStartedAt}
            stepDurationMs={message.stepDurationMs}
            stepAttempts={message.stepAttempts}
            queryText={queryText}
            isSuperseded={message.isSuperseded}
            stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
            onOutputClick={(output) => onOutputClick(message.stepNumber, output)}
            onRoleClick={onRoleClick}
          />
          {message.isFinalInsight && <InlineArtifacts />}
        </div>
      ))}
    </div>
  )
}

// Inline display of published artifacts (tables + non-table artifacts) within the response
function InlineArtifacts() {
  const { tables, artifacts } = useArtifactStore()
  const { showArtifactPanel, expandArtifactSection, expandResultStep } = useUIStore()
  const { session } = useSessionStore()

  // Filter to "published" artifacts — key results and starred tables
  const publishedTables = tables.filter((t) => t.is_starred)
  const internalTypes = new Set(['code', 'output', 'error', 'stdout', 'stderr', 'table'])
  const publishedArtifacts = artifacts.filter(
    (a) => a.step_number > 0 && !internalTypes.has(a.artifact_type) && (a.is_key_result || a.is_starred)
  )

  if (publishedTables.length === 0 && publishedArtifacts.length === 0) return null

  const handleTableClick = (tableName: string, stepNumber: number) => {
    showArtifactPanel()
    expandArtifactSection('results')
    if (stepNumber) expandResultStep(stepNumber)
    setTimeout(() => {
      const el = document.getElementById(`table-${tableName}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('ring-2', 'ring-primary-400')
        setTimeout(() => el.classList.remove('ring-2', 'ring-primary-400'), 2000)
      }
    }, 150)
  }

  const handleArtifactClick = (id: number, stepNumber: number) => {
    showArtifactPanel()
    expandArtifactSection('results')
    if (stepNumber) expandResultStep(stepNumber)
    setTimeout(() => {
      const el = document.getElementById(`artifact-${id}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('ring-2', 'ring-primary-400')
        setTimeout(() => el.classList.remove('ring-2', 'ring-primary-400'), 2000)
      }
    }, 150)
  }

  return (
    <div className="mt-3 space-y-2">
      {/* Inline tables */}
      {publishedTables.map((table) => (
        <InlineTablePreview
          key={table.name}
          table={table}
          sessionId={session?.session_id}
          onClick={() => handleTableClick(table.name, table.step_number)}
        />
      ))}
      {/* Non-table artifacts as banners */}
      {publishedArtifacts.map((artifact) => (
        <button
          key={artifact.id}
          onClick={() => handleArtifactClick(artifact.id, artifact.step_number)}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-left"
        >
          <DocumentTextIcon className="w-4 h-4 text-primary-500 flex-shrink-0" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100 flex-1 truncate">
            {artifact.title || artifact.name}
          </span>
          <span className="text-xs text-primary-600 dark:text-primary-400">View</span>
        </button>
      ))}
    </div>
  )
}

// Compact table preview — shows header + up to 5 rows
function InlineTablePreview({
  table,
  sessionId,
  onClick,
}: {
  table: { name: string; columns: string[]; row_count: number }
  sessionId?: string
  onClick: () => void
}) {
  const [rows, setRows] = useState<Record<string, unknown>[] | null>(null)

  useEffect(() => {
    if (!sessionId) return
    api.get<{ rows: Record<string, unknown>[] }>(`/sessions/${sessionId}/tables/${encodeURIComponent(table.name)}/preview?limit=5`)
      .then((data) => setRows(data.rows))
      .catch(() => {})
  }, [sessionId, table.name])

  return (
    <div
      className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden cursor-pointer hover:border-primary-300 dark:hover:border-primary-700 transition-colors"
      onClick={onClick}
    >
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800/50">
        <div className="flex items-center gap-2">
          <TableCellsIcon className="w-4 h-4 text-primary-500" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">{table.name}</span>
          <span className="text-xs text-gray-400">{table.row_count} rows</span>
        </div>
        <span className="text-xs text-primary-600 dark:text-primary-400">View</span>
      </div>
      {rows && rows.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                {table.columns.slice(0, 6).map((col) => (
                  <th key={col} className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap">
                    {col}
                  </th>
                ))}
                {table.columns.length > 6 && (
                  <th className="px-2 py-1 text-left font-medium text-gray-400">...</th>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-gray-100 dark:border-gray-800 last:border-0">
                  {table.columns.slice(0, 6).map((col) => (
                    <td key={col} className="px-2 py-1 text-gray-600 dark:text-gray-400 whitespace-nowrap max-w-[150px] truncate">
                      {String(row[col] ?? '')}
                    </td>
                  ))}
                  {table.columns.length > 6 && (
                    <td className="px-2 py-1 text-gray-400">...</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
