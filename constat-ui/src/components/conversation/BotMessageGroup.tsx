// Bot message group — collapses thinking/plan/steps into a summary with expandable detail

import { useState, useMemo, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
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
  stepSourcesRead?: string[]
  stepTablesCreated?: string[]
}

interface BotMessageGroupProps {
  messages: StoreMessage[]
  stepOverride?: { mode: StepDisplayMode; version: number }
  insightOverride?: { collapsed: boolean; version: number }
  groupOverride?: { expanded: boolean; version: number }
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
  insightOverride,
  groupOverride,
  stepOutputsMap,
  onOutputClick,
  onRoleClick,
  onStepEdit,
  onStepDelete,
  openProofPanel,
  allMessages,
}: BotMessageGroupProps) {
  const [expanded, setExpanded] = useState(false)

  // Sync group expanded state from conversation-level override
  useEffect(() => {
    if (groupOverride !== undefined) {
      setExpanded(groupOverride.expanded)
    }
  }, [groupOverride?.version])
  const { session, submitQuery } = useSessionStore()
  const { tables } = useArtifactStore()
  // Show all domains except synthetic root/user nodes (constants, not useful)
  const activeDomains = (session?.active_domains || []).filter(d => d !== 'root' && d !== 'user')

  // Format filename → display name (e.g., "sales-analytics" → "Sales Analytics")
  const domainDisplayName = (filename: string): string =>
    filename.replace(/\.ya?ml$/, '').split(/[-_]/).map(w => w[0].toUpperCase() + w.slice(1)).join(' ')

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

  // Detect clarification message in group
  const clarificationMsg = stepMessages.find((m) => m.type === 'system' && m.content?.startsWith('Please clarify'))

  // Standalone output group (e.g., /reason result) — collapsible when complete
  const isStandaloneOutput = stepMessages.length === 0 && outputMessages.length > 0 && !isInProgress
  const outputSummary = isStandaloneOutput
    ? (outputMessages[0].content.split(/[.\n]/)[0] || 'Complete')
    : ''

  const timestamp = messages[0]?.timestamp

  return (
    <div>
      {/* Avatar header */}
      <div className="flex items-center gap-3 mb-2">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-800 ${isInProgress ? 'animate-pulse' : ''}`}>
          <VeraIcon className="w-5 h-5" />
        </div>
        <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">Vera</span>
        {timestamp && !isInProgress && (
          <span className="text-xs text-gray-400 dark:text-gray-500">{timestamp.toLocaleTimeString()}</span>
        )}
        {activeDomains.length > 0 && (
          <div className="flex flex-wrap gap-1 ml-2">
            {activeDomains.map(d => (
              <span key={d} className="inline-flex items-center text-[10px] px-1.5 py-0.5 rounded-full bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400 font-medium">
                {domainDisplayName(d)}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Steps: show bubbles during execution, collapsible summary when complete */}
      {stepMessages.length > 0 && (
        <div className="ml-11 mb-2">
          {isInProgress ? (
            // In-progress: show actual step bubbles
            <div className="space-y-2">
              {stepMessages.map((message) => (
                <div key={message.id}>
                  <MessageBubble
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
                    contentExpanded={groupOverride?.expanded}
                    contentExpandedVersion={groupOverride?.version}
                    queryText={queryText}
                    isSuperseded={message.isSuperseded}
                    onStepEdit={onStepEdit}
                    onStepDelete={onStepDelete}
                    stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
                    onOutputClick={(output) => onOutputClick(message.stepNumber, output)}
                    onRoleClick={onRoleClick}
                    stepSourcesRead={message.stepSourcesRead}
                    stepTablesCreated={message.stepTablesCreated}
                    hideHeader
                  />
                  <StepTablePreview
                    message={message}
                    tables={tables}
                    sessionId={session?.session_id}
                    onOutputClick={onOutputClick}
                  />
                </div>
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
                <span className="text-left">
                  {clarificationMsg
                    ? 'Please clarify...'
                    : totalSteps > 0
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
                    <div key={message.id}>
                      <MessageBubble
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
                        contentExpanded={groupOverride?.expanded}
                        contentExpandedVersion={groupOverride?.version}
                        queryText={queryText}
                        isSuperseded={message.isSuperseded}
                        onStepEdit={onStepEdit}
                        onStepDelete={onStepDelete}
                        stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
                        onOutputClick={(output) => onOutputClick(message.stepNumber, output)}
                        onRoleClick={onRoleClick}
                        stepSourcesRead={message.stepSourcesRead}
                        stepTablesCreated={message.stepTablesCreated}
                        hideHeader
                      />
                      <StepTablePreview
                        message={message}
                        tables={tables}
                        sessionId={session?.session_id}
                        onOutputClick={onOutputClick}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* Output messages render as regular content below the step summary */}
      {isStandaloneOutput && (
        <div className="ml-11 mb-2">
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
          >
            <CheckCircleIcon className="w-4 h-4 text-green-500 flex-shrink-0" />
            <span className="text-left">{outputSummary}</span>
            {expanded ? (
              <ChevronUpIcon className="w-3.5 h-3.5" />
            ) : (
              <ChevronDownIcon className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      )}
      {(!isStandaloneOutput || expanded) && outputMessages.map((message) => {
        // Extract "Next Steps" items from final insight to render as pills
        const nextStepsMatch = message.isFinalInsight
          ? message.content.match(/\n+\*{0,2}Next Steps[:\s]*\*{0,2}[:\s]*\n([\s\S]*)$/i)
          : null
        const extractedFollowUps = nextStepsMatch
          ? nextStepsMatch[1].split(/\n?\d+\.\s+/).filter(Boolean).map(s => s.trim())
          : []
        const displayContent = extractedFollowUps.length > 0
          ? message.content.replace(/\n+\*{0,2}Next Steps[:\s]*\*{0,2}[:\s]*\n[\s\S]*$/i, '').trimEnd()
          : message.content
        // Only show pills on the very last final insight
        const isLastInsight = message.isFinalInsight &&
          message.id === [...allMessages].reverse().find(m => m.isFinalInsight)?.id

        return (
          <div key={message.id} className="ml-11">
            <MessageBubble
              hideHeader
              type={message.type}
              content={displayContent}
              timestamp={message.timestamp}
              stepNumber={message.stepNumber}
              isLive={message.isLive}
              isPending={message.isPending}
              defaultExpanded={message.defaultExpanded}
              isFinalInsight={message.isFinalInsight}
              insightCollapsed={insightOverride?.collapsed}
              insightCollapsedVersion={insightOverride?.version}
              contentExpanded={groupOverride?.expanded}
              contentExpandedVersion={groupOverride?.version}
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
            >
              {message.isFinalInsight && <InlineArtifacts />}
            </MessageBubble>
            {/* Follow-up suggestions as clickable pills below the final insight */}
            {isLastInsight && extractedFollowUps.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-3">
                {extractedFollowUps.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => submitQuery(s, true)}
                    className="px-3 py-1.5 text-sm rounded-full border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-200 transition-colors text-left"
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

// Inline display of published artifacts (tables + non-table artifacts) within the response
function InlineArtifacts() {
  const { tables, artifacts } = useArtifactStore()
  const { openFullscreenArtifact } = useUIStore()
  const { session } = useSessionStore()

  // Filter to "published" artifacts — key results and starred tables
  const publishedTables = tables.filter((t) => t.is_starred)
  const internalTypes = new Set(['code', 'output', 'error', 'stdout', 'stderr', 'table'])
  const mdTypes = new Set(['markdown', 'md'])
  const publishedArtifacts = artifacts.filter(
    (a) => a.step_number > 0 && !internalTypes.has(a.artifact_type) && (a.is_key_result || a.is_starred)
  )

  if (publishedTables.length === 0 && publishedArtifacts.length === 0) return null

  const handleTableClick = (tableName: string) => {
    openFullscreenArtifact({ type: 'table', name: tableName })
  }

  const handleArtifactClick = (id: number) => {
    openFullscreenArtifact({ type: 'artifact', id })
  }

  return (
    <div className="mt-3 space-y-2">
      {/* Inline tables */}
      {publishedTables.map((table) => (
        <InlineTablePreview
          key={table.name}
          table={table}
          sessionId={session?.session_id}
          onClick={() => handleTableClick(table.name)}
        />
      ))}
      {/* Markdown artifacts rendered inline; others as banners */}
      {publishedArtifacts.map((artifact) =>
        mdTypes.has(artifact.artifact_type?.toLowerCase()) ? (
          <InlineMarkdownArtifact
            key={artifact.id}
            artifact={artifact}
            sessionId={session?.session_id}
            onClick={() => handleArtifactClick(artifact.id)}
          />
        ) : (
          <button
            key={artifact.id}
            onClick={() => handleArtifactClick(artifact.id)}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors text-left"
          >
            <DocumentTextIcon className="w-4 h-4 text-primary-500 flex-shrink-0" />
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100 flex-1 truncate">
              {artifact.title || artifact.name}
            </span>
            <span className="text-xs text-primary-600 dark:text-primary-400">View</span>
          </button>
        )
      )}
    </div>
  )
}

// Inline rendered markdown artifact content
function InlineMarkdownArtifact({
  artifact,
  sessionId,
  onClick,
}: {
  artifact: { id: number; name: string; title?: string }
  sessionId?: string
  onClick: () => void
}) {
  const [content, setContent] = useState<string | null>(null)

  useEffect(() => {
    if (!sessionId) return
    api.get<{ content: string }>(`/sessions/${sessionId}/artifacts/${artifact.id}`)
      .then((data) => setContent(data.content))
      .catch(() => {})
  }, [sessionId, artifact.id])

  return (
    <div
      className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden cursor-pointer hover:border-primary-300 dark:hover:border-primary-700 transition-colors"
      onClick={onClick}
    >
      <div className="flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800/50">
        <div className="flex items-center gap-2">
          <DocumentTextIcon className="w-4 h-4 text-primary-500" />
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {artifact.title || artifact.name}
          </span>
        </div>
        <span className="text-xs text-primary-600 dark:text-primary-400">Expand</span>
      </div>
      {content && (
        <div className="px-3 py-2 text-sm text-gray-900 dark:text-gray-100 prose prose-sm dark:prose-invert max-w-none max-h-[300px] overflow-y-auto">
          <MarkdownContent content={content} />
        </div>
      )}
    </div>
  )
}

// Per-step inline table preview — renders when a step created exactly 1 table
function StepTablePreview({
  message,
  tables,
  sessionId,
  onOutputClick,
}: {
  message: StoreMessage
  tables: Array<{ name: string; columns: string[]; row_count: number }>
  sessionId?: string
  onOutputClick: (stepNumber: number | undefined, output: { type: 'table' | 'artifact'; name: string; id: string }) => void
}) {
  if (message.stepTablesCreated?.length !== 1 || message.stepDurationMs === undefined) return null
  const tableName = message.stepTablesCreated[0]
  const t = tables.find(t => t.name === tableName)
  if (!t) return null
  return (
    <InlineTablePreview
      table={t}
      sessionId={sessionId}
      onClick={() => onOutputClick(message.stepNumber, {
        type: 'table', name: tableName, id: `table-${tableName}`
      })}
    />
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
  const [cols, setCols] = useState<string[]>(table.columns)

  useEffect(() => {
    if (!sessionId) return
    api.get<{ data: Record<string, unknown>[]; columns: string[] }>(`/sessions/${sessionId}/tables/${encodeURIComponent(table.name)}?page=1&page_size=5`)
      .then((resp) => {
        setRows(resp.data)
        if (resp.columns?.length) setCols(resp.columns)
      })
      .catch(() => {})
  }, [sessionId, table.name])

  // Use API columns, fall back to prop columns, fall back to keys from first row
  const displayCols = cols.length > 0 ? cols : (rows?.[0] ? Object.keys(rows[0]) : [])

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
                {displayCols.slice(0, 6).map((col) => (
                  <th key={col} className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap">
                    {col}
                  </th>
                ))}
                {displayCols.length > 6 && (
                  <th className="px-2 py-1 text-left font-medium text-gray-400">...</th>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={i} className="border-b border-gray-100 dark:border-gray-800 last:border-0">
                  {displayCols.slice(0, 6).map((col) => (
                    <td key={col} className="px-2 py-1 text-gray-600 dark:text-gray-400 whitespace-nowrap max-w-[150px] truncate">
                      {String(row[col] ?? '')}
                    </td>
                  ))}
                  {displayCols.length > 6 && (
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

// Simple markdown renderer for inline artifact content
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2">{children}</ol>,
        li: ({ children }) => <li className="mb-1">{children}</li>,
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full text-xs border-collapse">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-gray-100 dark:bg-gray-700">{children}</thead>
        ),
        tr: ({ children }) => (
          <tr className="border-b border-gray-200 dark:border-gray-600">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-2 py-1 text-left font-medium text-gray-700 dark:text-gray-300">{children}</th>
        ),
        td: ({ children }) => (
          <td className="px-2 py-1 text-gray-600 dark:text-gray-400">{children}</td>
        ),
        code: ({ className, children }) => {
          const isInline = !className
          return isInline ? (
            <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs">{children}</code>
          ) : (
            <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-xs overflow-x-auto my-2">
              <code>{children}</code>
            </pre>
          )
        },
      }}
    >
      {content}
    </ReactMarkdown>
  )
}
