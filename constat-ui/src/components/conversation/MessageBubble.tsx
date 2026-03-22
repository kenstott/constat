// Message Bubble component

import { ReactNode, useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  EyeIcon,
  LightBulbIcon,
  BoltIcon,
  CheckIcon,
  PencilIcon,
  TrashIcon,
  CircleStackIcon,
  StarIcon,
} from '@heroicons/react/24/outline'
import { FlagButton } from './FlagButton'
import { VeraIcon } from './VeraIcon'
import { useAuthStore } from '@/store/authStore'

// Shared markdown renderer used by MessageBubble content sections
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ children }) => <p className="whitespace-pre-wrap mb-2 last:mb-0">{children}</p>,
        strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
        ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
        ol: ({ children }) => <ol className="list-decimal list-inside mb-2">{children}</ol>,
        li: ({ children }) => <li className="mb-1">{children}</li>,
        code: ({ className, children }) => {
          const match = /language-(\w+)/.exec(className || '')
          const isInline = !match
          return isInline ? (
            <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs">{children}</code>
          ) : (
            <CodeBlockWithCopy language={match[1]}>
              {String(children).replace(/\n$/, '')}
            </CodeBlockWithCopy>
          )
        },
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full text-xs border-collapse">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-gray-100 dark:bg-gray-700">{children}</thead>
        ),
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => (
          <tr className="border-b border-gray-200 dark:border-gray-600">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-2 py-1 text-left font-medium text-gray-700 dark:text-gray-300">{children}</th>
        ),
        td: ({ children }) => (
          <td className="px-2 py-1 text-gray-600 dark:text-gray-400">{children}</td>
        ),
      }}
    >
      {content}
    </ReactMarkdown>
  )
}

// Animated dots component for loading states
function AnimatedDots() {
  return (
    <span className="inline-flex">
      <span className="animate-dot" style={{ animationDelay: '0ms' }}>.</span>
      <span className="animate-dot" style={{ animationDelay: '200ms' }}>.</span>
      <span className="animate-dot" style={{ animationDelay: '400ms' }}>.</span>
    </span>
  )
}

// Code block with copy-on-hover button
function CodeBlockWithCopy({ language, children }: { language: string; children: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(children)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="relative group/code">
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1 rounded bg-gray-700/80 hover:bg-gray-600 text-gray-300 hover:text-white opacity-0 group-hover/code:opacity-100 transition-opacity z-10"
        title="Copy to clipboard"
      >
        {copied ? (
          <ClipboardDocumentCheckIcon className="w-4 h-4 text-green-400" />
        ) : (
          <ClipboardDocumentIcon className="w-4 h-4" />
        )}
      </button>
      <SyntaxHighlighter
        style={oneDark as { [key: string]: React.CSSProperties }}
        language={language}
        PreTag="div"
        customStyle={{
          margin: '0.5rem 0',
          padding: '0.75rem',
          borderRadius: '0.375rem',
          fontSize: '0.75rem',
        }}
      >
        {children}
      </SyntaxHighlighter>
    </div>
  )
}

type MessageType = 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
export type StepDisplayMode = 'oneline' | 'condensed' | 'full'

interface MessageBubbleProps {
  type: MessageType
  content: string
  timestamp?: Date
  stepNumber?: number
  isLive?: boolean
  isPending?: boolean
  defaultExpanded?: boolean
  isFinalInsight?: boolean
  onViewResult?: () => void
  children?: ReactNode
  role?: string // Role used for this step (e.g., "data_analyst")
  skills?: string[] // Skills used for this step
  stepStartedAt?: number
  stepDurationMs?: number
  stepAttempts?: number
  stepDisplayMode?: StepDisplayMode // External override for condense-all / expand-all
  stepDisplayModeVersion?: number // Increment to re-trigger override
  insightCollapsed?: boolean // External override for collapse-all / expand-all on final insights
  insightCollapsedVersion?: number // Increment to re-trigger override
  queryText?: string // The user query that produced this answer (for flagging)
  isSuperseded?: boolean // Step from a previous run (dimmed)
  onStepEdit?: (stepNumber: number, newGoal: string) => void
  onStepDelete?: (stepNumber: number) => void
  stepOutputs?: Array<{ type: 'table' | 'artifact'; name: string; id: string }>
  onOutputClick?: (output: { type: 'table' | 'artifact'; name: string; id: string }) => void
  stepSourcesRead?: string[]
  stepTablesCreated?: string[]
  onRoleClick?: (role: string) => void
  onEditMessage?: (content: string) => void
  hideHeader?: boolean // Suppress avatar+name header (used inside BotMessageGroup)
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const remainSec = Math.round(seconds % 60)
  return `${minutes}m ${remainSec}s`
}

function getUserInitials(): string {
  const user = useAuthStore.getState().user
  if (user?.displayName) {
    return user.displayName.split(' ').map(p => p[0]).join('').toUpperCase().slice(0, 2)
  }
  if (user?.email) {
    return user.email[0].toUpperCase()
  }
  return 'U'
}

// Max height for collapsed content (approximately 5 lines at 1.25rem line height + padding)
const MAX_COLLAPSED_HEIGHT = 120 // ~5 lines

export function MessageBubble({
  type,
  content,
  timestamp,
  stepNumber,
  isLive,
  isPending,
  defaultExpanded,
  isFinalInsight,
  onViewResult,
  children,
  role,
  skills,
  stepStartedAt,
  stepDurationMs,
  stepAttempts,
  stepDisplayMode: externalStepMode,
  stepDisplayModeVersion: externalStepModeVersion,
  insightCollapsed: externalInsightCollapsed,
  insightCollapsedVersion: externalInsightCollapsedVersion,
  queryText,
  isSuperseded,
  onStepEdit,
  onStepDelete,
  stepOutputs,
  onOutputClick,
  onRoleClick,
  stepSourcesRead,
  stepTablesCreated: _stepTablesCreated,
  onEditMessage,
  hideHeader,
}: MessageBubbleProps) {
  const isUser = type === 'user'

  // Elapsed timer for running steps
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    if (!stepStartedAt || stepDurationMs !== undefined) return
    setElapsed(Date.now() - stepStartedAt)
    const id = setInterval(() => setElapsed(Date.now() - stepStartedAt), 1000)
    return () => clearInterval(id)
  }, [stepStartedAt, stepDurationMs])

  // Strip "Step X:" or "Step X" prefix from content if stepNumber is shown in header
  const cleanedContent = stepNumber !== undefined
    ? content.replace(/^Step\s+\d+:?\s*/i, '')
    : content

  // Step 3-mode display state
  const isStep = type === 'step'
  const getInitialStepMode = (): StepDisplayMode => {
    if (defaultExpanded) return 'full'
    return 'condensed'
  }
  const [stepMode, setStepMode] = useState<StepDisplayMode>(getInitialStepMode)

  // Non-step expand/collapse state (unchanged)
  const [isExpanded, setIsExpanded] = useState(defaultExpanded ?? isFinalInsight ?? false)
  const [needsExpansion, setNeedsExpansion] = useState(false)
  const [copied, setCopied] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editGoal, setEditGoal] = useState('')
  const contentRef = useRef<HTMLDivElement>(null)

  // Sync external step mode override (condense-all / expand-all)
  useEffect(() => {
    if (isStep && externalStepMode !== undefined) {
      setStepMode(externalStepMode)
    }
  }, [externalStepMode, externalStepModeVersion])

  // Sync external insight collapsed override
  useEffect(() => {
    if (externalInsightCollapsed !== undefined) {
      setIsExpanded(!externalInsightCollapsed)
    }
  }, [externalInsightCollapsed, externalInsightCollapsedVersion])

  // Auto-collapse step to condensed when it completes
  useEffect(() => {
    if (isStep && !isLive && !isPending && stepMode === 'full') {
      setStepMode('condensed')
    }
  }, [isLive])

  // Copy message content to clipboard
  const handleCopy = async () => {
    await navigator.clipboard.writeText(content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Check if content exceeds max height
  useEffect(() => {
    if (contentRef.current) {
      const scrollHeight = contentRef.current.scrollHeight
      setNeedsExpansion(scrollHeight > MAX_COLLAPSED_HEIGHT)
    }
  }, [content, stepMode])

  // Check if content ends with "..." to show animated dots
  const showAnimatedDots = (isLive || isPending) && cleanedContent.endsWith('...')
  const displayContent = showAnimatedDots ? cleanedContent.slice(0, -3) : cleanedContent

  return (
    <div className={`group ${isSuperseded ? 'opacity-40' : ''}`}>
      {/* Message header — avatar + name + timestamp */}
      {!hideHeader && (
        <div className="flex items-center gap-3 mb-1">
          {isUser ? (
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-primary-600 text-white text-xs font-semibold ${isPending ? 'opacity-50' : ''}`}>
              {getUserInitials()}
            </div>
          ) : (
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gray-800 dark:bg-gray-200 text-white dark:text-gray-800 ${isLive ? 'animate-pulse' : ''} ${isPending ? 'opacity-50' : ''}`}>
              <VeraIcon className="w-5 h-5" />
            </div>
          )}
          <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">
            {isUser ? (useAuthStore.getState().user?.displayName || 'You') : 'Vera'}
          </span>
          {timestamp && !isLive && (
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {timestamp.toLocaleTimeString()}
            </span>
          )}
        </div>
      )}

      {/* Content area — indented under the avatar */}
      <div className={hideHeader ? '' : 'ml-11'}>
        <div className="flex items-start gap-1">
          <div
            className={`relative flex-1 min-w-0 ${
              isLive ? 'border-l-2 border-blue-500 pl-3' : ''
            } ${isPending ? 'border-l-2 border-gray-300 dark:border-gray-600 pl-3 opacity-60' : ''}`}
          >
          {stepNumber !== undefined && (
            <div className={`flex items-center gap-2 ${stepMode === 'oneline' ? '' : 'mb-1'}`}>
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                Step {stepNumber}
              </span>
              {isPending && isLive ? (
                <LightBulbIcon className="w-3.5 h-3.5 text-amber-500 animate-pulse" />
              ) : isLive ? (
                <BoltIcon className="w-3.5 h-3.5 text-blue-500 animate-pulse" />
              ) : stepDurationMs !== undefined ? (
                <CheckIcon className="w-3.5 h-3.5 text-green-500" />
              ) : null}
              {role && (() => {
                const parts = role.split('/')
                const agentName = parts.length > 1 ? parts[parts.length - 1] : role
                const domainPrefix = parts.length > 1 ? parts.slice(0, -1).join('/') : null
                return (
                  <button
                    onClick={() => onRoleClick?.(agentName)}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300 hover:bg-purple-200 dark:hover:bg-purple-800/50 cursor-pointer transition-colors"
                  >
                    @{domainPrefix && <span className="opacity-60">{domainPrefix}/</span>}{agentName}
                  </button>
                )
              })()}
              {skills && skills.length > 0 && skills.map((skill) => (
                <span
                  key={skill}
                  className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-300"
                >
                  {skill}
                </span>
              ))}
              {stepStartedAt && (
                <span className="text-[10px] text-gray-400 dark:text-gray-500 ml-auto">
                  {stepDurationMs !== undefined
                    ? formatMs(stepDurationMs)
                    : formatMs(elapsed)}
                  {(stepAttempts ?? 0) > 0 && (
                    <span className="ml-1 text-amber-500">{stepAttempts} {stepAttempts === 1 ? 'retry' : 'retries'}</span>
                  )}
                </span>
              )}
              {isStep && (
                <span className={`inline-flex items-center ${!stepStartedAt ? 'ml-auto' : ''}`}>
                  {(stepMode === 'condensed' || stepMode === 'full') && (
                    <button
                      onClick={() => setStepMode(stepMode === 'condensed' ? 'oneline' : 'condensed')}
                      className="p-0.5 rounded text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
                      title="Show less"
                    >
                      <ChevronUpIcon className="w-3.5 h-3.5" />
                    </button>
                  )}
                  {(stepMode === 'oneline' || (stepMode === 'condensed' && needsExpansion)) && (
                    <button
                      onClick={() => setStepMode(stepMode === 'oneline' ? 'condensed' : 'full')}
                      className="p-0.5 rounded text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
                      title="Show more"
                    >
                      <ChevronDownIcon className="w-3.5 h-3.5" />
                    </button>
                  )}
                </span>
              )}
            </div>
          )}
          {!(isStep && stepMode === 'oneline') && (
          <div
            ref={contentRef}
            className={`text-sm text-gray-900 dark:text-gray-100 ${
              isStep
                ? stepMode === 'condensed' ? 'overflow-y-auto' : ''
                : !isUser && !isExpanded && needsExpansion ? 'overflow-y-auto pr-[5px]' : ''
            }`}
            style={{
              maxHeight: isStep
                ? stepMode === 'condensed' ? `${MAX_COLLAPSED_HEIGHT}px` : undefined
                : !isUser && !isExpanded && needsExpansion ? `${MAX_COLLAPSED_HEIGHT}px` : undefined,
            }}
          >
            {type === 'thinking' ? (
              <span>{cleanedContent.replace(/\.+$/, '') || 'Thinking'}<AnimatedDots /></span>
            ) : showAnimatedDots ? (
              <span>{displayContent}<AnimatedDots /></span>
            ) : isFinalInsight && children ? (
              // Split content at Key Insight to insert published artifacts between Answer and Key Insight
              (() => {
                const splitMatch = displayContent.match(/\n\s*\*{0,2}Key Insight/)
                const splitIdx = splitMatch?.index ?? -1
                if (splitIdx > 0) {
                  const before = displayContent.slice(0, splitIdx)
                  const after = displayContent.slice(splitIdx)
                  return (
                    <>
                      <MarkdownContent content={before} />
                      {children}
                      <hr className="border-gray-200 dark:border-gray-700 my-3" />
                      <MarkdownContent content={after} />
                    </>
                  )
                }
                return (
                  <>
                    <MarkdownContent content={displayContent} />
                    {children}
                  </>
                )
              })()
            ) : (
              <MarkdownContent content={displayContent} />
            )}
          </div>
          )}
          {/* Action buttons row — non-step expand/collapse + view proof */}
          {((!isStep && !isUser && needsExpansion) || (isFinalInsight && onViewResult)) && (
            <div className="mt-2 flex items-center gap-3">
              {!isStep && !isUser && needsExpansion && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                >
                  {isExpanded ? (
                    <>
                      <ChevronUpIcon className="w-3 h-3" />
                      Show less
                    </>
                  ) : (
                    <>
                      <ChevronDownIcon className="w-3 h-3" />
                      Show more
                    </>
                  )}
                </button>
              )}
              {isFinalInsight && onViewResult && content.toLowerCase().includes('proof') && (
                <button
                  onClick={onViewResult}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md transition-colors"
                >
                  <EyeIcon className="w-4 h-4" />
                  View Reason-Chain
                </button>
              )}
            </div>
          )}
          {/* Action chips — Reading (green) during execution */}
          {isStep && stepSourcesRead && stepSourcesRead.length > 0 && isLive && !isPending && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {stepSourcesRead.map((source) => (
                <span key={source} className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400">
                  <CircleStackIcon className="w-3 h-3" />
                  Reading {source.split('.').pop()}
                </span>
              ))}
            </div>
          )}
          {/* Action chips — Reading (green) + Created (purple) after completion */}
          {isStep && stepDurationMs !== undefined && !isLive && (
            ((stepSourcesRead && stepSourcesRead.length > 0) || (stepOutputs && stepOutputs.length > 0)) && (
            <div className="flex flex-wrap gap-1 mt-1.5">
              {(stepSourcesRead || []).map((source) => (
                <span key={`read-${source}`} className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400">
                  Reading {source.split('.').pop()}
                </span>
              ))}
              {(stepOutputs || []).map((output) => (
                <button
                  key={output.id}
                  onClick={() => onOutputClick?.(output)}
                  className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-purple-50 text-purple-700 hover:bg-purple-100 dark:bg-purple-900/20 dark:text-purple-400 dark:hover:bg-purple-900/40 transition-colors"
                  title={`Jump to ${output.name}`}
                >
                  {output.type === 'table' ? <CircleStackIcon className="w-3 h-3" /> : <StarIcon className="w-3 h-3" />}
                  Created {output.name}
                </button>
              ))}
            </div>
            )
          )}
          {/* Render children at bottom only when not already inlined above */}
          {!(isFinalInsight && children) && children}
          {isEditing && stepNumber !== undefined && onStepEdit && (
            <div className="mt-2 flex gap-2">
              <input
                type="text"
                value={editGoal}
                onChange={(e) => setEditGoal(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && editGoal.trim()) {
                    onStepEdit(stepNumber, editGoal.trim())
                    setIsEditing(false)
                  } else if (e.key === 'Escape') {
                    setIsEditing(false)
                  }
                }}
                className="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                autoFocus
              />
              <button
                onClick={() => {
                  if (editGoal.trim()) {
                    onStepEdit(stepNumber, editGoal.trim())
                    setIsEditing(false)
                  }
                }}
                className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Replan
              </button>
              <button
                onClick={() => setIsEditing(false)}
                className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                Cancel
              </button>
            </div>
          )}
          </div>
          {/* Action buttons — dedicated right margin, stacked vertically */}
          {(
            <div className="flex flex-col gap-1 pt-1 w-6 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
              <button
                onClick={handleCopy}
                className={`p-1 rounded transition-all ${
                  copied
                    ? 'text-green-500 dark:text-green-400'
                    : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300'
                }`}
                title={copied ? 'Copied!' : 'Copy message'}
              >
                {copied ? (
                  <ClipboardDocumentCheckIcon className="w-4 h-4" />
                ) : (
                  <ClipboardDocumentIcon className="w-4 h-4" />
                )}
              </button>
              {isUser && onEditMessage && (
                <button
                  onClick={() => onEditMessage(content)}
                  className="p-1 rounded text-gray-400 hover:text-blue-500 dark:text-gray-500 dark:hover:text-blue-400 transition-colors"
                  title="Edit and resubmit"
                >
                  <PencilIcon className="w-4 h-4" />
                </button>
              )}
              {!isUser && queryText && (
                <FlagButton queryText={queryText} answerSummary={content.slice(0, 200)} />
              )}
              {isStep && stepDurationMs !== undefined && !isLive && !isSuperseded && onStepEdit && stepNumber !== undefined && (
                <button
                  onClick={() => {
                    setEditGoal(cleanedContent.split('\n')[0])
                    setIsEditing(true)
                  }}
                  className="p-1 rounded text-gray-400 hover:text-blue-500 dark:text-gray-500 dark:hover:text-blue-400 transition-colors"
                  title="Edit step goal and replan"
                >
                  <PencilIcon className="w-4 h-4" />
                </button>
              )}
              {isStep && stepDurationMs !== undefined && !isLive && !isSuperseded && onStepDelete && stepNumber !== undefined && (
                <button
                  onClick={() => onStepDelete(stepNumber)}
                  className="p-1 rounded text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 transition-colors"
                  title="Delete step and replan"
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
