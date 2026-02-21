// Message Bubble component

import { ReactNode, useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  UserIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  EyeIcon,
  ArrowPathIcon,
  LightBulbIcon,
  BoltIcon,
  CheckIcon,
} from '@heroicons/react/24/outline'

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
  onRedo?: (guidance?: string) => void
  children?: ReactNode
  role?: string // Role used for this step (e.g., "data_analyst")
  skills?: string[] // Skills used for this step
  stepStartedAt?: number
  stepDurationMs?: number
  stepAttempts?: number
  stepDisplayMode?: StepDisplayMode // External override for condense-all / expand-all
  stepDisplayModeVersion?: number // Increment to re-trigger override
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const remainSec = Math.round(seconds % 60)
  return `${minutes}m ${remainSec}s`
}

const typeStyles: Record<MessageType, { bg: string; text: string; icon: typeof UserIcon; iconColor: string }> = {
  user: {
    bg: 'bg-primary-100 dark:bg-primary-900/50',
    text: 'text-gray-900 dark:text-gray-100',
    icon: UserIcon,
    iconColor: 'text-primary-600 dark:text-primary-400',
  },
  system: {
    bg: 'bg-white dark:bg-gray-800',
    text: 'text-gray-900 dark:text-gray-100',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
  },
  thinking: {
    bg: 'bg-purple-50 dark:bg-purple-900/30',
    text: 'text-gray-700 dark:text-gray-300',
    icon: CpuChipIcon,
    iconColor: 'text-purple-500 dark:text-purple-400',
  },
  plan: {
    bg: 'bg-blue-50 dark:bg-blue-900/30',
    text: 'text-gray-900 dark:text-gray-100',
    icon: CpuChipIcon,
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
  step: {
    bg: 'bg-slate-100 dark:bg-gray-800',
    text: 'text-gray-900 dark:text-gray-100',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
  },
  output: {
    bg: 'bg-green-50 dark:bg-gray-800 border border-green-200 dark:border-green-700',
    text: 'text-gray-900 dark:text-gray-100',
    icon: CheckCircleIcon,
    iconColor: 'text-green-600 dark:text-green-400',
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-900/30',
    text: 'text-gray-900 dark:text-red-100',
    icon: ExclamationTriangleIcon,
    iconColor: 'text-red-600 dark:text-red-400',
  },
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
  onRedo,
  children,
  role,
  skills,
  stepStartedAt,
  stepDurationMs,
  stepAttempts,
  stepDisplayMode: externalStepMode,
  stepDisplayModeVersion: externalStepModeVersion,
}: MessageBubbleProps) {
  const styles = typeStyles[type]
  const Icon = styles.icon
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
  const [isExpanded, setIsExpanded] = useState(defaultExpanded ?? false)
  const [needsExpansion, setNeedsExpansion] = useState(false)
  const [copied, setCopied] = useState(false)
  const [showRedoForm, setShowRedoForm] = useState(false)
  const [redoGuidance, setRedoGuidance] = useState('')
  const contentRef = useRef<HTMLDivElement>(null)

  // Sync external step mode override (condense-all / expand-all)
  useEffect(() => {
    if (isStep && externalStepMode !== undefined) {
      setStepMode(externalStepMode)
    }
  }, [externalStepMode, externalStepModeVersion])

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
    <div className={`group flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-primary-100 dark:bg-primary-900' : 'bg-gray-100 dark:bg-gray-700'
        } ${isLive ? 'animate-pulse' : ''} ${isPending ? 'opacity-50' : ''}`}
      >
        <Icon className={`w-4 h-4 ${styles.iconColor}`} />
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`relative inline-block rounded-lg px-4 py-3 ${styles.bg} ${
            isUser ? 'rounded-tr-none' : 'rounded-tl-none'
          } ${isLive ? 'border-l-2 border-blue-500' : ''} ${isPending ? 'border-l-2 border-gray-300 dark:border-gray-600 opacity-60' : ''}`}
        >
          {/* Copy button for non-step messages - appears on hover, tucked into corner */}
          {!isStep && (
            <button
              onClick={handleCopy}
              className={`absolute top-[-2px] right-[-2px] p-1 rounded transition-all ${
                copied
                  ? 'text-green-500 dark:text-green-400'
                  : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 opacity-0 group-hover:opacity-100'
              }`}
              title={copied ? 'Copied!' : 'Copy message'}
            >
              {copied ? (
                <ClipboardDocumentCheckIcon className="w-4 h-4" />
              ) : (
                <ClipboardDocumentIcon className="w-4 h-4" />
              )}
            </button>
          )}
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
              {role && (
                <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300">
                  @{role}
                </span>
              )}
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
              {/* Copy button for step messages - inline in header after chevrons */}
              {isStep && (
                <button
                  onClick={handleCopy}
                  className={`p-0.5 rounded transition-all ${
                    copied
                      ? 'text-green-500 dark:text-green-400'
                      : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 opacity-0 group-hover:opacity-100'
                  }`}
                  title={copied ? 'Copied!' : 'Copy message'}
                >
                  {copied ? (
                    <ClipboardDocumentCheckIcon className="w-3.5 h-3.5" />
                  ) : (
                    <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                  )}
                </button>
              )}
            </div>
          )}
          {!(isStep && stepMode === 'oneline') && (
          <div
            ref={contentRef}
            className={`text-sm ${styles.text} ${
              isStep
                ? stepMode === 'condensed' ? 'overflow-y-auto' : ''
                : !isExpanded && needsExpansion ? 'overflow-y-auto pr-[5px]' : ''
            }`}
            style={{
              maxHeight: isStep
                ? stepMode === 'condensed' ? `${MAX_COLLAPSED_HEIGHT}px` : undefined
                : !isExpanded && needsExpansion ? `${MAX_COLLAPSED_HEIGHT}px` : undefined,
            }}
          >
            {type === 'thinking' ? (
              <span>{cleanedContent.replace(/\.+$/, '') || 'Thinking'}<AnimatedDots /></span>
            ) : showAnimatedDots ? (
              <span>{displayContent}<AnimatedDots /></span>
            ) : (
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
                      <SyntaxHighlighter
                        style={oneDark as { [key: string]: React.CSSProperties }}
                        language={match[1]}
                        PreTag="div"
                        customStyle={{
                          margin: '0.5rem 0',
                          padding: '0.75rem',
                          borderRadius: '0.375rem',
                          fontSize: '0.75rem',
                        }}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
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
                {displayContent}
              </ReactMarkdown>
            )}
          </div>
          )}
          {/* Action buttons row — non-step expand/collapse + view/redo */}
          {((!isStep && needsExpansion) || (isFinalInsight && onViewResult) || onRedo) && (
            <div className="mt-2 flex items-center gap-3">
              {!isStep && needsExpansion && (
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
              {isFinalInsight && onViewResult && (
                <button
                  onClick={onViewResult}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md transition-colors"
                >
                  <EyeIcon className="w-4 h-4" />
                  {content.toLowerCase().includes('proof') ? 'View Proof' : 'View Result'}
                </button>
              )}
              {isFinalInsight && stepDurationMs != null && stepDurationMs > 0 && (
                <span className="text-xs text-gray-400 dark:text-gray-500" title="Total elapsed time">
                  {formatMs(stepDurationMs)}
                </span>
              )}
              {onRedo && (
                <button
                  onClick={() => setShowRedoForm(true)}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-md transition-colors"
                >
                  <ArrowPathIcon className="w-4 h-4" />
                  Redo
                </button>
              )}
            </div>
          )}
          {showRedoForm && onRedo && (
            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => { setShowRedoForm(false); setRedoGuidance('') }}>
              <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-[480px] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
                <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">Redo Analysis</h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">What should be different this time? (optional)</p>
                </div>
                <form
                  className="px-5 py-4"
                  onSubmit={(e) => {
                    e.preventDefault()
                    onRedo(redoGuidance.trim() || undefined)
                    setShowRedoForm(false)
                    setRedoGuidance('')
                  }}
                >
                  <textarea
                    value={redoGuidance}
                    onChange={(e) => setRedoGuidance(e.target.value)}
                    placeholder="e.g. Focus on quarterly trends instead of monthly..."
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500 resize-none"
                    rows={4}
                    autoFocus
                  />
                  <div className="flex justify-end gap-2 mt-4">
                    <button
                      type="button"
                      onClick={() => { setShowRedoForm(false); setRedoGuidance('') }}
                      className="px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-lg transition-colors"
                    >
                      Redo
                    </button>
                  </div>
                </form>
              </div>
            </div>
          )}
          {children}
        </div>
        {timestamp && !isLive && (
          <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
            {timestamp.toLocaleTimeString()}
          </p>
        )}
      </div>
    </div>
  )
}