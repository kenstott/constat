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
    bg: 'bg-gray-50 dark:bg-gray-800',
    text: 'text-gray-600 dark:text-gray-400',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
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
  children,
}: MessageBubbleProps) {
  const styles = typeStyles[type]
  const Icon = styles.icon
  const isUser = type === 'user'

  // Strip "Step X:" or "Step X" prefix from content if stepNumber is shown in header
  const cleanedContent = stepNumber !== undefined
    ? content.replace(/^Step\s+\d+:?\s*/i, '')
    : content

  // Expand/collapse state
  const [isExpanded, setIsExpanded] = useState(defaultExpanded ?? false)
  const [needsExpansion, setNeedsExpansion] = useState(false)
  const [copied, setCopied] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)

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
  }, [content])

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
          {/* Copy button - appears on hover */}
          <button
            onClick={handleCopy}
            className={`absolute top-2 right-2 p-1 rounded transition-all ${
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
          {stepNumber !== undefined && (
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Step {stepNumber}
            </div>
          )}
          <div
            ref={contentRef}
            className={`text-sm ${styles.text} ${
              !isExpanded && needsExpansion
                ? 'overflow-y-auto pr-[5px]'
                : ''
            }`}
            style={{
              maxHeight: !isExpanded && needsExpansion ? `${MAX_COLLAPSED_HEIGHT}px` : undefined,
            }}
          >
            {type === 'thinking' ? (
              <span>{cleanedContent.replace(/\.+$/, '')}<AnimatedDots /></span>
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
          {/* View Result button for final insights */}
          {isFinalInsight && onViewResult && (
            <button
              onClick={onViewResult}
              className="mt-3 flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md transition-colors"
            >
              <EyeIcon className="w-4 h-4" />
              View Result
            </button>
          )}
          {/* Expand/Collapse button */}
          {needsExpansion && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-2 flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
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