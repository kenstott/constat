// Message Bubble component

import { ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  UserIcon,
  CpuChipIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
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
  children?: ReactNode
}

const typeStyles: Record<MessageType, { bg: string; icon: typeof UserIcon; iconColor: string }> = {
  user: {
    bg: 'bg-primary-100 dark:bg-primary-800',
    icon: UserIcon,
    iconColor: 'text-primary-600 dark:text-primary-400',
  },
  system: {
    bg: 'bg-gray-50 dark:bg-gray-800',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
  },
  thinking: {
    bg: 'bg-gray-50 dark:bg-gray-800',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
  },
  plan: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    icon: CpuChipIcon,
    iconColor: 'text-blue-600 dark:text-blue-400',
  },
  step: {
    bg: 'bg-gray-50 dark:bg-gray-800',
    icon: CpuChipIcon,
    iconColor: 'text-gray-500 dark:text-gray-400',
  },
  output: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    icon: CheckCircleIcon,
    iconColor: 'text-green-600 dark:text-green-400',
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    icon: ExclamationTriangleIcon,
    iconColor: 'text-red-600 dark:text-red-400',
  },
}

export function MessageBubble({
  type,
  content,
  timestamp,
  stepNumber,
  isLive,
  children,
}: MessageBubbleProps) {
  const styles = typeStyles[type]
  const Icon = styles.icon
  const isUser = type === 'user'

  // Check if content ends with "..." to show animated dots
  const showAnimatedDots = isLive && content.endsWith('...')
  const displayContent = showAnimatedDots ? content.slice(0, -3) : content

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser ? 'bg-primary-100 dark:bg-primary-900' : 'bg-gray-100 dark:bg-gray-700'
        } ${isLive ? 'animate-pulse' : ''}`}
      >
        <Icon className={`w-4 h-4 ${styles.iconColor}`} />
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[80%] ${isUser ? 'text-right' : ''}`}>
        <div
          className={`inline-block rounded-lg px-4 py-3 ${styles.bg} ${
            isUser ? 'rounded-tr-none' : 'rounded-tl-none'
          } ${isLive ? 'border-l-2 border-blue-500' : ''}`}
        >
          {stepNumber !== undefined && (
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              Step {stepNumber}
            </div>
          )}
          <div className="text-sm text-gray-800 dark:text-gray-200">
            {type === 'thinking' ? (
              <AnimatedDots />
            ) : showAnimatedDots ? (
              <span>{displayContent}<AnimatedDots /></span>
            ) : (
              <ReactMarkdown
                components={{
                  p: ({ children }) => <p className="whitespace-pre-wrap mb-2 last:mb-0">{children}</p>,
                  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                  ul: ({ children }) => <ul className="list-disc list-inside mb-2">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal list-inside mb-2">{children}</ol>,
                  li: ({ children }) => <li className="mb-1">{children}</li>,
                  code: ({ children }) => (
                    <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs">{children}</code>
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            )}
          </div>
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