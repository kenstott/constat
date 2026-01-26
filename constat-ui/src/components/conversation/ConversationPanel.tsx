// Conversation Panel container

import { useEffect, useRef, useState } from 'react'
import { useSessionStore } from '@/store/sessionStore'
import { MessageBubble } from './MessageBubble'
import { AutocompleteInput } from './AutocompleteInput'
import {
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
} from '@heroicons/react/24/outline'

export function ConversationPanel() {
  const { session, messages, submitQuery } = useSessionStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [copiedAll, setCopiedAll] = useState(false)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (query: string) => {
    const isFollowup = messages.some((m) => m.type === 'user')
    submitQuery(query, isFollowup)
  }

  // Copy entire conversation to clipboard
  const handleCopyAll = async () => {
    const conversationText = messages
      .map((m) => {
        const role = m.type === 'user' ? 'User' : 'Assistant'
        return `${role}: ${m.content}`
      })
      .join('\n\n')
    await navigator.clipboard.writeText(conversationText)
    setCopiedAll(true)
    setTimeout(() => setCopiedAll(false), 2000)
  }

  if (!session) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center text-gray-500 dark:text-gray-400">
          <p className="text-lg font-medium">No active session</p>
          <p className="text-sm">Create a session to start querying your data</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Copy All button - shown when there are messages */}
      {messages.length > 0 && (
        <div className="flex justify-end px-4 pt-2">
          <button
            onClick={handleCopyAll}
            className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors ${
              copiedAll
                ? 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20'
                : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
            }`}
            title="Copy entire conversation"
          >
            {copiedAll ? (
              <>
                <ClipboardDocumentCheckIcon className="w-4 h-4" />
                Copied!
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="w-4 h-4" />
                Copy All
              </>
            )}
          </button>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-gray-500 dark:text-gray-400 max-w-md">
              <p className="text-lg font-medium mb-2">Ready to analyze your data</p>
              <p className="text-sm">
                Ask questions in natural language. For example:
              </p>
              <ul className="mt-3 text-sm text-left space-y-1">
                <li>"What are the top 10 customers by revenue?"</li>
                <li>"Show me monthly sales trends for 2024"</li>
                <li>"Which products have declining sales?"</li>
              </ul>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                type={message.type}
                content={message.content}
                timestamp={message.timestamp}
                stepNumber={message.stepNumber}
                isLive={message.isLive}
                isPending={message.isPending}
                defaultExpanded={message.defaultExpanded}
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Query input */}
      <AutocompleteInput onSubmit={handleSubmit} />
    </div>
  )
}