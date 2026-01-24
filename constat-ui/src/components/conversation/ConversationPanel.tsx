// Conversation Panel container

import { useEffect, useRef } from 'react'
import { useSessionStore } from '@/store/sessionStore'
import { MessageBubble } from './MessageBubble'
import { QueryInput } from './QueryInput'

export function ConversationPanel() {
  const { session, messages, submitQuery } = useSessionStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (query: string) => {
    const isFollowup = messages.some((m) => m.type === 'user')
    submitQuery(query, isFollowup)
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
              />
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Query input */}
      <QueryInput onSubmit={handleSubmit} />
    </div>
  )
}