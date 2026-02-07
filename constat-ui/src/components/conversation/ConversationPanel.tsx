// Conversation Panel container

import { useEffect, useRef, useState, useCallback } from 'react'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useUIStore } from '@/store/uiStore'
import { useProofStore } from '@/store/proofStore'
import { MessageBubble } from './MessageBubble'
import { AutocompleteInput } from './AutocompleteInput'
import {
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  XMarkIcon,
  ClockIcon,
} from '@heroicons/react/24/outline'

export function ConversationPanel() {
  const { session, messages, submitQuery, queuedMessages, removeQueuedMessage, lastQueryStartStep, isCreatingSession } = useSessionStore()
  const { artifacts, tables } = useArtifactStore()
  const { openFullscreenArtifact } = useUIStore()
  const { openPanel: openProofPanel } = useProofStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [copiedAll, setCopiedAll] = useState(false)

  // Auto-scroll to bottom on new messages or queued messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, queuedMessages])

  const handleSubmit = (query: string) => {
    const isFollowup = messages.some((m) => m.type === 'user')
    submitQuery(query, isFollowup)
  }

  // Filter to only include items from the current query (step >= lastQueryStartStep)
  const isFromCurrentQuery = useCallback((stepNumber?: number): boolean => {
    if (lastQueryStartStep === 0) return true // No query started yet, include all
    return (stepNumber ?? 0) >= lastQueryStartStep
  }, [lastQueryStartStep])

  // Check if there are any viewable results from the current query
  const hasViewableResults = useCallback((): boolean => {
    const currentQueryArtifacts = artifacts.filter((a) => isFromCurrentQuery(a.step_number))
    const currentQueryTables = tables.filter((t) => isFromCurrentQuery(t.step_number))
    return currentQueryArtifacts.length > 0 || currentQueryTables.length > 0
  }, [artifacts, tables, isFromCurrentQuery])

  // Find and open the best artifact fullscreen (prioritize current query's artifacts)
  const handleViewResult = useCallback(() => {
    // Priority keywords for finding the best result
    const hasPriorityKeyword = (name?: string, title?: string): boolean => {
      const text = `${name || ''} ${title || ''}`.toLowerCase()
      return ['final', 'recommended', 'answer', 'result', 'conclusion'].some(kw => text.includes(kw))
    }

    // Helper to get the most recent item (highest step_number)
    const getMostRecent = <T extends { step_number?: number }>(items: T[]): T | undefined => {
      if (items.length === 0) return undefined
      return items.reduce((best, curr) =>
        (curr.step_number ?? 0) > (best.step_number ?? 0) ? curr : best
      )
    }

    // Key artifacts from current query (published/starred)
    const currentQueryArtifacts = artifacts.filter((a) => isFromCurrentQuery(a.step_number))
    let keyArtifacts = currentQueryArtifacts.filter((a) => a.is_key_result)

    // If no key artifacts in current query, fall back to ALL key artifacts
    if (keyArtifacts.length === 0) {
      keyArtifacts = artifacts.filter((a) => a.is_key_result)
      console.log('[viewResult] No key artifacts in current query, using all key artifacts')
    }

    // Markdown documents (highest priority for final results)
    const keyMarkdown = keyArtifacts.filter((a) =>
      ['markdown', 'md'].includes(a.artifact_type?.toLowerCase())
    )

    // Other visualizations (charts, images, etc.)
    const keyVisualizations = keyArtifacts.filter((a) =>
      ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega'].includes(a.artifact_type?.toLowerCase())
    )

    // Tables in key artifacts
    const keyTables = keyArtifacts.filter((a) => a.artifact_type === 'table')

    // Find best item - prioritize markdown documents
    console.log('[viewResult v2025-02-05] lastQueryStartStep:', lastQueryStartStep)
    console.log('[viewResult v2025-02-05] keyArtifacts:', keyArtifacts.map(a => `${a.id}:${a.name}(${a.artifact_type})`))
    console.log('[viewResult v2025-02-05] keyMarkdown:', keyMarkdown.map(a => `${a.id}:${a.name}`))
    if (keyMarkdown.length > 0) {
      const best = keyMarkdown.find(a => hasPriorityKeyword(a.name, a.title)) || getMostRecent(keyMarkdown)
      console.log('[viewResult v2025-02-05] Selected markdown:', best?.id, best?.name)
      if (best) openFullscreenArtifact({ type: 'artifact', id: best.id })
    } else if (keyVisualizations.length > 0) {
      const best = keyVisualizations.find(a => hasPriorityKeyword(a.name, a.title)) || getMostRecent(keyVisualizations)
      if (best) openFullscreenArtifact({ type: 'artifact', id: best.id })
    } else if (keyTables.length > 0) {
      const best = keyTables.find(a => hasPriorityKeyword(a.name, a.title)) || getMostRecent(keyTables)
      if (best) openFullscreenArtifact({ type: 'table', name: best.name })
    } else if (tables.length > 0) {
      // Fallback to tables from current query
      const currentQueryTables = tables.filter(t => isFromCurrentQuery(t.step_number))
      if (currentQueryTables.length > 0) {
        const best = currentQueryTables.find(t => hasPriorityKeyword(t.name)) || getMostRecent(currentQueryTables)
        if (best) openFullscreenArtifact({ type: 'table', name: best.name })
      } else {
        // Ultimate fallback - most recent table overall
        const best = getMostRecent(tables)
        if (best) openFullscreenArtifact({ type: 'table', name: best.name })
      }
    }
  }, [artifacts, tables, isFromCurrentQuery, openFullscreenArtifact])

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
                isFinalInsight={message.isFinalInsight}
                onViewResult={message.isFinalInsight ? (
                  message.content?.toLowerCase().includes('proof') ? openProofPanel : (hasViewableResults() ? handleViewResult : undefined)
                ) : undefined}
                role={message.role}
                skills={message.skills}
              />
            ))}
            {/* Queued messages */}
            {queuedMessages.map((queued, index) => (
              <div key={queued.id} className="group flex gap-3 flex-row-reverse">
                {/* Avatar placeholder for alignment */}
                <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-primary-100 dark:bg-primary-900 opacity-50">
                  <ClockIcon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                </div>
                {/* Queued message content */}
                <div className="flex-1 max-w-[80%] text-right">
                  <div className="relative inline-block rounded-lg rounded-tr-none px-4 py-3 bg-primary-100/50 dark:bg-primary-900/30 border border-dashed border-primary-300 dark:border-primary-700">
                    {/* Cancel button */}
                    <button
                      onClick={() => removeQueuedMessage(queued.id)}
                      className="absolute top-2 right-2 p-1 rounded text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="Cancel queued message"
                    >
                      <XMarkIcon className="w-4 h-4" />
                    </button>
                    {/* Queued badge */}
                    <div className="flex items-center gap-1.5 text-xs text-primary-600 dark:text-primary-400 mb-1">
                      <ClockIcon className="w-3 h-3" />
                      <span>Queued {index > 0 ? `#${index + 1}` : ''}</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                      {queued.content}
                    </p>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Query input */}
      <AutocompleteInput onSubmit={handleSubmit} disabled={isCreatingSession} />
    </div>
  )
}