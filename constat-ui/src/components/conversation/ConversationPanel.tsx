// Conversation Panel container

import { useEffect, useRef, useState, useMemo } from 'react'
import { useSessionStore } from '@/store/sessionStore'
import { useProofStore } from '@/store/proofStore'
import { useArtifactStore } from '@/store/artifactStore'
import { MessageBubble, StepDisplayMode } from './MessageBubble'
import { BotMessageGroup } from './BotMessageGroup'
import { AutocompleteInput } from './AutocompleteInput'
import {
  ClipboardDocumentIcon,
  ClipboardDocumentCheckIcon,
  XMarkIcon,
  ClockIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ShareIcon,
  LinkIcon,
} from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'
import { useAuthStore } from '@/store/authStore'
import * as sessionsApi from '@/api/sessions'

// Use the actual messages type from the store
type StoreMessage = ReturnType<typeof useSessionStore.getState>['messages'][number]

type MessageGroupTyped =
  | { kind: 'user'; message: StoreMessage }
  | { kind: 'bot'; messages: StoreMessage[] }
  | { kind: 'single'; message: StoreMessage }

function groupMessages(messages: StoreMessage[]): MessageGroupTyped[] {
  const groups: MessageGroupTyped[] = []
  let currentBotMessages: StoreMessage[] = []

  const flushBot = () => {
    if (currentBotMessages.length > 0) {
      groups.push({ kind: 'bot', messages: [...currentBotMessages] })
      currentBotMessages = []
    }
  }

  for (const msg of messages) {
    if (msg.type === 'user') {
      flushBot()
      groups.push({ kind: 'user', message: msg })
    } else if (msg.type === 'error') {
      flushBot()
      groups.push({ kind: 'single', message: msg })
    } else {
      // thinking, plan, step, output, system — group together
      currentBotMessages.push(msg)
    }
  }
  flushBot()

  return groups
}

export function ConversationPanel() {
  const { session, messages, suggestions, welcomeTagline, submitQuery, queuedMessages, removeQueuedMessage, isCreatingSession, shareSession, replanFromStep } = useSessionStore()
  const { openPanel: openProofPanel } = useProofStore()
  const { tables, artifacts } = useArtifactStore()
  const { expandArtifactSection, expandResultStep, showArtifactPanel } = useUIStore()

  const handleRoleClick = (role: string) => {
    showArtifactPanel()
    expandArtifactSection('agents')
    setTimeout(() => {
      const el = document.getElementById(`agent-${role}`)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('ring-2', 'ring-purple-400')
        setTimeout(() => el.classList.remove('ring-2', 'ring-purple-400'), 2000)
      }
    }, 150)
  }

  const handleOutputClick = (stepNumber: number | undefined, output: { type: 'table' | 'artifact'; name: string; id: string }) => {
    // Ensure the artifact panel is visible and results section is open
    showArtifactPanel()
    expandArtifactSection('results')
    // Switch to "all" so unpublished items are visible
    useUIStore.getState().setResultsShowPublishedOnly(false)
    // Ensure the step group is expanded
    if (stepNumber) expandResultStep(stepNumber)
    // Scroll to the item after DOM updates
    setTimeout(() => {
      const item = document.getElementById(output.id)
      if (item) {
        item.scrollIntoView({ behavior: 'smooth', block: 'center' })
        item.classList.add('ring-2', 'ring-primary-400')
        setTimeout(() => item.classList.remove('ring-2', 'ring-primary-400'), 2000)
      }
    }, 150)
  }

  // Build step outputs map: stepNumber -> array of {type, name, id}
  const stepOutputsMap = useMemo(() => {
    const map = new Map<number, Array<{ type: 'table' | 'artifact'; name: string; id: string }>>()
    for (const t of tables) {
      if (t.step_number > 0) {
        const arr = map.get(t.step_number) || []
        arr.push({ type: 'table', name: t.name, id: `table-${t.name}` })
        map.set(t.step_number, arr)
      }
    }
    const internalTypes = new Set(['code', 'output', 'error', 'stdout', 'stderr', 'table'])
    for (const a of artifacts) {
      if (a.step_number > 0 && !internalTypes.has(a.artifact_type)) {
        const arr = map.get(a.step_number) || []
        arr.push({ type: 'artifact', name: a.title || a.name, id: `artifact-${a.id}` })
        map.set(a.step_number, arr)
      }
    }
    return map
  }, [tables, artifacts])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [copiedAll, setCopiedAll] = useState(false)
  const [stepOverride, setStepOverride] = useState<{ mode: StepDisplayMode; version: number } | undefined>()
  const [insightOverride, setInsightOverride] = useState<{ collapsed: boolean; version: number } | undefined>()
  const [groupOverride, setGroupOverride] = useState<{ expanded: boolean; version: number } | undefined>()
  const [shareOpen, setShareOpen] = useState(false)
  const [shareEmail, setShareEmail] = useState('')
  const [shareResult, setShareResult] = useState<string | null>(null)
  const [shareError, setShareError] = useState<string | null>(null)
  const [isPublic, setIsPublic] = useState(session?.is_public ?? false)
  const [publicUrl, setPublicUrl] = useState<string | null>(null)
  const [publicCopied, setPublicCopied] = useState(false)
  const [editValue, setEditValue] = useState<string | null>(null)

  const hasSteps = messages.some((m) => m.type === 'step')
  const hasInsights = messages.some((m) => m.isFinalInsight)

  // Compute grouped messages
  const groups = useMemo(() => groupMessages(messages), [messages])

  // Auto-scroll to bottom on new messages or queued messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, queuedMessages])

  const handleSubmit = (query: string) => {
    // Check both client-side messages and server-side session state for follow-up detection
    // session.status === 'completed' catches restored sessions where messages were cleared
    const isFollowup = messages.some((m) => m.type === 'user') || session?.status === 'completed'
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

  const handleShare = async () => {
    if (!shareEmail.trim()) return
    setShareError(null)
    setShareResult(null)
    try {
      const result = await shareSession(shareEmail.trim())
      setShareResult(result.share_url)
      setShareEmail('')
    } catch (err: unknown) {
      setShareError(err instanceof Error ? err.message : 'Failed to share')
    }
  }

  const handleTogglePublic = async () => {
    if (!session) return
    const newPublic = !isPublic
    try {
      const result = await sessionsApi.togglePublicSharing(session.session_id, newPublic)
      setIsPublic(result.public)
      setPublicUrl(result.public ? result.share_url : null)
    } catch (err: unknown) {
      setShareError(err instanceof Error ? err.message : 'Failed to toggle public sharing')
    }
  }

  const handleCopyPublicUrl = () => {
    if (publicUrl) {
      navigator.clipboard.writeText(publicUrl)
      setPublicCopied(true)
      setTimeout(() => setPublicCopied(false), 2000)
    }
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

  const greeting = (() => {
    const user = useAuthStore.getState().user
    const firstName = user?.displayName?.split(' ')[0]
    return firstName ? `${firstName}, what can I help you with?` : 'What can I help you with?'
  })()

  // Intro screen: greeting + suggestions + input centered together
  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1" />
        <div className="w-full max-w-2xl mx-auto px-6">
          {welcomeTagline && (
            <p className="text-base text-gray-600 dark:text-gray-300 mb-2 text-center"
               dangerouslySetInnerHTML={{ __html: welcomeTagline.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/_(.+?)_/g, '<em>$1</em>') }} />
          )}
          <h1 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6 text-center">
            {greeting}
          </h1>
          {suggestions.length > 0 && (
            <div className="flex flex-wrap justify-center gap-2 mb-6">
              {suggestions.map((s, i) => (
                <button
                  key={i}
                  onClick={() => handleSubmit(s)}
                  className="px-3 py-1.5 text-sm rounded-full border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          <AutocompleteInput onSubmit={(q) => { setEditValue(null); handleSubmit(q) }} disabled={isCreatingSession} editValue={editValue} />
        </div>
        <div className="flex-1" />
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="flex justify-end gap-1 pl-4 pr-12 pt-2 opacity-0 hover:opacity-100 focus-within:opacity-100 transition-opacity duration-200">
        {(hasSteps || hasInsights) && (
          <>
            <button
              onClick={() => {
                setStepOverride({ mode: 'oneline', version: (stepOverride?.version ?? 0) + 1 })
                setInsightOverride({ collapsed: true, version: (insightOverride?.version ?? 0) + 1 })
                setGroupOverride({ expanded: false, version: (groupOverride?.version ?? 0) + 1 })
              }}
              className="flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              title="Collapse all steps"
            >
              <ChevronUpIcon className="w-3.5 h-3.5" />
              Collapse
            </button>
            <button
              onClick={() => {
                setStepOverride({ mode: 'condensed', version: (stepOverride?.version ?? 0) + 1 })
                setInsightOverride({ collapsed: false, version: (insightOverride?.version ?? 0) + 1 })
                setGroupOverride({ expanded: true, version: (groupOverride?.version ?? 0) + 1 })
              }}
              className="flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              title="Expand all steps"
            >
              <ChevronDownIcon className="w-3.5 h-3.5" />
              Expand
            </button>
          </>
        )}
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
        <div className="relative">
          <button
            onClick={() => { setShareOpen(!shareOpen); setShareResult(null); setShareError(null) }}
            className="flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
            title="Share session"
          >
            <ShareIcon className="w-4 h-4" />
            Share
          </button>
          {shareOpen && (
            <div className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 w-72">
              <div className="text-xs font-medium text-gray-700 dark:text-gray-300 mb-2">Share this session</div>

              {/* Public link toggle */}
              <div className="flex items-center justify-between mb-2 pb-2 border-b border-gray-100 dark:border-gray-700">
                <div className="flex items-center gap-1.5">
                  <LinkIcon className="w-3.5 h-3.5 text-gray-400" />
                  <span className="text-xs text-gray-600 dark:text-gray-400">Public link</span>
                </div>
                <button
                  onClick={handleTogglePublic}
                  className={`relative w-8 h-4 rounded-full transition-colors ${
                    isPublic ? 'bg-blue-500' : 'bg-gray-300 dark:bg-gray-600'
                  }`}
                >
                  <span className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform ${
                    isPublic ? 'translate-x-4' : 'translate-x-0.5'
                  }`} />
                </button>
              </div>
              {isPublic && publicUrl && (
                <div className="flex gap-1 mb-2">
                  <input
                    type="text"
                    value={publicUrl}
                    readOnly
                    className="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-gray-50 dark:bg-gray-700 text-gray-700 dark:text-gray-300 truncate"
                  />
                  <button
                    onClick={handleCopyPublicUrl}
                    className={`px-2 py-1 text-xs rounded transition-colors ${
                      publicCopied ? 'bg-green-100 dark:bg-green-900/30 text-green-600' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                    }`}
                  >
                    {publicCopied ? 'Copied' : 'Copy'}
                  </button>
                </div>
              )}

              {/* Email sharing */}
              <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Or invite by email</div>
              <div className="flex gap-1">
                <input
                  type="email"
                  value={shareEmail}
                  onChange={(e) => setShareEmail(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleShare()}
                  placeholder="Email address"
                  className="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  autoFocus
                />
                <button
                  onClick={handleShare}
                  className="px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                >
                  Send
                </button>
              </div>
              {shareResult && (
                <div className="mt-2 text-xs text-green-600 dark:text-green-400">
                  <span>Shared! </span>
                  <button
                    onClick={() => { navigator.clipboard.writeText(shareResult); }}
                    className="underline hover:no-underline"
                  >
                    Copy link
                  </button>
                </div>
              )}
              {shareError && (
                <div className="mt-2 text-xs text-red-600 dark:text-red-400">{shareError}</div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-4 space-y-6">
          {groups.map((group) => {
            if (group.kind === 'user') {
              const message = group.message
              return (
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
                  onViewResult={message.isFinalInsight && message.content?.toLowerCase().includes('proof')
                    ? openProofPanel : undefined}
                  role={message.role}
                  skills={message.skills}
                  stepStartedAt={message.stepStartedAt}
                  stepDurationMs={message.stepDurationMs}
                  stepAttempts={message.stepAttempts}
                  stepDisplayMode={message.type === 'step' ? stepOverride?.mode : undefined}
                  stepDisplayModeVersion={stepOverride?.version}
                  contentExpanded={groupOverride?.expanded}
                  contentExpandedVersion={groupOverride?.version}
                  isSuperseded={message.isSuperseded}
                  onStepEdit={(stepNumber, newGoal) => replanFromStep(stepNumber, 'edit', newGoal)}
                  onStepDelete={(stepNumber) => replanFromStep(stepNumber, 'delete')}
                  stepOutputs={message.stepNumber ? stepOutputsMap.get(message.stepNumber) : undefined}
                  onOutputClick={(output) => handleOutputClick(message.stepNumber, output)}
                  onRoleClick={handleRoleClick}
                  onEditMessage={(text) => setEditValue(text)}
                />
              )
            }

            if (group.kind === 'single') {
              const message = group.message
              let queryText: string | undefined
              for (let i = messages.indexOf(message) - 1; i >= 0; i--) {
                if (messages[i].type === 'user') {
                  queryText = messages[i].content
                  break
                }
              }
              return (
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
                  contentExpanded={groupOverride?.expanded}
                  contentExpandedVersion={groupOverride?.version}
                  queryText={queryText}
                  isSuperseded={message.isSuperseded}
                />
              )
            }

            // Bot group
            return (
              <BotMessageGroup
                key={group.messages[0].id}
                messages={group.messages}
                stepOverride={stepOverride}
                insightOverride={insightOverride}
                groupOverride={groupOverride}
                stepOutputsMap={stepOutputsMap}
                onOutputClick={handleOutputClick}
                onRoleClick={handleRoleClick}
                onStepEdit={(stepNumber, newGoal) => replanFromStep(stepNumber, 'edit', newGoal)}
                onStepDelete={(stepNumber) => replanFromStep(stepNumber, 'delete')}
                openProofPanel={openProofPanel}
                allMessages={messages}
              />
            )
          })}
          {/* Queued messages */}
          {queuedMessages.map((queued, index) => (
            <div key={queued.id} className="group">
              <div className="flex items-center gap-3 mb-1">
                <div className="w-8 h-8 rounded-full flex items-center justify-center bg-primary-100 dark:bg-primary-900 opacity-50">
                  <ClockIcon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                </div>
                <span className="font-semibold text-sm text-gray-900 dark:text-gray-100">You</span>
                <span className="text-xs text-primary-600 dark:text-primary-400">Queued {index > 0 ? `#${index + 1}` : ''}</span>
              </div>
              <div className="ml-11 relative">
                <button
                  onClick={() => removeQueuedMessage(queued.id)}
                  className="absolute top-0 right-0 p-1 rounded text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Cancel queued message"
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
                <p className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap opacity-60">
                  {queued.content}
                </p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Query input */}
      <AutocompleteInput onSubmit={(q) => { setEditValue(null); handleSubmit(q) }} disabled={isCreatingSession} editValue={editValue} />
    </div>
  )
}
