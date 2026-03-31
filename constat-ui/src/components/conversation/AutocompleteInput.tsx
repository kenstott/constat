// Copyright (c) 2025 Kenneth Stott
// Canary: 0f66d623-7041-45f3-9d3e-ccc963839489
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Autocomplete-enabled query input wrapper

import { useState, useRef, useEffect, useCallback, KeyboardEvent } from 'react'
import { QueueListIcon, ArrowUpIcon } from '@heroicons/react/24/solid'
import { XMarkIcon, AtSymbolIcon, PaperClipIcon, CheckBadgeIcon, BoltIcon, StopIcon } from '@heroicons/react/24/outline'
import { useArtifactContext } from '@/contexts/ArtifactContext'
import { useReactiveVar } from '@apollo/client'
import { briefModeVar, toggleBriefMode, enterReasonChainMode } from '@/graphql/ui-state'
// proofStore actions accessed via SessionContext
import { AutocompleteDropdown, AutocompleteItem } from './AutocompleteDropdown'
import { useSessionContext } from '@/contexts/SessionContext'
import { apolloClient } from '@/graphql/client'
import { REQUEST_AUTOCOMPLETE } from '@/graphql/operations/execution'
import {
  filterCommands,
  getCommandByName,
  DISCOVER_SCOPES,
  SUMMARIZE_TARGETS,
} from '@/data/commands'
import { listAllPermissions, type UserPermissions } from '@/api/users'
import { useAuth } from '@/contexts/AuthContext'

interface AutocompleteInputProps {
  onSubmit: (query: string) => void
  disabled?: boolean
  editValue?: string | null  // When set, populates input and focuses it
}

type CompletionContext = 'command' | 'table' | 'entity' | 'scope' | 'mention' | 'none'

interface ParsedInput {
  context: CompletionContext
  command?: string
  prefix: string
  parent?: string
}

function parseInput(value: string, cursorPosition?: number): ParsedInput {
  const cursor = cursorPosition ?? value.length

  // Check for @mention before cursor
  const beforeCursor = value.slice(0, cursor)
  const atMatch = beforeCursor.match(/(^|\s)@(\S*)$/)
  if (atMatch) {
    return { context: 'mention', prefix: '@' + atMatch[2] }
  }

  const trimmed = value.trimStart()

  // Check if starts with /
  if (!trimmed.startsWith('/')) {
    return { context: 'none', prefix: '' }
  }

  // Find the command (first word)
  const spaceIndex = trimmed.indexOf(' ')

  if (spaceIndex === -1) {
    // Still typing the command itself
    return { context: 'command', prefix: trimmed }
  }

  // Command is complete, check what argument it expects
  const command = trimmed.substring(0, spaceIndex)
  const rest = trimmed.substring(spaceIndex + 1)
  const commandDef = getCommandByName(command)

  if (!commandDef || !commandDef.argType) {
    return { context: 'none', prefix: '' }
  }

  // Map argType to completion context
  switch (commandDef.argType) {
    case 'table':
      return { context: 'table', command, prefix: rest }
    case 'entity':
      return { context: 'entity', command, prefix: rest }
    case 'scope':
      return { context: 'scope', command, prefix: rest }
    default:
      return { context: 'none', prefix: '' }
  }
}

function getMentionCompletions(prefix: string, users: { user_id: string; email: string | null }[]): AutocompleteItem[] {
  const lowerPrefix = prefix.toLowerCase()
  const mentions: AutocompleteItem[] = [
    { label: '@vera', value: '@vera', description: 'Teach Vera a rule or correction', category: 'Learning' },
    ...users.map((u) => {
      const displayName = u.email ? u.email.split('@')[0] : u.user_id
      return {
        label: `@${displayName}`,
        value: `@${u.user_id}`,
        description: u.email ?? u.user_id,
        category: 'Share',
      }
    }),
  ]
  return mentions.filter((m) => m.label.toLowerCase().startsWith(lowerPrefix) || m.value.toLowerCase().startsWith(lowerPrefix))
}

function getClientCompletions(parsed: ParsedInput, users: { user_id: string; email: string | null }[] = []): AutocompleteItem[] {
  if (parsed.context === 'mention') {
    return getMentionCompletions(parsed.prefix, users)
  }

  if (parsed.context === 'command') {
    return filterCommands(parsed.prefix).map((cmd) => ({
      label: cmd.command,
      value: cmd.command,
      description: cmd.description,
      category: cmd.category,
    }))
  }

  if (parsed.context === 'scope') {
    const lowerPrefix = parsed.prefix.toLowerCase()
    return DISCOVER_SCOPES.filter((s) => s.startsWith(lowerPrefix)).map((scope) => ({
      label: scope,
      value: scope,
      description: `Search ${scope} sources`,
    }))
  }

  // For /summarize, provide static options first, then server-side tables
  if (parsed.context === 'entity' && parsed.command === '/summarize') {
    const lowerPrefix = parsed.prefix.toLowerCase()
    return SUMMARIZE_TARGETS.filter((t) => t.startsWith(lowerPrefix)).map((target) => ({
      label: target,
      value: target,
      description: `Summarize ${target}`,
    }))
  }

  return []
}

const HISTORY_STORAGE_KEY = 'constat-query-history'
const MAX_HISTORY_SIZE = 100

function loadHistory(): string[] {
  try {
    const stored = localStorage.getItem(HISTORY_STORAGE_KEY)
    if (stored) {
      const parsed = JSON.parse(stored)
      if (Array.isArray(parsed)) {
        return parsed.slice(-MAX_HISTORY_SIZE)
      }
    }
  } catch (e) {
    console.warn('Failed to load query history:', e)
  }
  return []
}

function saveHistory(history: string[]): void {
  try {
    // Keep only the last MAX_HISTORY_SIZE entries
    const trimmed = history.slice(-MAX_HISTORY_SIZE)
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmed))
  } catch (e) {
    console.warn('Failed to save query history:', e)
  }
}

function InputToolbar({
  query,
  setQuery,
  textareaRef,
  isBusy,
  isDisabled,
  onSubmit,
  onInsertAt,
}: {
  query: string
  setQuery: (v: string) => void
  textareaRef: React.RefObject<HTMLTextAreaElement | null>
  isBusy: boolean
  isDisabled?: boolean
  onSubmit: () => void
  onInsertAt: () => void
}) {
  const { session, status, cancelExecution, submitQuery, openProofPanel, clearProofFacts: clearFacts } = useSessionContext()
  const { stepCodes, tables } = useArtifactContext()
  const briefMode = useReactiveVar(briefModeVar)

  const isExecuting = status === 'planning' || status === 'executing'
  const hasExecutedPlan = stepCodes.length > 0 || tables.length > 0

  const handleShowProof = () => {
    clearFacts()
    openProofPanel()
    enterReasonChainMode()
    submitQuery('/reason', true)
  }

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleAttachClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelected = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return
    if (!session) return
    try {
      const { uploadDocuments } = await import('@/api/sessions')
      await uploadDocuments(session.session_id, Array.from(files))
    } catch (err) {
      console.error('File upload failed:', err)
    }
    // Reset so the same file can be re-selected
    e.target.value = ''
  }

  return (
    <div className="flex items-center justify-between px-3 pb-2.5">
      <div className="flex items-center gap-1">
        <button
          onClick={handleAttachClick}
          className="px-2 py-1 rounded-md text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          title="Attach file as data source"
        >
          <PaperClipIcon className="w-4 h-4" />
        </button>
        <button
          onClick={onInsertAt}
          className="px-2 py-1 rounded-md text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          title="Insert @mention"
        >
          <AtSymbolIcon className="w-4 h-4" />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          className="hidden"
          onChange={handleFileSelected}
        />
        <div className="w-px h-4 bg-gray-200 dark:bg-gray-700 mx-1" />
        {isExecuting && (
          <button
            onClick={cancelExecution}
            className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 animate-pulse transition-colors"
            title="Cancel execution"
          >
            <StopIcon className="w-3.5 h-3.5" />
            Cancel
          </button>
        )}
        <button
          onClick={handleShowProof}
          disabled={isExecuting || !hasExecutedPlan}
          className="flex items-center gap-1 px-2 py-1 rounded-md text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-30 transition-colors"
          title={hasExecutedPlan ? 'Show reasoning chain' : 'Execute a query first'}
        >
          <CheckBadgeIcon className="w-3.5 h-3.5" />
          Reason-Chain
        </button>
        <button
          onClick={toggleBriefMode}
          className={`flex items-center gap-1 px-2 py-1 rounded-md text-xs transition-colors ${
            briefMode ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
          title={briefMode ? 'Brief mode ON' : 'Brief mode OFF'}
        >
          <BoltIcon className="w-3.5 h-3.5" />
          Brief
        </button>
      </div>
      <div className="flex items-center gap-1">
        {query && (
          <button
            onClick={() => {
              setQuery('')
              textareaRef.current?.focus()
            }}
            className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title="Clear"
          >
            <XMarkIcon className="w-4 h-4" />
          </button>
        )}
        <button
          onClick={onSubmit}
          disabled={isDisabled || !query.trim()}
          className={`w-8 h-8 flex items-center justify-center rounded-full text-white transition-colors disabled:opacity-30 disabled:cursor-not-allowed ${
            isBusy && query.trim()
              ? 'bg-amber-500 hover:bg-amber-600'
              : 'bg-green-600 hover:bg-green-700'
          }`}
          title={isBusy && query.trim() ? 'Queue message' : 'Send message'}
        >
          {isBusy && query.trim() ? (
            <QueueListIcon className="w-4 h-4" />
          ) : (
            <ArrowUpIcon className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  )
}

export function AutocompleteInput({ onSubmit, disabled, editValue }: AutocompleteInputProps) {
  const { user: authUser } = useAuth()
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [items, setItems] = useState<AutocompleteItem[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loading, setLoading] = useState(false)

  // Cached user list for @mention autocomplete
  const [mentionUsers, setMentionUsers] = useState<Pick<UserPermissions, 'user_id' | 'email'>[]>([])
  useEffect(() => {
    listAllPermissions()
      .then((perms) => {
        const users = perms.map((p) => {
          // Enrich with current user's email from auth (permissions API doesn't return email)
          const email = (authUser && p.user_id === authUser.uid) ? (authUser.email ?? p.email) : p.email
          return { user_id: p.user_id, email }
        })
        setMentionUsers(users)
      })
      .catch(() => {}) // non-admin users may not have access
  }, [])

  // Command history state - initialized from localStorage
  const [history, setHistory] = useState<string[]>(() => loadHistory())
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [savedQuery, setSavedQuery] = useState('')  // Save current input when navigating history

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Populate input when editValue changes
  useEffect(() => {
    if (editValue != null) {
      setQuery(editValue)
      setTimeout(() => textareaRef.current?.focus(), 0)
    }
  }, [editValue])

  // Check if session is busy (will queue instead of send)
  const { session, status, executionPhase } = useSessionContext()
  const isBusy = status === 'planning' || status === 'executing' || status === 'awaiting_approval' ||
    executionPhase !== 'idle'

  const isDisabled = disabled

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [query])

  // Handle input changes with debounced server requests
  const handleChange = useCallback(
    (value: string, cursorPos?: number) => {
      setQuery(value)

      // Reset history navigation when user types manually
      if (historyIndex !== -1) {
        setHistoryIndex(-1)
        setSavedQuery('')
      }

      // Clear pending debounce
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
        debounceRef.current = null
      }

      const cursor = cursorPos ?? textareaRef.current?.selectionStart ?? value.length
      const parsed = parseInput(value, cursor)

      // No completion context
      if (parsed.context === 'none') {
        setIsOpen(false)
        setItems([])
        return
      }

      // Get client-side completions immediately
      const clientItems = getClientCompletions(parsed, mentionUsers)

      // For command, scope, and mention contexts, use only client-side
      if (parsed.context === 'command' || parsed.context === 'scope' || parsed.context === 'mention') {
        setItems(clientItems)
        setSelectedIndex(0)
        setIsOpen(clientItems.length > 0)
        return
      }

      // For table and entity contexts, fetch from server
      if (parsed.context === 'table' || parsed.context === 'entity') {
        // Show client items immediately (for /summarize)
        if (clientItems.length > 0) {
          setItems(clientItems)
          setSelectedIndex(0)
          setIsOpen(true)
        }

        // Debounce server request
        setLoading(true)
        debounceRef.current = setTimeout(() => {
          if (!session) {
            setLoading(false)
            return
          }
          apolloClient.mutate({
            mutation: REQUEST_AUTOCOMPLETE,
            variables: {
              sessionId: session.session_id,
              context: parsed.context,
              prefix: parsed.prefix,
              parent: parsed.parent,
            },
          }).then(({ data }) => {
            const serverItems: AutocompleteItem[] = data?.requestAutocomplete?.items ?? []
            setLoading(false)
            // Merge client items with server items
            const merged = [...clientItems, ...serverItems]
            setItems(merged)
            setSelectedIndex(0)
            setIsOpen(merged.length > 0)
          }).catch(() => {
            setLoading(false)
          })
        }, 150)
      }
    },
    [historyIndex, mentionUsers]
  )

  const handleSubmit = () => {
    const trimmed = query.trim()
    if (trimmed && !isDisabled) {
      // Add to history (avoid duplicates at the end)
      if (history.length === 0 || history[history.length - 1] !== trimmed) {
        const newHistory = [...history, trimmed]
        setHistory(newHistory)
        saveHistory(newHistory)  // Persist to localStorage
      }
      // Reset history navigation state
      setHistoryIndex(-1)
      setSavedQuery('')

      onSubmit(trimmed)
      setQuery('')
      setIsOpen(false)
    }
  }

  const acceptCompletion = (item: AutocompleteItem) => {
    const cursor = textareaRef.current?.selectionStart ?? query.length
    const parsed = parseInput(query, cursor)

    let newValue: string
    if (parsed.context === 'mention') {
      // Replace @prefix at cursor with selected value
      const beforeCursor = query.slice(0, cursor)
      const atIdx = beforeCursor.lastIndexOf('@')
      const after = query.slice(cursor)
      newValue = query.slice(0, atIdx) + item.value + ' ' + after
    } else if (parsed.context === 'command') {
      // Replace the entire command
      newValue = item.value + ' '
    } else {
      // Replace just the argument part
      const spaceIndex = query.indexOf(' ')
      const commandPart = query.substring(0, spaceIndex + 1)
      newValue = commandPart + item.value + ' '
    }

    setQuery(newValue)
    setIsOpen(false)
    textareaRef.current?.focus()
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (isOpen && items.length > 0) {
      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault()
          setSelectedIndex((prev) => (prev + 1) % items.length)
          return

        case 'ArrowUp':
          e.preventDefault()
          setSelectedIndex((prev) => (prev - 1 + items.length) % items.length)
          return

        case 'Tab':
        case 'Enter':
          if (items[selectedIndex]) {
            e.preventDefault()
            acceptCompletion(items[selectedIndex])
            return
          }
          break

        case 'Escape':
          e.preventDefault()
          setIsOpen(false)
          return
      }
    }

    // History navigation when autocomplete is NOT open
    if (!isOpen && history.length > 0) {
      if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (historyIndex === -1) {
          // Starting to navigate history, save current input
          setSavedQuery(query)
          setHistoryIndex(history.length - 1)
          setQuery(history[history.length - 1])
        } else if (historyIndex > 0) {
          // Go further back in history
          setHistoryIndex(historyIndex - 1)
          setQuery(history[historyIndex - 1])
        }
        return
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault()
        if (historyIndex !== -1) {
          if (historyIndex < history.length - 1) {
            // Go forward in history
            setHistoryIndex(historyIndex + 1)
            setQuery(history[historyIndex + 1])
          } else {
            // Reached the end, restore saved query
            setHistoryIndex(-1)
            setQuery(savedQuery)
          }
        }
        return
      }
    }

    if (e.key === 'Enter' && !e.shiftKey && !isOpen) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const insertAt = useCallback(() => {
    const ta = textareaRef.current
    if (ta) {
      const start = ta.selectionStart
      const before = query.slice(0, start)
      const after = query.slice(ta.selectionEnd)
      const newValue = before + '@' + after
      const newCursor = start + 1
      handleChange(newValue, newCursor)
      setTimeout(() => {
        ta.focus()
        ta.setSelectionRange(newCursor, newCursor)
      }, 0)
    }
  }, [query, handleChange])

  const handleBlur = () => {
    // Small delay to allow click on dropdown item
    setTimeout(() => setIsOpen(false), 150)
  }

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  return (
    <div className="px-4 pb-4 pt-2">
      <div className="max-w-3xl mx-auto">
      {/* Queue indicator when busy */}
      {isBusy && query.trim() && (
        <div className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400 mb-2">
          <QueueListIcon className="w-3.5 h-3.5" />
          <span>Message will be queued and sent when current task completes</span>
        </div>
      )}
      <div className="rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        {/* Autocomplete dropdown (positioned above) */}
        <div className="relative">
          <AutocompleteDropdown
            items={items}
            selectedIndex={selectedIndex}
            onSelect={acceptCompletion}
            loading={loading}
            visible={isOpen}
          />
        </div>
        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => handleChange(e.target.value, e.target.selectionStart ?? undefined)}
          onKeyDown={handleKeyDown}
          onBlur={handleBlur}
          onFocus={() => {
            const parsed = parseInput(query)
            if (parsed.context !== 'none') {
              handleChange(query)
            }
          }}
          placeholder="Ask a question about your data..."
          disabled={isDisabled}
          rows={1}
          className="w-full resize-none bg-transparent px-4 pt-3 pb-2 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed scrollbar-hide border-0 focus:ring-0"
        />
        {/* Toolbar row: all actions on one line */}
        <InputToolbar
          query={query}
          setQuery={(v) => { setQuery(v); setIsOpen(false) }}
          textareaRef={textareaRef}
          isBusy={isBusy}
          isDisabled={isDisabled}
          onSubmit={handleSubmit}
          onInsertAt={insertAt}
        />
      </div>
      </div>
    </div>
  )
}
