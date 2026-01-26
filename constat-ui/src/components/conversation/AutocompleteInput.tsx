// Autocomplete-enabled query input wrapper

import { useState, useRef, useEffect, useCallback, KeyboardEvent } from 'react'
import { PaperAirplaneIcon, QueueListIcon } from '@heroicons/react/24/solid'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { AutocompleteDropdown, AutocompleteItem } from './AutocompleteDropdown'
import { wsManager } from '@/api/websocket'
import { useSessionStore } from '@/store/sessionStore'
import {
  filterCommands,
  getCommandByName,
  DISCOVER_SCOPES,
  SUMMARIZE_TARGETS,
} from '@/data/commands'

interface AutocompleteInputProps {
  onSubmit: (query: string) => void
  disabled?: boolean
}

type CompletionContext = 'command' | 'table' | 'entity' | 'scope' | 'none'

interface ParsedInput {
  context: CompletionContext
  command?: string
  prefix: string
  parent?: string
}

function parseInput(value: string): ParsedInput {
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

function getClientCompletions(parsed: ParsedInput): AutocompleteItem[] {
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

export function AutocompleteInput({ onSubmit, disabled }: AutocompleteInputProps) {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [items, setItems] = useState<AutocompleteItem[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [loading, setLoading] = useState(false)

  // Command history state - initialized from localStorage
  const [history, setHistory] = useState<string[]>(() => loadHistory())
  const [historyIndex, setHistoryIndex] = useState(-1)
  const [savedQuery, setSavedQuery] = useState('')  // Save current input when navigating history

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Check if session is busy (will queue instead of send)
  const { status, executionPhase } = useSessionStore()
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
    (value: string) => {
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

      const parsed = parseInput(value)

      // No completion context
      if (parsed.context === 'none') {
        setIsOpen(false)
        setItems([])
        return
      }

      // Get client-side completions immediately
      const clientItems = getClientCompletions(parsed)

      // For command and scope contexts, use only client-side
      if (parsed.context === 'command' || parsed.context === 'scope') {
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
          wsManager.requestAutocomplete(
            parsed.context as 'table' | 'entity',
            parsed.prefix,
            (serverItems) => {
              setLoading(false)
              // Merge client items with server items
              const merged = [...clientItems, ...serverItems]
              setItems(merged)
              setSelectedIndex(0)
              setIsOpen(merged.length > 0)
            },
            parsed.parent
          )
        }, 150)
      }
    },
    [historyIndex]
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
    const parsed = parseInput(query)

    let newValue: string
    if (parsed.context === 'command') {
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
    <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
      {/* Queue indicator when busy */}
      {isBusy && query.trim() && (
        <div className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400 mb-2">
          <QueueListIcon className="w-3.5 h-3.5" />
          <span>Message will be queued and sent when current task completes</span>
        </div>
      )}
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <AutocompleteDropdown
            items={items}
            selectedIndex={selectedIndex}
            onSelect={acceptCompletion}
            loading={loading}
            visible={isOpen}
          />
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            onFocus={() => {
              // Re-trigger autocomplete on focus if we have a valid context
              const parsed = parseInput(query)
              if (parsed.context !== 'none') {
                handleChange(query)
              }
            }}
            placeholder="Ask a question about your data..."
            disabled={isDisabled}
            rows={1}
            className="w-full resize-none rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-4 py-3 pr-20 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed scrollbar-hide"
          />
          {/* Clear button - only show when there's text */}
          {query && (
            <button
              onClick={() => {
                setQuery('')
                setIsOpen(false)
                textareaRef.current?.focus()
              }}
              className="absolute right-12 top-1/2 -translate-y-1/2 p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Clear"
            >
              <XMarkIcon className="w-4 h-4" />
            </button>
          )}
          <button
            onClick={handleSubmit}
            disabled={isDisabled || !query.trim()}
            className={`absolute p-2 rounded-md text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              isBusy && query.trim()
                ? 'bg-amber-500 hover:bg-amber-600'
                : 'bg-primary-600 hover:bg-primary-700'
            }`}
            style={{ right: '7px', top: '50%', transform: 'translateY(calc(-50% - 3px))' }}
            title={isBusy && query.trim() ? 'Queue message' : 'Send message'}
          >
            {isBusy && query.trim() ? (
              <QueueListIcon className="w-4 h-4" />
            ) : (
              <PaperAirplaneIcon className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
