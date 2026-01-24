// Query Input component

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'

interface QueryInputProps {
  onSubmit: (query: string) => void
  disabled?: boolean
}

export function QueryInput({ onSubmit, disabled }: QueryInputProps) {
  const [query, setQuery] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isDisabled = disabled

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }, [query])

  const handleSubmit = () => {
    const trimmed = query.trim()
    if (trimmed && !isDisabled) {
      onSubmit(trimmed)
      setQuery('')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-white dark:bg-gray-800">
      <div className="flex gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your data..."
            disabled={isDisabled}
            rows={1}
            className="w-full resize-none rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-4 py-3 pr-12 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={handleSubmit}
            disabled={isDisabled || !query.trim()}
            className="absolute right-2 bottom-2 p-2 rounded-md text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <PaperAirplaneIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-400 dark:text-gray-500">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  )
}