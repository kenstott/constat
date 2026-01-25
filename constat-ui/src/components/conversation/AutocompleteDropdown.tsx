// Autocomplete dropdown component

import { useRef, useEffect } from 'react'

export interface AutocompleteItem {
  label: string
  value: string
  description?: string
  category?: string
}

interface AutocompleteDropdownProps {
  items: AutocompleteItem[]
  selectedIndex: number
  onSelect: (item: AutocompleteItem) => void
  loading?: boolean
  visible: boolean
}

export function AutocompleteDropdown({
  items,
  selectedIndex,
  onSelect,
  loading,
  visible,
}: AutocompleteDropdownProps) {
  const listRef = useRef<HTMLDivElement>(null)
  const selectedRef = useRef<HTMLDivElement>(null)

  // Scroll selected item into view
  useEffect(() => {
    if (selectedRef.current && listRef.current) {
      const list = listRef.current
      const selected = selectedRef.current
      const listRect = list.getBoundingClientRect()
      const selectedRect = selected.getBoundingClientRect()

      if (selectedRect.top < listRect.top) {
        selected.scrollIntoView({ block: 'start', behavior: 'smooth' })
      } else if (selectedRect.bottom > listRect.bottom) {
        selected.scrollIntoView({ block: 'end', behavior: 'smooth' })
      }
    }
  }, [selectedIndex])

  if (!visible) {
    return null
  }

  return (
    <div
      className="absolute bottom-full left-0 right-0 mb-1 max-h-64 overflow-y-auto rounded-lg border border-gray-200 bg-white shadow-lg dark:border-gray-700 dark:bg-gray-800 z-50"
      ref={listRef}
    >
      {loading && (
        <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400">
          Loading...
        </div>
      )}

      {!loading && items.length === 0 && (
        <div className="px-4 py-2 text-sm text-gray-500 dark:text-gray-400">
          No matches found
        </div>
      )}

      {!loading && items.length > 0 && (
        <div className="py-1">
          {items.map((item, index) => (
            <div
              key={`${item.value}-${index}`}
              ref={index === selectedIndex ? selectedRef : null}
              className={`px-4 py-2 cursor-pointer flex flex-col ${
                index === selectedIndex
                  ? 'bg-primary-100 dark:bg-primary-900/30'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
              }`}
              onClick={() => onSelect(item)}
              onMouseDown={(e) => e.preventDefault()} // Prevent blur before click
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-sm font-medium ${
                    index === selectedIndex
                      ? 'text-primary-700 dark:text-primary-300'
                      : 'text-gray-900 dark:text-gray-100'
                  }`}
                >
                  {item.label}
                </span>
                {item.category && (
                  <span className="text-xs text-gray-400 dark:text-gray-500 ml-2">
                    {item.category}
                  </span>
                )}
              </div>
              {item.description && (
                <span className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  {item.description}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="border-t border-gray-200 dark:border-gray-700 px-3 py-1.5 text-xs text-gray-400 dark:text-gray-500 flex gap-4">
        <span>
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-[10px]">
            ↑↓
          </kbd>{' '}
          navigate
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-[10px]">
            Enter
          </kbd>{' '}
          select
        </span>
        <span>
          <kbd className="px-1 py-0.5 bg-gray-100 dark:bg-gray-700 rounded text-[10px]">
            Esc
          </kbd>{' '}
          close
        </span>
      </div>
    </div>
  )
}
