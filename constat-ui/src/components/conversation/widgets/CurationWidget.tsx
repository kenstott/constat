// Copyright (c) 2025 Kenneth Stott
// Canary: 859aaf5f-0192-430e-b9d3-fd458c601262
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// CurationWidget - Filterable checklist for selecting/removing items from a list

import { useState, useMemo } from 'react'
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline'

interface CurationWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: { kept: string[]; removed: string[] }
  onAnswer: (freeform: string, structured: { kept: string[]; removed: string[] }) => void
}

export function CurationWidget({ config, structuredValue, onAnswer }: CurationWidgetProps) {
  const items = (config.items as string[]) || []
  const groupBy = config.groupBy as string | undefined

  // Initialize selected set from structuredValue or all items
  const [selected, setSelected] = useState<Set<string>>(() => {
    if (structuredValue?.kept) return new Set(structuredValue.kept)
    return new Set(items)
  })
  const [search, setSearch] = useState('')

  // Group items if groupBy is specified
  const groupedItems = useMemo(() => {
    if (!groupBy) return { '': items }
    const groups: Record<string, string[]> = {}
    for (const item of items) {
      // Simple groupBy: split on separator and use first part as group
      const parts = item.split(groupBy)
      const group = parts.length > 1 ? parts[0].trim() : ''
      if (!groups[group]) groups[group] = []
      groups[group].push(item)
    }
    return groups
  }, [items, groupBy])

  const filteredItems = useMemo(() => {
    if (!search) return items
    const lower = search.toLowerCase()
    return items.filter(item => item.toLowerCase().includes(lower))
  }, [items, search])

  const emitAnswer = (newSelected: Set<string>) => {
    const kept = items.filter(i => newSelected.has(i))
    const removed = items.filter(i => !newSelected.has(i))
    const freeform = `Selected: ${kept.join(', ')} (${kept.length} of ${items.length})`
    onAnswer(freeform, { kept, removed })
  }

  const toggleItem = (item: string) => {
    const next = new Set(selected)
    if (next.has(item)) next.delete(item)
    else next.add(item)
    setSelected(next)
    emitAnswer(next)
  }

  const selectAll = () => {
    const next = new Set(items)
    setSelected(next)
    emitAnswer(next)
  }

  const deselectAll = () => {
    const next = new Set<string>()
    setSelected(next)
    emitAnswer(next)
  }

  const renderItems = (itemList: string[]) =>
    itemList.map(item => (
      <label
        key={item}
        className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
      >
        <input
          type="checkbox"
          checked={selected.has(item)}
          onChange={() => toggleItem(item)}
          className="w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
        />
        <span className="text-sm text-gray-700 dark:text-gray-300">{item}</span>
      </label>
    ))

  return (
    <div className="space-y-3">
      {/* Search + select/deselect all */}
      <div className="flex items-center gap-2">
        <div className="relative flex-1">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            onKeyUp={(e) => e.stopPropagation()}
            placeholder="Filter items..."
            className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
          />
        </div>
        <button
          onClick={selectAll}
          className="px-2 py-1.5 text-xs font-medium text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/30 rounded transition-colors"
        >
          All
        </button>
        <button
          onClick={deselectAll}
          className="px-2 py-1.5 text-xs font-medium text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
        >
          None
        </button>
      </div>

      {/* Counter */}
      <div className="text-xs text-gray-500 dark:text-gray-400">
        {selected.size} of {items.length} selected
      </div>

      {/* Item list */}
      <div className="max-h-64 overflow-y-auto space-y-1 border border-gray-200 dark:border-gray-700 rounded-lg p-2">
        {groupBy ? (
          Object.entries(groupedItems).map(([group, groupItems]) => {
            const visibleItems = groupItems.filter(i => filteredItems.includes(i))
            if (visibleItems.length === 0) return null
            return (
              <div key={group}>
                {group && (
                  <div className="px-3 py-1 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    {group}
                  </div>
                )}
                {renderItems(visibleItems)}
              </div>
            )
          })
        ) : (
          renderItems(filteredItems)
        )}
      </div>
    </div>
  )
}
