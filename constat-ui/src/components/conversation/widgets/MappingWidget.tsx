// MappingWidget - Two-column connector for creating left-right mappings

import { useState, useCallback } from 'react'
import { XMarkIcon, PlusIcon } from '@heroicons/react/24/outline'

interface Mapping {
  left: string
  right: string
}

interface MappingWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: { mappings: Mapping[] }
  onAnswer: (freeform: string, structured: { mappings: Mapping[] }) => void
}

export function MappingWidget({ config, structuredValue, onAnswer }: MappingWidgetProps) {
  const leftItems = (config.left as string[]) || []
  const rightItems = (config.right as string[]) || []
  const leftLabel = (config.leftLabel as string) || 'Source'
  const rightLabel = (config.rightLabel as string) || 'Target'

  const [mappings, setMappings] = useState<Mapping[]>(() => {
    if (structuredValue?.mappings?.length) return structuredValue.mappings
    return []
  })

  const [selectedLeft, setSelectedLeft] = useState<string | null>(null)

  const emitAnswer = useCallback((newMappings: Mapping[]) => {
    const freeform = newMappings.length > 0
      ? newMappings.map(m => `${m.left} -> ${m.right}`).join(', ')
      : 'No mappings defined'
    onAnswer(freeform, { mappings: newMappings })
  }, [onAnswer])

  const addMapping = (left: string, right: string) => {
    // Don't duplicate
    if (mappings.some(m => m.left === left && m.right === right)) return
    const newMappings = [...mappings, { left, right }]
    setMappings(newMappings)
    setSelectedLeft(null)
    emitAnswer(newMappings)
  }

  const removeMapping = (index: number) => {
    const newMappings = mappings.filter((_, i) => i !== index)
    setMappings(newMappings)
    emitAnswer(newMappings)
  }

  const handleLeftClick = (item: string) => {
    setSelectedLeft(selectedLeft === item ? null : item)
  }

  const handleRightClick = (item: string) => {
    if (selectedLeft) {
      addMapping(selectedLeft, item)
    }
  }

  // Track which items are already mapped
  const mappedLeft = new Set(mappings.map(m => m.left))
  const mappedRight = new Set(mappings.map(m => m.right))

  return (
    <div className="space-y-3">
      {/* Two-column selection */}
      <div className="grid grid-cols-2 gap-4">
        {/* Left column */}
        <div>
          <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
            {leftLabel}
          </div>
          <div className="space-y-1 border border-gray-200 dark:border-gray-700 rounded-lg p-2 max-h-48 overflow-y-auto">
            {leftItems.map(item => (
              <button
                key={item}
                onClick={() => handleLeftClick(item)}
                className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${
                  selectedLeft === item
                    ? 'bg-primary-100 dark:bg-primary-900/40 text-primary-700 dark:text-primary-300 ring-1 ring-primary-500'
                    : mappedLeft.has(item)
                    ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50'
                }`}
              >
                {item}
                {mappedLeft.has(item) && (
                  <span className="ml-1 text-xs text-green-500">&#10003;</span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Right column */}
        <div>
          <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
            {rightLabel}
          </div>
          <div className="space-y-1 border border-gray-200 dark:border-gray-700 rounded-lg p-2 max-h-48 overflow-y-auto">
            {rightItems.map(item => (
              <button
                key={item}
                onClick={() => handleRightClick(item)}
                disabled={!selectedLeft}
                className={`w-full text-left px-3 py-2 text-sm rounded-lg transition-colors ${
                  !selectedLeft
                    ? 'text-gray-400 dark:text-gray-500 cursor-not-allowed'
                    : mappedRight.has(item)
                    ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 hover:bg-green-100 dark:hover:bg-green-900/30'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-primary-50 dark:hover:bg-primary-900/30'
                }`}
              >
                {item}
                {mappedRight.has(item) && (
                  <span className="ml-1 text-xs text-green-500">&#10003;</span>
                )}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Instructions */}
      {selectedLeft ? (
        <div className="text-xs text-primary-600 dark:text-primary-400">
          Now click a {rightLabel.toLowerCase()} item to map &quot;{selectedLeft}&quot; to it
        </div>
      ) : (
        <div className="text-xs text-gray-400 dark:text-gray-500">
          Click a {leftLabel.toLowerCase()} item, then click a {rightLabel.toLowerCase()} item to create a mapping
        </div>
      )}

      {/* Current mappings */}
      {mappings.length > 0 && (
        <div className="space-y-1">
          <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
            Mappings ({mappings.length})
          </div>
          {mappings.map((mapping, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg text-sm"
            >
              <span className="text-gray-700 dark:text-gray-300">{mapping.left}</span>
              <PlusIcon className="w-3 h-3 text-gray-400 rotate-45" />
              <span className="text-gray-700 dark:text-gray-300">{mapping.right}</span>
              <button
                onClick={() => removeMapping(index)}
                className="ml-auto p-0.5 text-gray-400 hover:text-red-500 transition-colors"
                title="Remove mapping"
              >
                <XMarkIcon className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
