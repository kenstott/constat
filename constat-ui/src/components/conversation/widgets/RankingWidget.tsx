// Copyright (c) 2025 Kenneth Stott
// Canary: a3948323-940a-427f-a316-4d0179e442dd
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// RankingWidget - Drag-to-reorder list for ranking items

import { useState, useRef, useCallback } from 'react'
import { Bars3Icon } from '@heroicons/react/24/outline'

interface RankingWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: { ranked: string[] } | { tiers: Record<string, string[]> }
  onAnswer: (freeform: string, structured: { ranked: string[] } | { tiers: Record<string, string[]> }) => void
}

export function RankingWidget({ config, structuredValue, onAnswer }: RankingWidgetProps) {
  const items = (config.items as string[]) || []
  const tiers = config.tiers as string[] | undefined

  // Initialize from structuredValue or original order
  const [ranked, setRanked] = useState<string[]>(() => {
    if (structuredValue && 'ranked' in structuredValue) return structuredValue.ranked
    return [...items]
  })

  const [tierAssignment, setTierAssignment] = useState<Record<string, string[]>>(() => {
    if (tiers && structuredValue && 'tiers' in structuredValue) return structuredValue.tiers
    if (tiers) {
      // Put all items in first tier initially
      const assignment: Record<string, string[]> = {}
      tiers.forEach((t, i) => {
        assignment[t] = i === 0 ? [...items] : []
      })
      return assignment
    }
    return {}
  })

  const dragItem = useRef<number | null>(null)
  const dragOverItem = useRef<number | null>(null)

  const emitAnswer = useCallback((newRanked: string[]) => {
    if (tiers) {
      // Emit tier-based answer
      const freeform = tiers.map(t => `${t}: ${tierAssignment[t]?.join(', ') || 'none'}`).join('; ')
      onAnswer(freeform, { tiers: tierAssignment })
    } else {
      const freeform = newRanked.map((item, i) => `${i + 1}. ${item}`).join(', ')
      onAnswer(freeform, { ranked: newRanked })
    }
  }, [tiers, tierAssignment, onAnswer])

  const handleDragStart = (index: number) => {
    dragItem.current = index
  }

  const handleDragEnter = (index: number) => {
    dragOverItem.current = index
  }

  const handleDragEnd = () => {
    if (dragItem.current === null || dragOverItem.current === null) return
    const newRanked = [...ranked]
    const draggedItem = newRanked[dragItem.current]
    newRanked.splice(dragItem.current, 1)
    newRanked.splice(dragOverItem.current, 0, draggedItem)
    dragItem.current = null
    dragOverItem.current = null
    setRanked(newRanked)
    emitAnswer(newRanked)
  }

  const moveItem = (fromIndex: number, toIndex: number) => {
    if (toIndex < 0 || toIndex >= ranked.length) return
    const newRanked = [...ranked]
    const [moved] = newRanked.splice(fromIndex, 1)
    newRanked.splice(toIndex, 0, moved)
    setRanked(newRanked)
    emitAnswer(newRanked)
  }

  if (tiers) {
    // Tier-based ranking
    return (
      <div className="space-y-3">
        {tiers.map(tier => (
          <div key={tier} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
            <div className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
              {tier}
            </div>
            <div className="min-h-[2rem] space-y-1">
              {(tierAssignment[tier] || []).map(item => (
                <div
                  key={item}
                  className="flex items-center gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg text-sm text-gray-700 dark:text-gray-300"
                >
                  <span className="flex-1">{item}</span>
                  <select
                    value={tier}
                    onChange={(e) => {
                      const newTier = e.target.value
                      const newAssignment = { ...tierAssignment }
                      newAssignment[tier] = (newAssignment[tier] || []).filter(i => i !== item)
                      newAssignment[newTier] = [...(newAssignment[newTier] || []), item]
                      setTierAssignment(newAssignment)
                      const freeform = tiers.map(t => `${t}: ${newAssignment[t]?.join(', ') || 'none'}`).join('; ')
                      onAnswer(freeform, { tiers: newAssignment })
                    }}
                    className="text-xs px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300"
                  >
                    {tiers.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              ))}
              {(tierAssignment[tier] || []).length === 0 && (
                <div className="text-xs text-gray-400 dark:text-gray-500 italic px-3 py-2">
                  No items
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    )
  }

  // Simple ranked list with drag-to-reorder
  return (
    <div className="space-y-1 border border-gray-200 dark:border-gray-700 rounded-lg p-2">
      {ranked.map((item, index) => (
        <div
          key={item}
          draggable
          onDragStart={() => handleDragStart(index)}
          onDragEnter={() => handleDragEnter(index)}
          onDragEnd={handleDragEnd}
          onDragOver={(e) => e.preventDefault()}
          className="flex items-center gap-3 px-3 py-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 cursor-grab active:cursor-grabbing hover:border-primary-300 dark:hover:border-primary-600 transition-colors"
        >
          <Bars3Icon className="w-4 h-4 text-gray-400 flex-shrink-0" />
          <span className="text-xs font-medium text-gray-400 dark:text-gray-500 w-5 text-center">
            {index + 1}
          </span>
          <span className="text-sm text-gray-700 dark:text-gray-300 flex-1">{item}</span>
          <div className="flex gap-1">
            <button
              onClick={() => moveItem(index, index - 1)}
              disabled={index === 0}
              className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Move up"
            >
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            </button>
            <button
              onClick={() => moveItem(index, index + 1)}
              disabled={index === ranked.length - 1}
              className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 disabled:opacity-30 disabled:cursor-not-allowed"
              title="Move down"
            >
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
