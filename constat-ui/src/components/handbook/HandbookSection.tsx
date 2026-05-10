// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState, useCallback } from 'react'
import { useMutation } from '@apollo/client'
import {
  ChevronRightIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { UPDATE_HANDBOOK_ENTRY } from '@/graphql/operations/handbook'

export interface HandbookEntry {
  key: string
  display: string
  metadata?: Record<string, unknown> | null
  editable: boolean
}

export interface HandbookSectionData {
  title: string
  content: HandbookEntry[]
  lastUpdated?: string | null
}

interface HandbookSectionProps {
  section: HandbookSectionData
  sectionKey: string
  sessionId: string
  defaultExpanded?: boolean
}

interface EditState {
  key: string
  value: string
  reason: string
}

export function HandbookSection({
  section,
  sectionKey,
  sessionId,
  defaultExpanded = false,
}: HandbookSectionProps) {
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [editState, setEditState] = useState<EditState | null>(null)
  const [updateEntry, { loading: saving }] = useMutation(UPDATE_HANDBOOK_ENTRY)

  const handleEdit = useCallback((entry: HandbookEntry) => {
    setEditState({ key: entry.key, value: entry.display, reason: '' })
  }, [])

  const handleCancel = useCallback(() => {
    setEditState(null)
  }, [])

  const handleSave = useCallback(async () => {
    if (!editState) return
    await updateEntry({
      variables: {
        sessionId,
        section: sectionKey,
        key: editState.key,
        fieldName: 'display',
        newValue: editState.value,
        reason: editState.reason || undefined,
      },
      refetchQueries: ['Handbook'],
    })
    setEditState(null)
  }, [editState, sessionId, sectionKey, updateEntry])

  const entryCount = section.content.length

  return (
    <div id={`section-${sectionKey}`} className="border border-gray-200 dark:border-gray-700 rounded-lg">
      {/* Collapsible header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg transition-colors"
        aria-expanded={expanded}
      >
        <ChevronRightIcon
          className={`h-4 w-4 text-gray-500 transition-transform ${expanded ? 'rotate-90' : ''}`}
        />
        <span className="font-medium text-gray-900 dark:text-gray-100">{section.title}</span>
        <span className="ml-auto text-xs text-gray-500 dark:text-gray-400">
          {entryCount} {entryCount === 1 ? 'entry' : 'entries'}
        </span>
      </button>

      {/* Content */}
      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          {entryCount === 0 && (
            <p className="text-sm text-gray-400 dark:text-gray-500 italic">No entries in this section.</p>
          )}

          {section.content.map((entry) => {
            const isEditing = editState?.key === entry.key

            return (
              <div
                key={entry.key}
                className="flex items-start gap-3 p-2 rounded-md bg-gray-50 dark:bg-gray-800/50"
              >
                {isEditing ? (
                  <div className="flex-1 space-y-2">
                    <textarea
                      value={editState.value}
                      onChange={(e) => setEditState({ ...editState, value: e.target.value })}
                      className="w-full px-2 py-1 text-sm border border-blue-300 dark:border-blue-600 rounded bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-blue-500"
                      rows={3}
                    />
                    <input
                      type="text"
                      placeholder="Reason for change (optional)"
                      value={editState.reason}
                      onChange={(e) => setEditState({ ...editState, reason: e.target.value })}
                      className="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900 text-gray-700 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    <div className="flex gap-2">
                      <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                      >
                        <CheckIcon className="h-3 w-3" />
                        {saving ? 'Saving...' : 'Save'}
                      </button>
                      <button
                        onClick={handleCancel}
                        disabled={saving}
                        className="flex items-center gap-1 px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                      >
                        <XMarkIcon className="h-3 w-3" />
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-medium text-gray-500 dark:text-gray-400">
                        {entry.key}
                      </div>
                      <div className="text-sm text-gray-900 dark:text-gray-100 whitespace-pre-wrap">
                        {entry.display}
                      </div>
                    </div>
                    {entry.editable && (
                      <button
                        onClick={() => handleEdit(entry)}
                        className="shrink-0 p-1 text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors"
                        title="Edit entry"
                        aria-label={`Edit ${entry.key}`}
                      >
                        <PencilIcon className="h-4 w-4" />
                      </button>
                    )}
                  </>
                )}
              </div>
            )
          })}

          {section.lastUpdated && (
            <p className="text-xs text-gray-400 dark:text-gray-500 pt-1">
              Last updated: {new Date(section.lastUpdated).toLocaleString()}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
