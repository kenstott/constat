// Copyright (c) 2025 Kenneth Stott
// Canary: 8d6bb15b-26f6-419a-9871-1f70a0ec2f1e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState } from 'react'
import { useMutation } from '@apollo/client'
import { UPDATE_DOCUMENT } from '@/graphql/operations/sources'
import type { DocumentSourceInfo } from '@/types/api'

interface Props {
  doc: DocumentSourceInfo
  onSuccess: () => void
  onCancel: () => void
}

export function EditDocumentModal({ doc, onSuccess, onCancel }: Props) {
  const [newName, setNewName] = useState(doc.name)
  const [description, setDescription] = useState(doc.description ?? '')
  const [uri, setUri] = useState(doc.path ?? '')
  const [followLinks, setFollowLinks] = useState(false)
  const [maxDepth, setMaxDepth] = useState('3')
  const [maxDocuments, setMaxDocuments] = useState('50')
  const [error, setError] = useState<string | null>(null)

  const [updateDocument, { loading }] = useMutation(UPDATE_DOCUMENT)

  const isWeb = uri.startsWith('http://') || uri.startsWith('https://')

  function canSubmit(): boolean {
    return uri.length > 0
  }

  async function handleSubmit() {
    if (!canSubmit()) return
    setError(null)
    const input: Record<string, unknown> = {
      name: doc.name,
      description: description || undefined,
      uri: uri || undefined,
    }
    if (newName.trim() && newName.trim() !== doc.name) input.new_name = newName.trim()
    if (isWeb) {
      input.follow_links = followLinks
      input.max_depth = parseInt(maxDepth, 10) || 3
      input.max_documents = parseInt(maxDocuments, 10) || 50
    }
    try {
      await updateDocument({ variables: { input } })
      onSuccess()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update document')
    }
  }

  const input = 'w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const half = 'flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const lbl = 'text-xs text-gray-500 dark:text-gray-400 mb-0.5'

  return (
    <div className="space-y-3">
      <div>
        <p className={lbl}>Name</p>
        <input type="text" value={newName} onChange={(e) => setNewName(e.target.value)} className={input} />
      </div>

      <div>
        <p className={lbl}>Description</p>
        <textarea
          rows={2}
          placeholder="Description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className={input + ' resize-y'}
        />
      </div>

      <div>
        <p className={lbl}>URI / Path</p>
        <input type="text" placeholder="https://example.com or /path/to/file.pdf" value={uri} onChange={(e) => setUri(e.target.value)} className={input} />
      </div>

      {isWeb && (
        <>
          <div className="flex items-center gap-2">
            <input
              id="follow-links"
              type="checkbox"
              checked={followLinks}
              onChange={(e) => setFollowLinks(e.target.checked)}
              className="w-4 h-4 text-primary-600 border-gray-300 rounded"
            />
            <label htmlFor="follow-links" className="text-xs text-gray-600 dark:text-gray-400">Follow links</label>
          </div>

          {followLinks && (
            <div className="flex gap-2">
              <div className="flex-1">
                <p className={lbl}>Max depth</p>
                <input type="number" min="1" max="10" value={maxDepth} onChange={(e) => setMaxDepth(e.target.value)} className={half} />
              </div>
              <div className="flex-1">
                <p className={lbl}>Max documents</p>
                <input type="number" min="1" max="500" value={maxDocuments} onChange={(e) => setMaxDocuments(e.target.value)} className={half} />
              </div>
            </div>
          )}
        </>
      )}

      {error && <p className="text-xs text-red-500 dark:text-red-400">{error}</p>}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={loading || !canSubmit()}
          className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />}
          {loading ? 'Saving...' : 'Save'}
        </button>
      </div>
    </div>
  )
}
