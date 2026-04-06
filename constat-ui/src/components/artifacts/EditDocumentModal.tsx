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
import { useMutation, useLazyQuery } from '@apollo/client'
import { UPDATE_DOCUMENT, VALIDATE_URI } from '@/graphql/operations/sources'
import type { DocumentSourceInfo } from '@/types/api'

interface Props {
  doc: DocumentSourceInfo
  onSuccess: () => void
  onCancel: () => void
}

interface FieldErrors {
  name?: string
  description?: string
  uri?: string
}

export const URL_SCHEMES = ['http://', 'https://', 'ftp://', 'sftp://', 's3://', 's3a://', 'file://']

export function detectScheme(uri: string): string | null {
  const lower = uri.toLowerCase()
  return URL_SCHEMES.find(s => lower.startsWith(s)) ?? null
}

export function isValidUri(uri: string): boolean {
  const trimmed = uri.trim()
  if (!trimmed) return false
  const scheme = detectScheme(trimmed)
  if (scheme === 'http://' || scheme === 'https://') {
    try { new URL(trimmed); return true } catch { return false }
  }
  if (scheme === 's3://' || scheme === 's3a://') {
    // must have bucket: s3://bucket/key
    return trimmed.replace(/^s3a?:\/\//i, '').split('/')[0].length > 0
  }
  if (scheme === 'ftp://' || scheme === 'sftp://') {
    try { new URL(trimmed); return true } catch { return false }
  }
  if (scheme === 'file://') {
    return trimmed.length > 'file://'.length
  }
  // bare file path — non-empty is sufficient
  return trimmed.length > 0
}

export function EditDocumentModal({ doc, onSuccess, onCancel }: Props) {
  const [newName, setNewName] = useState(doc.name)
  const [description, setDescription] = useState(doc.description ?? '')
  const [uri, setUri] = useState(doc.path ?? '')
  const [followLinks, setFollowLinks] = useState(false)
  const [maxDepth, setMaxDepth] = useState('3')
  const [maxDocuments, setMaxDocuments] = useState('50')
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [uriChecking, setUriChecking] = useState(false)
  const [uriOk, setUriOk] = useState<boolean | null>(null)

  const [updateDocument, { loading }] = useMutation(UPDATE_DOCUMENT)
  const [validateUri] = useLazyQuery(VALIDATE_URI, { fetchPolicy: 'no-cache' })

  const isHttpLike = detectScheme(uri) === 'http://' || detectScheme(uri) === 'https://'

  function validate(): FieldErrors {
    const errors: FieldErrors = {}
    if (!newName.trim()) errors.name = 'Name is required'
    if (!description.trim()) errors.description = 'Description is required'
    if (!uri.trim()) {
      errors.uri = 'URI or path is required'
    } else if (!isValidUri(uri)) {
      errors.uri = `Enter a valid URI (https://…, s3://bucket/key, ftp://host/path, sftp://host/path, file:///path, or a local file path)`
    }
    return errors
  }

  async function handleCheckUri() {
    if (!uri.trim() || !isValidUri(uri)) return
    setUriChecking(true)
    setUriOk(null)
    setFieldErrors(prev => ({ ...prev, uri: undefined }))
    try {
      const { data } = await validateUri({ variables: { uri } })
      if (data?.validateUri?.reachable) {
        setUriOk(true)
      } else {
        setUriOk(false)
        setFieldErrors(prev => ({
          ...prev,
          uri: data?.validateUri?.error ?? 'URI is not reachable',
        }))
      }
    } finally {
      setUriChecking(false)
    }
  }

  async function handleSubmit() {
    const errors = validate()
    setFieldErrors(errors)
    if (Object.keys(errors).length > 0) return

    setSubmitError(null)
    const input: Record<string, unknown> = {
      name: doc.name,
      description: description.trim(),
      uri: uri.trim(),
    }
    if (newName.trim() !== doc.name) input.new_name = newName.trim()
    if (isHttpLike) {
      input.follow_links = followLinks
      input.max_depth = parseInt(maxDepth, 10) || 3
      input.max_documents = parseInt(maxDocuments, 10) || 50
    }
    try {
      await updateDocument({ variables: { input } })
      onSuccess()
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Failed to update document')
    }
  }

  const inputCls = (hasError: boolean) =>
    `w-full px-3 py-2 text-sm border rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 ${
      hasError
        ? 'border-red-400 dark:border-red-500 focus:ring-red-400'
        : 'border-gray-300 dark:border-gray-600'
    }`
  const half = 'flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const lbl = 'text-xs text-gray-500 dark:text-gray-400 mb-0.5'
  const errMsg = 'mt-1 text-xs text-red-500 dark:text-red-400'

  return (
    <div className="space-y-3">
      <div>
        <p className={lbl}>Name</p>
        <input
          type="text"
          value={newName}
          onChange={(e) => { setNewName(e.target.value); setFieldErrors(p => ({ ...p, name: undefined })) }}
          className={inputCls(!!fieldErrors.name)}
        />
        {fieldErrors.name && <p className={errMsg}>{fieldErrors.name}</p>}
      </div>

      <div>
        <p className={lbl}>Description</p>
        <textarea
          rows={2}
          placeholder="Description"
          value={description}
          onChange={(e) => { setDescription(e.target.value); setFieldErrors(p => ({ ...p, description: undefined })) }}
          className={inputCls(!!fieldErrors.description) + ' resize-y'}
        />
        {fieldErrors.description && <p className={errMsg}>{fieldErrors.description}</p>}
      </div>

      <div>
        <p className={lbl}>URI / Path</p>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="https://example.com or /path/to/file.pdf"
            value={uri}
            onChange={(e) => { setUri(e.target.value); setUriOk(null); setFieldErrors(p => ({ ...p, uri: undefined })) }}
            className={inputCls(!!fieldErrors.uri) + ' flex-1'}
          />
          <button
            type="button"
            onClick={handleCheckUri}
            disabled={uriChecking || !uri.trim()}
            title="Test reachability"
            className="px-2.5 py-1.5 text-xs border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {uriChecking
              ? <span className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin inline-block" />
              : uriOk === true
                ? <span className="text-green-500">✓</span>
                : uriOk === false
                  ? <span className="text-red-500">✗</span>
                  : null}
            Test
          </button>
        </div>
        {uri.trim() && (() => {
          const scheme = detectScheme(uri.trim())
          const hints: Record<string, string> = {
            'http://': 'Web page',
            'https://': 'Web page (secure)',
            'ftp://': 'FTP server',
            'sftp://': 'SFTP server',
            's3://': 'AWS S3 object',
            's3a://': 'S3-compatible storage',
            'file://': 'Local file',
          }
          const label = scheme ? hints[scheme] : 'Local file path'
          return <p className="mt-0.5 text-xs text-gray-400 dark:text-gray-500">{label}</p>
        })()}
        {fieldErrors.uri && <p className={errMsg}>{fieldErrors.uri}</p>}
      </div>

      {isHttpLike && (
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

      {submitError && <p className="text-xs text-red-500 dark:text-red-400">{submitError}</p>}

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading && <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />}
          {loading ? 'Saving...' : 'Save'}
        </button>
      </div>
    </div>
  )
}
