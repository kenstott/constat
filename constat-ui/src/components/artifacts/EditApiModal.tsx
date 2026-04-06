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
import { UPDATE_API, VALIDATE_URI } from '@/graphql/operations/sources'
import { useSessionContext } from '@/contexts/SessionContext'
import type { ApiSourceInfo } from '@/types/api'

const API_TYPES = [
  { value: 'rest', label: 'REST' },
  { value: 'graphql', label: 'GraphQL' },
  { value: 'openapi', label: 'OpenAPI' },
]

const AUTH_TYPES = [
  { value: 'none', label: 'No Auth' },
  { value: 'bearer', label: 'Bearer Token' },
  { value: 'basic', label: 'Username / Password' },
  { value: 'api_key', label: 'API Key' },
  { value: 'oauth2', label: 'OAuth2 Client Credentials' },
]

interface Props {
  api: ApiSourceInfo
  onSuccess: () => void
  onCancel: () => void
}

export function EditApiModal({ api, onSuccess, onCancel }: Props) {
  const { session } = useSessionContext()
  const [newName, setNewName] = useState(api.name)
  const [baseUrl, setBaseUrl] = useState(api.base_url ?? '')
  const [apiType, setApiType] = useState(api.type ?? 'rest')
  const [description, setDescription] = useState(api.description ?? '')
  const [authType, setAuthType] = useState('none')
  const [authToken, setAuthToken] = useState('')
  const [authUsername, setAuthUsername] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authHeader, setAuthHeader] = useState('X-API-Key')
  const [authClientId, setAuthClientId] = useState('')
  const [authClientSecret, setAuthClientSecret] = useState('')
  const [authTokenUrl, setAuthTokenUrl] = useState('')
  const [urlChecking, setUrlChecking] = useState(false)
  const [urlOk, setUrlOk] = useState<boolean | null>(null)
  const [urlError, setUrlError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [updateApi, { loading }] = useMutation(UPDATE_API)
  const [validateUri] = useLazyQuery(VALIDATE_URI, { fetchPolicy: 'no-cache' })

  function canSubmit(): boolean {
    if (!baseUrl) return false
    if (authType === 'bearer') return authToken.length > 0
    if (authType === 'basic') return authUsername.length > 0
    if (authType === 'api_key') return authHeader.length > 0 && authToken.length > 0
    if (authType === 'oauth2') return authClientId.length > 0 && authClientSecret.length > 0 && authTokenUrl.length > 0
    return true
  }

  async function handleCheckUrl() {
    if (!baseUrl.trim()) return
    setUrlChecking(true)
    setUrlOk(null)
    setUrlError(null)
    try {
      const { data } = await validateUri({ variables: { uri: baseUrl.trim() } })
      if (data?.validateUri?.reachable) {
        setUrlOk(true)
      } else {
        setUrlOk(false)
        setUrlError(data?.validateUri?.error ?? 'URL is not reachable')
      }
    } finally {
      setUrlChecking(false)
    }
  }

  async function handleSubmit() {
    if (!canSubmit()) return
    setError(null)
    const input: Record<string, unknown> = {
      name: api.name,
      base_url: baseUrl,
      type: apiType,
      description: description || undefined,
      auth_type: authType,
    }
    if (newName.trim() !== api.name) input.new_name = newName.trim()
    if (authType === 'bearer') input.auth_token = authToken
    if (authType === 'basic') { input.auth_username = authUsername; input.auth_password = authPassword }
    if (authType === 'api_key') { input.auth_header = authHeader; input.auth_token = authToken }
    if (authType === 'oauth2') { input.auth_client_id = authClientId; input.auth_client_secret = authClientSecret; input.auth_token_url = authTokenUrl }
    try {
      await updateApi({ variables: { sessionId: session?.session_id, input } })
      onSuccess()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update API')
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
        <p className={lbl}>Base URL</p>
        <div className="flex gap-2">
          <input
            type="url"
            placeholder="https://api.example.com"
            value={baseUrl}
            onChange={(e) => { setBaseUrl(e.target.value); setUrlOk(null); setUrlError(null) }}
            className={input + ' flex-1'}
          />
          <button
            type="button"
            onClick={handleCheckUrl}
            disabled={urlChecking || !baseUrl.trim()}
            title="Test reachability"
            className="px-2.5 py-1.5 text-xs border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
          >
            {urlChecking
              ? <span className="w-3 h-3 border-2 border-gray-400 border-t-transparent rounded-full animate-spin inline-block" />
              : urlOk === true
                ? <span className="text-green-500">✓</span>
                : urlOk === false
                  ? <span className="text-red-500">✗</span>
                  : null}
            Test
          </button>
        </div>
        {urlError && <p className="mt-1 text-xs text-red-500 dark:text-red-400">{urlError}</p>}
      </div>

      <div>
        <p className={lbl}>Type</p>
        <select value={apiType} onChange={(e) => setApiType(e.target.value)} className={input}>
          {API_TYPES.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      <div>
        <p className={lbl}>Description (optional)</p>
        <textarea rows={2} placeholder="What this API provides" value={description} onChange={(e) => setDescription(e.target.value)} className={input + ' resize-y'} />
      </div>

      <div>
        <p className={lbl}>Authentication</p>
        <select value={authType} onChange={(e) => setAuthType(e.target.value)} className={input}>
          {AUTH_TYPES.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      {authType === 'bearer' && (
        <div>
          <p className={lbl}>Token</p>
          <input type="password" placeholder="Bearer token" value={authToken} onChange={(e) => setAuthToken(e.target.value)} className={input} />
        </div>
      )}

      {authType === 'basic' && (
        <div className="flex gap-2">
          <div className="flex-1">
            <p className={lbl}>Username</p>
            <input type="text" placeholder="user" value={authUsername} onChange={(e) => setAuthUsername(e.target.value)} className={half} />
          </div>
          <div className="flex-1">
            <p className={lbl}>Password</p>
            <input type="password" placeholder="password" value={authPassword} onChange={(e) => setAuthPassword(e.target.value)} className={half} />
          </div>
        </div>
      )}

      {authType === 'api_key' && (
        <>
          <div>
            <p className={lbl}>Header Name</p>
            <input type="text" placeholder="X-API-Key" value={authHeader} onChange={(e) => setAuthHeader(e.target.value)} className={input} />
          </div>
          <div>
            <p className={lbl}>API Key Value</p>
            <input type="password" placeholder="key value" value={authToken} onChange={(e) => setAuthToken(e.target.value)} className={input} />
          </div>
        </>
      )}

      {authType === 'oauth2' && (
        <>
          <div className="flex gap-2">
            <div className="flex-1">
              <p className={lbl}>Client ID</p>
              <input type="text" placeholder="client_id" value={authClientId} onChange={(e) => setAuthClientId(e.target.value)} className={half} />
            </div>
            <div className="flex-1">
              <p className={lbl}>Client Secret</p>
              <input type="password" placeholder="client_secret" value={authClientSecret} onChange={(e) => setAuthClientSecret(e.target.value)} className={half} />
            </div>
          </div>
          <div>
            <p className={lbl}>Token URL</p>
            <input type="url" placeholder="https://auth.example.com/oauth/token" value={authTokenUrl} onChange={(e) => setAuthTokenUrl(e.target.value)} className={input} />
          </div>
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
