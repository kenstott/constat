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

export interface ApiAddFields {
  name: string
  baseUrl: string
  type: string
  description: string
  authType: string
  authToken: string
  authUsername: string
  authPassword: string
  authHeader: string
  authClientId: string
  authClientSecret: string
  authTokenUrl: string
}

interface Props {
  onAdd: (fields: ApiAddFields) => void
  onCancel: () => void
  uploading: boolean
}

export function AddApiModal({ onAdd, onCancel, uploading }: Props) {
  const [name, setName] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiType, setApiType] = useState('rest')
  const [description, setDescription] = useState('')
  const [authType, setAuthType] = useState('none')
  const [authToken, setAuthToken] = useState('')
  const [authUsername, setAuthUsername] = useState('')
  const [authPassword, setAuthPassword] = useState('')
  const [authHeader, setAuthHeader] = useState('X-API-Key')
  const [authClientId, setAuthClientId] = useState('')
  const [authClientSecret, setAuthClientSecret] = useState('')
  const [authTokenUrl, setAuthTokenUrl] = useState('')

  function canSubmit() {
    if (!name || !baseUrl) return false
    if (authType === 'bearer') return authToken.length > 0
    if (authType === 'basic') return authUsername.length > 0
    if (authType === 'api_key') return authHeader.length > 0 && authToken.length > 0
    if (authType === 'oauth2') return authClientId.length > 0 && authClientSecret.length > 0 && authTokenUrl.length > 0
    return true
  }

  function handleSubmit() {
    if (!canSubmit()) return
    onAdd({ name, baseUrl, type: apiType, description, authType, authToken, authUsername, authPassword, authHeader, authClientId, authClientSecret, authTokenUrl })
  }

  const input = 'w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const half = 'flex-1 px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100'
  const lbl = 'text-xs text-gray-500 dark:text-gray-400 mb-0.5'

  return (
    <div className="space-y-3">
      <input type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)} className={input} />

      <input type="url" placeholder="Base URL (https://api.example.com)" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} className={input} />

      <div>
        <p className={lbl}>Type</p>
        <select value={apiType} onChange={(e) => setApiType(e.target.value)} className={input}>
          {API_TYPES.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
        </select>
      </div>

      <div>
        <p className={lbl}>Description (optional)</p>
        <input type="text" placeholder="What this API provides" value={description} onChange={(e) => setDescription(e.target.value)} className={input} />
      </div>

      <div>
        <p className={lbl}>Authentication</p>
        <select value={authType} onChange={(e) => { setAuthType(e.target.value) }} className={input}>
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

      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={onCancel}
          className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
        >
          Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={uploading || !canSubmit()}
          className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {uploading && <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />}
          {uploading ? 'Adding...' : 'Add'}
        </button>
      </div>
    </div>
  )
}
