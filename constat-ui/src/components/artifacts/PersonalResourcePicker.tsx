// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  XMarkIcon,
  EnvelopeIcon,
  FolderIcon,
  CalendarDaysIcon,
  GlobeAltIcon,
  ShieldExclamationIcon,
} from '@heroicons/react/24/outline'
import { useAuth } from '@/contexts/AuthContext'

interface PersonalResourcePickerProps {
  isOpen: boolean
  onClose: () => void
  sessionId: string
}

interface ResourceType {
  type: string
  provider: 'google' | 'microsoft'
  label: string
  icon: typeof EnvelopeIcon
  accent: string
  bgHover: string
}

const RESOURCE_TYPES: ResourceType[] = [
  { type: 'email', provider: 'google', label: 'Gmail', icon: EnvelopeIcon, accent: 'text-red-500 dark:text-red-400', bgHover: 'hover:border-red-300 dark:hover:border-red-700' },
  { type: 'email', provider: 'microsoft', label: 'Outlook', icon: EnvelopeIcon, accent: 'text-blue-500 dark:text-blue-400', bgHover: 'hover:border-blue-300 dark:hover:border-blue-700' },
  { type: 'drive', provider: 'google', label: 'Google Drive', icon: FolderIcon, accent: 'text-green-500 dark:text-green-400', bgHover: 'hover:border-green-300 dark:hover:border-green-700' },
  { type: 'drive', provider: 'microsoft', label: 'OneDrive', icon: FolderIcon, accent: 'text-sky-500 dark:text-sky-400', bgHover: 'hover:border-sky-300 dark:hover:border-sky-700' },
  { type: 'calendar', provider: 'google', label: 'Google Calendar', icon: CalendarDaysIcon, accent: 'text-blue-500 dark:text-blue-400', bgHover: 'hover:border-blue-300 dark:hover:border-blue-700' },
  { type: 'calendar', provider: 'microsoft', label: 'Outlook Calendar', icon: CalendarDaysIcon, accent: 'text-indigo-500 dark:text-indigo-400', bgHover: 'hover:border-indigo-300 dark:hover:border-indigo-700' },
  { type: 'sharepoint', provider: 'microsoft', label: 'SharePoint', icon: GlobeAltIcon, accent: 'text-teal-500 dark:text-teal-400', bgHover: 'hover:border-teal-300 dark:hover:border-teal-700' },
]

const GoogleIcon = () => (
  <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 24 24">
    <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
    <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
    <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
    <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
  </svg>
)

const MicrosoftIcon = () => (
  <svg className="w-4 h-4 flex-shrink-0" viewBox="0 0 23 23">
    <path fill="#f35325" d="M1 1h10v10H1z"/>
    <path fill="#81bc06" d="M12 1h10v10H12z"/>
    <path fill="#05a6f0" d="M1 12h10v10H1z"/>
    <path fill="#ffba08" d="M12 12h10v10H12z"/>
  </svg>
)

export function PersonalResourcePicker({ isOpen, onClose, sessionId: _sessionId }: PersonalResourcePickerProps) {
  const { user, isAuthDisabled } = useAuth()
  const [connecting, setConnecting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hasVault, setHasVault] = useState<boolean | null>(null)
  const [vaultPassword, setVaultPassword] = useState('')
  const [vaultConfirm, setVaultConfirm] = useState('')
  const [vaultError, setVaultError] = useState<string | null>(null)
  const [creatingVault, setCreatingVault] = useState(false)
  const passwordRef = useRef<HTMLInputElement>(null)

  const userId = isAuthDisabled ? 'default' : (user?.uid ?? 'default')

  const availableResources = useMemo(() => {
    if (isAuthDisabled) return RESOURCE_TYPES
    const linkedProviderIds = new Set(user?.providerData.map((p) => p.providerId) ?? [])
    return RESOURCE_TYPES.filter((r) => {
      if (r.provider === 'google') return linkedProviderIds.has('google.com')
      if (r.provider === 'microsoft') return linkedProviderIds.has('microsoft.com')
      return false
    })
  }, [isAuthDisabled, user])

  // Check vault status on open
  useEffect(() => {
    if (!isOpen) return
    setHasVault(null)
    setVaultPassword('')
    setVaultConfirm('')
    setVaultError(null)

    fetch(`/api/vault/${userId}/status`)
      .then((r) => {
        if (!r.ok) throw new Error(`Vault status check failed (${r.status})`)
        return r.json()
      })
      .then((data: { has_vault: boolean }) => setHasVault(data.has_vault))
      .catch((err) => setError(err instanceof Error ? err.message : 'Vault status check failed'))
  }, [isOpen, userId])

  const handleCreateVault = useCallback(async () => {
    setVaultError(null)
    if (!vaultPassword) {
      setVaultError('Password is required.')
      return
    }
    if (vaultPassword !== vaultConfirm) {
      setVaultError('Passwords do not match.')
      return
    }
    setCreatingVault(true)
    try {
      const resp = await fetch(`/api/vault/${userId}/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password: vaultPassword }),
      })
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        throw new Error(body.detail || `Failed to create vault (${resp.status})`)
      }
      setHasVault(true)
      setVaultPassword('')
      setVaultConfirm('')
    } catch (err) {
      setVaultError(err instanceof Error ? err.message : 'Failed to create vault')
    } finally {
      setCreatingVault(false)
    }
  }, [userId, vaultPassword, vaultConfirm])

  const handleConnect = useCallback((resource: ResourceType) => {
    setConnecting(`${resource.provider}-${resource.type}`)
    setError(null)

    const redirectUri = `${window.location.origin}/api/oauth/callback`
    const authUrl = `/api/oauth/authorize?provider=${resource.provider}&resource_type=${resource.type}&redirect_uri=${encodeURIComponent(redirectUri)}`

    const popup = window.open(authUrl, 'oauth-popup', 'width=600,height=700,scrollbars=yes')
    if (!popup) {
      setError('Popup blocked. Please allow popups for this site.')
      setConnecting(null)
      return
    }

    const handleMessage = async (event: MessageEvent) => {
      if (event.data?.type !== 'oauth-complete') return
      window.removeEventListener('message', handleMessage)

      const { provider, resourceType, email, refresh_token } = event.data
      const accountName = `${resourceType}-${provider}-${email?.split('@')[0] || 'account'}`

      try {
        const resp = await fetch(`/api/accounts/?user_id=default`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: accountName,
            type: resourceType === 'email' ? 'imap' : resourceType,
            provider,
            display_name: `${resource.label} (${email || 'connected'})`,
            email: email || '',
            refresh_token,
          }),
        })
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}))
          throw new Error(body.detail || `Failed to save account (${resp.status})`)
        }
        onClose()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to save account')
      } finally {
        setConnecting(null)
      }
    }

    window.addEventListener('message', handleMessage)
  }, [onClose])

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // Focus password field when vault warning is shown
  useEffect(() => {
    if (hasVault === false) {
      setTimeout(() => passwordRef.current?.focus(), 50)
    }
  }, [hasVault])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-lg"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              Connect Personal Resource
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              Choose a service to connect via OAuth
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5">
          {error && (
            <div className="mb-4 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {/* Vault not established — show warning + create form */}
          {hasVault === false && (
            <div
              data-testid="vault-warning"
              className="mb-4 px-4 py-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-700 rounded-md"
            >
              <div className="flex items-start gap-2 mb-3">
                <ShieldExclamationIcon className="w-5 h-5 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-800 dark:text-amber-200">
                  A vault password is required to securely store your credentials. Set one before connecting.
                </p>
              </div>

              <div className="space-y-2">
                <input
                  ref={passwordRef}
                  type="password"
                  placeholder="Vault password"
                  value={vaultPassword}
                  onChange={(e) => setVaultPassword(e.target.value)}
                  data-testid="vault-password-input"
                  className="w-full px-3 py-1.5 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-400"
                />
                <input
                  type="password"
                  placeholder="Confirm password"
                  value={vaultConfirm}
                  onChange={(e) => setVaultConfirm(e.target.value)}
                  data-testid="vault-confirm-input"
                  className="w-full px-3 py-1.5 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-400"
                />
                {vaultError && (
                  <p className="text-xs text-red-600 dark:text-red-400">{vaultError}</p>
                )}
                <button
                  onClick={handleCreateVault}
                  disabled={creatingVault}
                  data-testid="vault-set-password-btn"
                  className="w-full py-1.5 text-xs font-medium bg-amber-600 hover:bg-amber-700 disabled:opacity-50 text-white rounded transition-colors"
                >
                  {creatingVault ? 'Setting password…' : 'Set Password'}
                </button>
              </div>
            </div>
          )}

          {availableResources.length === 0 && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              No OAuth providers are linked to your account. Sign in with Google or Microsoft to connect personal resources.
            </p>
          )}

          <div className={`grid grid-cols-2 gap-3 ${hasVault === false ? 'opacity-40 pointer-events-none' : ''}`}>
            {availableResources.map((resource) => {
              const key = `${resource.provider}-${resource.type}`
              const isConnecting = connecting === key
              const Icon = resource.icon
              const ProviderLogo = resource.provider === 'google' ? GoogleIcon : MicrosoftIcon

              return (
                <button
                  key={key}
                  onClick={() => handleConnect(resource)}
                  disabled={isConnecting || connecting !== null || hasVault !== true}
                  data-testid={`resource-card-${key}`}
                  className={`
                    group relative flex flex-col items-start gap-3 p-4 rounded-lg border border-gray-200 dark:border-gray-700
                    bg-white dark:bg-gray-800/50 transition-all duration-150
                    ${resource.bgHover}
                    hover:shadow-md dark:hover:shadow-lg dark:hover:shadow-black/20
                    disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:shadow-none
                  `}
                >
                  <div className="flex items-center gap-2 w-full">
                    <Icon className={`w-5 h-5 ${resource.accent}`} />
                    <ProviderLogo />
                  </div>
                  <div className="text-left">
                    <span className="text-sm font-medium text-gray-800 dark:text-gray-200 block">
                      {resource.label}
                    </span>
                    <span className="text-[11px] text-gray-400 dark:text-gray-500">
                      {resource.type === 'email' ? 'Inbox' : resource.type === 'drive' ? 'Files' : resource.type === 'calendar' ? 'Events' : 'Sites'}
                    </span>
                  </div>
                  {isConnecting ? (
                    <div className="absolute top-3 right-3">
                      <div className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                  ) : (
                    <span className="absolute top-3 right-3 text-[10px] font-medium text-gray-400 dark:text-gray-500 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors">
                      Connect
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}
