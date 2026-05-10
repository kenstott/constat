// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useCallback, useEffect, useState } from 'react'
import {
  XMarkIcon,
  ArrowPathIcon,
  TrashIcon,
  CheckCircleIcon,
  PauseCircleIcon,
  EnvelopeIcon,
  FolderIcon,
  CalendarDaysIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline'

export interface AccountSummary {
  name: string
  type: string
  provider: string
  display_name: string
  email: string
  active: boolean
  created_at: string
}

interface AccountManagerProps {
  isOpen: boolean
  onClose: () => void
}

const TYPE_ICONS: Record<string, typeof EnvelopeIcon> = {
  imap: EnvelopeIcon,
  drive: FolderIcon,
  calendar: CalendarDaysIcon,
  sharepoint: GlobeAltIcon,
}

const TYPE_COLORS: Record<string, string> = {
  imap: 'text-red-500 dark:text-red-400',
  drive: 'text-green-500 dark:text-green-400',
  calendar: 'text-blue-500 dark:text-blue-400',
  sharepoint: 'text-teal-500 dark:text-teal-400',
}

const PROVIDER_LABELS: Record<string, string> = {
  google: 'Google',
  microsoft: 'Microsoft',
}

export function AccountManager({ isOpen, onClose }: AccountManagerProps) {
  const [accounts, setAccounts] = useState<AccountSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [actionPending, setActionPending] = useState<string | null>(null)
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const fetchAccounts = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const resp = await fetch('/api/accounts/?user_id=default')
      if (!resp.ok) throw new Error(`Failed to load accounts (${resp.status})`)
      const data = await resp.json()
      setAccounts(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load accounts')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isOpen) fetchAccounts()
  }, [isOpen, fetchAccounts])

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  const handleToggleActive = useCallback(async (account: AccountSummary) => {
    setActionPending(account.name)
    try {
      const resp = await fetch(`/api/accounts/${encodeURIComponent(account.name)}?user_id=default`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: !account.active }),
      })
      if (!resp.ok) throw new Error('Failed to update account')
      setAccounts((prev) =>
        prev.map((a) => a.name === account.name ? { ...a, active: !a.active } : a)
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update account')
    } finally {
      setActionPending(null)
    }
  }, [])

  const handleRemove = useCallback(async (accountName: string) => {
    setActionPending(accountName)
    setConfirmDelete(null)
    try {
      const resp = await fetch(`/api/accounts/${encodeURIComponent(accountName)}?user_id=default`, {
        method: 'DELETE',
      })
      if (!resp.ok) throw new Error('Failed to remove account')
      setAccounts((prev) => prev.filter((a) => a.name !== accountName))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remove account')
    } finally {
      setActionPending(null)
    }
  }, [])

  const handleReAuth = useCallback((account: AccountSummary) => {
    const redirectUri = `${window.location.origin}/api/oauth/callback`
    const resourceType = account.type === 'imap' ? 'email' : account.type
    const authUrl = `/api/oauth/authorize?provider=${account.provider}&resource_type=${resourceType}&redirect_uri=${encodeURIComponent(redirectUri)}`
    window.open(authUrl, 'oauth-popup', 'width=600,height=700,scrollbars=yes')
  }, [])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-md flex flex-col max-h-[80vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-200 dark:border-gray-700 shrink-0">
          <div>
            <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
              My Accounts
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              {accounts.length} connected {accounts.length === 1 ? 'account' : 'accounts'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {error && (
            <div className="mb-3 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-xs text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-5 h-5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : accounts.length === 0 ? (
            <div className="text-center py-12">
              <p className="text-sm text-gray-500 dark:text-gray-400">No accounts connected yet.</p>
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                Use the + button to connect a personal resource.
              </p>
            </div>
          ) : (
            <div className="space-y-2" role="list" aria-label="Connected accounts">
              {accounts.map((account) => {
                const Icon = TYPE_ICONS[account.type] || EnvelopeIcon
                const iconColor = TYPE_COLORS[account.type] || 'text-gray-400'
                const isPending = actionPending === account.name
                const isConfirmingDelete = confirmDelete === account.name

                return (
                  <div
                    key={account.name}
                    role="listitem"
                    data-testid={`account-row-${account.name}`}
                    className={`
                      group flex items-center gap-3 px-3 py-3 rounded-lg border transition-colors
                      ${account.active
                        ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50'
                        : 'border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/30 opacity-60'
                      }
                    `}
                  >
                    {/* Icon */}
                    <Icon className={`w-5 h-5 shrink-0 ${iconColor}`} />

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">
                          {account.display_name}
                        </span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded-full font-medium shrink-0" style={{
                          ...(account.active
                            ? { color: 'rgb(22 163 74)', backgroundColor: 'rgb(22 163 74 / 0.1)' }
                            : { color: 'rgb(156 163 175)', backgroundColor: 'rgb(156 163 175 / 0.1)' }
                          ),
                        }}>
                          {account.active ? 'active' : 'paused'}
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5 mt-0.5">
                        <span className="text-[11px] text-gray-400 dark:text-gray-500 truncate">
                          {account.email}
                        </span>
                        <span className="text-[10px] text-gray-300 dark:text-gray-600">
                          {PROVIDER_LABELS[account.provider] || account.provider}
                        </span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                      {isPending ? (
                        <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                      ) : isConfirmingDelete ? (
                        <div className="flex items-center gap-1">
                          <button
                            onClick={() => handleRemove(account.name)}
                            className="px-2 py-0.5 text-[10px] font-medium text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 rounded hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors"
                            data-testid={`confirm-delete-${account.name}`}
                          >
                            Remove
                          </button>
                          <button
                            onClick={() => setConfirmDelete(null)}
                            className="px-2 py-0.5 text-[10px] text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                          >
                            No
                          </button>
                        </div>
                      ) : (
                        <>
                          <button
                            onClick={() => handleToggleActive(account)}
                            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                            title={account.active ? 'Pause account' : 'Activate account'}
                            data-testid={`toggle-active-${account.name}`}
                          >
                            {account.active
                              ? <PauseCircleIcon className="w-4 h-4" />
                              : <CheckCircleIcon className="w-4 h-4" />
                            }
                          </button>
                          <button
                            onClick={() => handleReAuth(account)}
                            className="p-1 text-gray-400 hover:text-amber-500 dark:hover:text-amber-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                            title="Re-authenticate"
                          >
                            <ArrowPathIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setConfirmDelete(account.name)}
                            className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                            title="Remove account"
                            data-testid={`delete-${account.name}`}
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
