// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// MCP Server Picker — modal for browsing and adding MCP servers from the catalog

import { useState, useEffect, useCallback } from 'react'
import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react'
import {
  XMarkIcon,
  MagnifyingGlassIcon,
  ServerStackIcon,
  PlusIcon,
} from '@heroicons/react/24/outline'
import { useDocumentMutations } from '@/hooks/useDataSources'

interface McpServer {
  name: string
  slug: string
  description: string
  category: string
  capabilities: string[]
}

interface McpServerPickerProps {
  open: boolean
  onClose: () => void
}

const CATEGORIES = [
  { value: '', label: 'All' },
  { value: 'development', label: 'Development' },
  { value: 'communication', label: 'Communication' },
  { value: 'documents', label: 'Documents' },
  { value: 'project-management', label: 'Project Mgmt' },
  { value: 'database', label: 'Database' },
  { value: 'drive', label: 'Drive' },
]

const CAPABILITY_COLORS: Record<string, string> = {
  resources: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
  tools: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200',
}

export function McpServerPicker({ open, onClose }: McpServerPickerProps) {
  const [servers, setServers] = useState<McpServer[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const [configuring, setConfiguring] = useState<McpServer | null>(null)

  const fetchServers = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams()
      if (category) params.set('category', category)
      if (search) params.set('q', search)
      const qs = params.toString()
      const resp = await fetch(`/api/mcp/catalog${qs ? `?${qs}` : ''}`)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const data = await resp.json()
      setServers(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load catalog')
    } finally {
      setLoading(false)
    }
  }, [category, search])

  useEffect(() => {
    if (open) fetchServers()
  }, [open, fetchServers])

  return (
    <Dialog open={open} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="mx-auto w-full max-w-2xl rounded-lg bg-white shadow-xl dark:bg-gray-800">
          <div className="flex items-center justify-between border-b border-gray-200 px-6 py-4 dark:border-gray-700">
            <DialogTitle className="flex items-center gap-2 text-lg font-semibold text-gray-900 dark:text-gray-100">
              <ServerStackIcon className="h-5 w-5" />
              MCP Server Catalog
            </DialogTitle>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>

          {configuring ? (
            <ConfigureForm server={configuring} onBack={() => setConfiguring(null)} onClose={onClose} />
          ) : (
            <CatalogBrowser
              servers={servers}
              loading={loading}
              error={error}
              search={search}
              category={category}
              onSearchChange={setSearch}
              onCategoryChange={setCategory}
              onAdd={setConfiguring}
            />
          )}
        </DialogPanel>
      </div>
    </Dialog>
  )
}

// ---------------------------------------------------------------------------
// Catalog browser view
// ---------------------------------------------------------------------------

interface CatalogBrowserProps {
  servers: McpServer[]
  loading: boolean
  error: string | null
  search: string
  category: string
  onSearchChange: (v: string) => void
  onCategoryChange: (v: string) => void
  onAdd: (server: McpServer) => void
}

function CatalogBrowser({
  servers, loading, error, search, category,
  onSearchChange, onCategoryChange, onAdd,
}: CatalogBrowserProps) {
  return (
    <div className="p-6">
      {/* Search */}
      <div className="relative mb-4">
        <MagnifyingGlassIcon className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" />
        <input
          type="text"
          value={search}
          onChange={e => onSearchChange(e.target.value)}
          placeholder="Search servers..."
          className="w-full rounded-md border border-gray-300 py-2 pl-9 pr-3 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
        />
      </div>

      {/* Category tabs */}
      <div className="mb-4 flex flex-wrap gap-1">
        {CATEGORIES.map(c => (
          <button
            key={c.value}
            onClick={() => onCategoryChange(c.value)}
            className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
              category === c.value
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600'
            }`}
          >
            {c.label}
          </button>
        ))}
      </div>

      {/* Server list */}
      <div className="max-h-80 overflow-y-auto space-y-2">
        {loading && <p className="text-sm text-gray-500">Loading catalog...</p>}
        {error && <p className="text-sm text-red-500">Error: {error}</p>}
        {!loading && !error && servers.length === 0 && (
          <p className="text-sm text-gray-500">No servers found.</p>
        )}
        {servers.map(server => (
          <ServerCard key={server.slug} server={server} onAdd={() => onAdd(server)} />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Server card
// ---------------------------------------------------------------------------

function ServerCard({ server, onAdd }: { server: McpServer; onAdd: () => void }) {
  return (
    <div className="flex items-center justify-between rounded-md border border-gray-200 p-3 dark:border-gray-600">
      <div className="min-w-0 flex-1">
        <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">{server.name}</h4>
        <p className="text-xs text-gray-500 dark:text-gray-400">{server.description}</p>
        <div className="mt-1 flex gap-1">
          {server.capabilities.map(cap => (
            <span
              key={cap}
              className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${
                CAPABILITY_COLORS[cap] ?? 'bg-gray-100 text-gray-600'
              }`}
            >
              {cap}
            </span>
          ))}
        </div>
      </div>
      <button
        onClick={onAdd}
        className="ml-3 flex items-center gap-1 rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700"
      >
        <PlusIcon className="h-3.5 w-3.5" />
        Add
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Configure form
// ---------------------------------------------------------------------------

function ConfigureForm({
  server, onBack, onClose,
}: { server: McpServer; onBack: () => void; onClose: () => void }) {
  const { addDocumentUri } = useDocumentMutations()
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) {
      setFormError('URL is required')
      return
    }
    setSubmitting(true)
    setFormError(null)
    try {
      await addDocumentUri({
        name: server.slug,
        uri: url.trim(),
        type: 'mcp',
        description: server.description,
        auth: token ? { method: 'bearer', token: token.trim() } : undefined,
      })
      onClose()
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to add source')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="p-6 space-y-4">
      <div className="flex items-center gap-2 mb-2">
        <button type="button" onClick={onBack} className="text-sm text-blue-600 hover:underline">
          Back
        </button>
        <span className="text-sm text-gray-500">/ Configure {server.name}</span>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Server URL
        </label>
        <input
          type="text"
          value={url}
          onChange={e => setUrl(e.target.value)}
          placeholder="https://mcp-server.example.com or stdio://command"
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Auth Token <span className="text-gray-400">(optional)</span>
        </label>
        <input
          type="password"
          value={token}
          onChange={e => setToken(e.target.value)}
          placeholder="Bearer token"
          className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-gray-100"
        />
      </div>

      {formError && <p className="text-sm text-red-500">{formError}</p>}

      <div className="flex justify-end gap-2 pt-2">
        <button
          type="button"
          onClick={onBack}
          className="rounded-md px-4 py-2 text-sm text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-200"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={submitting}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {submitting ? 'Adding...' : 'Add Server'}
        </button>
      </div>
    </form>
  )
}
