// Domain Panel — recursive tree view with toggle, expand/collapse, CRUD, drag-and-drop

import { useEffect, useState, useCallback, useRef } from 'react'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  PlusIcon,
  ArrowsRightLeftIcon,
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import * as sessionsApi from '@/api/sessions'
import type { DomainTreeNode } from '@/api/sessions'

interface DragItem {
  type: 'database' | 'api' | 'document'
  name: string
  sourceDomain: string
}

function DomainTreeNodeView({
  node,
  depth,
  activeDomains,
  isAdmin,
  userId,
  onToggle,
  onEdit,
  onDelete,
  onRename,
  onPromote,
  onDrop,
}: {
  node: DomainTreeNode
  depth: number
  activeDomains: string[]
  isAdmin: boolean
  userId: string
  onToggle: (filename: string) => void
  onEdit: (filename: string) => void
  onDelete: (filename: string) => void
  onRename: (filename: string, newName: string) => void
  onPromote: (filename: string) => void
  onDrop: (item: DragItem, targetDomain: string) => void
}) {
  const [expanded, setExpanded] = useState(depth < 2)
  const [menuOpen, setMenuOpen] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [renameValue, setRenameValue] = useState(node.name)
  const [dragOver, setDragOver] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const hasChildren = node.children.length > 0
  const isSystem = node.filename === 'system'
  const isActive = isSystem || activeDomains.includes(node.filename)
  const resourceCount =
    node.databases.length + node.apis.length + node.documents.length
  const canModify = isAdmin || (node.tier !== 'system' && (!node.owner || node.owner === userId))
  const showLock = !canModify
  const canPromote = node.tier === 'user' && (isAdmin || !node.owner || node.owner === userId)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragOver(false)
    try {
      const data = JSON.parse(e.dataTransfer.getData('application/json')) as DragItem
      if (data.sourceDomain !== node.filename) {
        onDrop(data, node.filename)
      }
    } catch {
      // Ignore invalid drag data
    }
  }

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [menuOpen])

  const handleRenameSubmit = () => {
    const trimmed = renameValue.trim()
    if (trimmed && trimmed !== node.name) {
      onRename(node.filename, trimmed)
    }
    setRenaming(false)
  }

  return (
    <div>
      <div
        className={`group flex items-center gap-1.5 py-1 rounded transition-colors ${
          dragOver
            ? 'bg-primary-50 dark:bg-primary-900/30 ring-1 ring-primary-400'
            : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
        }`}
        style={{ paddingLeft: depth * 16 }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Expand/collapse toggle */}
        <button
          onClick={() => hasChildren && setExpanded(!expanded)}
          className={`w-4 h-4 flex items-center justify-center flex-shrink-0 ${
            hasChildren
              ? 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 cursor-pointer'
              : 'text-transparent cursor-default'
          }`}
        >
          {hasChildren &&
            (expanded ? (
              <ChevronDownIcon className="w-3 h-3" />
            ) : (
              <ChevronRightIcon className="w-3 h-3" />
            ))}
        </button>

        {/* Checkbox — system domain is always active */}
        <input
          type="checkbox"
          checked={isActive}
          disabled={isSystem}
          onChange={() => !isSystem && onToggle(node.filename)}
          className={`w-3.5 h-3.5 rounded border-gray-300 text-primary-600 focus:ring-primary-500 ${isSystem ? 'opacity-50 cursor-default' : 'cursor-pointer'}`}
        />

        {/* Lock — shown only when user cannot modify this domain */}
        {showLock && (
          <span className="text-[10px] text-gray-400" title="Read-only">
            {'\u{1F512}'}
          </span>
        )}

        {/* Name or rename input */}
        {renaming ? (
          <input
            autoFocus
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onBlur={handleRenameSubmit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleRenameSubmit()
              if (e.key === 'Escape') setRenaming(false)
            }}
            className="text-xs flex-1 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded px-1 py-0.5 outline-none focus:ring-1 focus:ring-primary-500"
          />
        ) : (
          <span
            className={`text-xs flex-1 truncate ${
              isActive
                ? 'text-gray-800 dark:text-gray-200 font-medium'
                : 'text-gray-600 dark:text-gray-400'
            }`}
          >
            {node.name}
          </span>
        )}

        {/* Resource count badge */}
        {resourceCount > 0 && (
          <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400">
            {resourceCount}
          </span>
        )}

        {/* Context menu */}
        <div className="relative" ref={menuRef}>
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="w-5 h-5 flex items-center justify-center opacity-0 group-hover:opacity-100 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-opacity"
          >
            <EllipsisVerticalIcon className="w-3.5 h-3.5" />
          </button>
          {menuOpen && (
            <div className="absolute right-0 top-5 z-50 w-32 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg py-1">
              {canModify && (
                <button
                  onClick={() => {
                    setMenuOpen(false)
                    setRenameValue(node.name)
                    setRenaming(true)
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                >
                  <PencilIcon className="w-3 h-3" /> Rename
                </button>
              )}
              <button
                onClick={() => {
                  setMenuOpen(false)
                  onEdit(node.filename)
                }}
                className="w-full text-left px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
              >
                <PencilIcon className="w-3 h-3" /> Edit YAML
              </button>
              {canPromote && (
                <button
                  onClick={() => {
                    setMenuOpen(false)
                    onPromote(node.filename)
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                >
                  <ArrowsRightLeftIcon className="w-3 h-3" /> Promote to Shared
                </button>
              )}
              {canModify && (
                <button
                  onClick={() => {
                    setMenuOpen(false)
                    onDelete(node.filename)
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs text-red-600 dark:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                >
                  <TrashIcon className="w-3 h-3" /> Delete
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Children */}
      {expanded && node.children.map((child) => (
        <DomainTreeNodeView
          key={child.filename}
          node={child}
          depth={depth + 1}
          activeDomains={activeDomains}
          isAdmin={isAdmin}
          userId={userId}
          onToggle={onToggle}
          onEdit={onEdit}
          onDelete={onDelete}
          onRename={onRename}
          onPromote={onPromote}
          onDrop={onDrop}
        />
      ))}
    </div>
  )
}

export default function DomainPanel() {
  const { session, updateSession } = useSessionStore()
  const { fetchDataSources } = useArtifactStore()
  const authState = useAuthStore()
  // Derive from raw state — Zustand getter `isAdmin` goes stale after set()
  const isAdmin = isAuthDisabled || authState.permissions?.persona === 'platform_admin'
  const userId = isAuthDisabled ? 'default' : (authState.user?.uid || 'default')
  const [tree, setTree] = useState<DomainTreeNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // Create domain form
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [creating, setCreating] = useState(false)
  // YAML editor
  const [editingFilename, setEditingFilename] = useState<string | null>(null)
  const [yamlContent, setYamlContent] = useState('')
  const [yamlPath, setYamlPath] = useState('')
  const [saving, setSaving] = useState(false)
  // Delete confirmation
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const refreshTree = useCallback(async () => {
    try {
      const nodes = await sessionsApi.getDomainTree()
      setTree(nodes)
    } catch (err) {
      setError(String(err))
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    sessionsApi
      .getDomainTree()
      .then((nodes) => {
        if (!cancelled) setTree(nodes)
      })
      .catch((err) => {
        if (!cancelled) setError(String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const handleToggle = useCallback(
    async (filename: string) => {
      if (!session) return
      setError(null)
      const current = session.active_domains || []
      const isSelected = current.includes(filename)
      const newDomains = isSelected
        ? current.filter((d) => d !== filename)
        : [...current, filename]

      try {
        await sessionsApi.setActiveDomains(session.session_id, newDomains)
        updateSession({ active_domains: newDomains })
        await fetchDataSources(session.session_id)
      } catch (err: unknown) {
        console.error('Failed to toggle domain:', err)
        if (err && typeof err === 'object' && 'message' in err) {
          setError((err as { message: string }).message)
        }
      }
    },
    [session, updateSession, fetchDataSources],
  )

  const handleEdit = useCallback(async (filename: string) => {
    try {
      const result = await sessionsApi.getDomainContent(filename)
      setYamlContent(result.content)
      setYamlPath(result.path)
      setEditingFilename(filename)
    } catch (err) {
      console.error('Failed to load domain:', err)
    }
  }, [])

  const handleSaveYaml = useCallback(async () => {
    if (!editingFilename) return
    setSaving(true)
    try {
      await sessionsApi.updateDomainContent(editingFilename, yamlContent)
      setEditingFilename(null)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    } finally {
      setSaving(false)
    }
  }, [editingFilename, yamlContent, refreshTree])

  const handleDelete = useCallback(async (filename: string) => {
    try {
      await sessionsApi.deleteDomain(filename)
      setConfirmDelete(null)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree])

  const handleRename = useCallback(async (filename: string, newName: string) => {
    try {
      await sessionsApi.updateDomain(filename, { name: newName })
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree])

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return
    setCreating(true)
    try {
      await sessionsApi.createDomain(newName.trim(), newDesc.trim())
      setNewName('')
      setNewDesc('')
      setShowCreate(false)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    } finally {
      setCreating(false)
    }
  }, [newName, newDesc, refreshTree])

  const handlePromote = useCallback(async (filename: string) => {
    try {
      await sessionsApi.promoteDomain(filename)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree])

  const handleDrop = useCallback(async (item: DragItem, targetDomain: string) => {
    try {
      await sessionsApi.moveDomainSource({
        source_type: item.type === 'database' ? 'databases' : item.type === 'api' ? 'apis' : 'documents',
        source_name: item.name,
        from_domain: item.sourceDomain,
        to_domain: targetDomain,
        session_id: session?.session_id,
      })
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree, session])

  const activeDomains = session?.active_domains || []

  if (loading) {
    return (
      <div className="flex items-center justify-center py-4">
        <div className="w-4 h-4 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {error && (
        <p className="text-xs text-red-500 mb-2">{error}</p>
      )}

      {/* Domain tree */}
      {tree.length === 0 ? (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          No domains configured
        </p>
      ) : (
        <div className="space-y-0.5">
          {tree.map((node) => (
            <DomainTreeNodeView
              key={node.filename}
              node={node}
              depth={0}
              activeDomains={activeDomains}
              isAdmin={isAdmin}
              userId={userId}
              onToggle={handleToggle}
              onEdit={handleEdit}
              onDelete={(f) => setConfirmDelete(f)}
              onRename={handleRename}
              onPromote={handlePromote}
              onDrop={handleDrop}
            />
          ))}
        </div>
      )}

      {/* Delete confirmation */}
      {confirmDelete && (
        <div className="flex items-center gap-2 p-2 bg-red-50 dark:bg-red-900/20 rounded text-xs">
          <span className="text-red-600 dark:text-red-400 flex-1">
            Delete {confirmDelete}?
          </span>
          <button
            onClick={() => handleDelete(confirmDelete)}
            className="px-2 py-1 bg-red-600 text-white rounded hover:bg-red-700 text-xs"
          >
            Delete
          </button>
          <button
            onClick={() => setConfirmDelete(null)}
            className="px-2 py-1 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600 text-xs"
          >
            Cancel
          </button>
        </div>
      )}

      {/* YAML editor modal (inline) */}
      {editingFilename && (
        <div className="border border-gray-200 dark:border-gray-700 rounded p-2 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
              {editingFilename}
            </span>
            {yamlPath && (
              <span className="text-[10px] text-gray-400 truncate ml-2">{yamlPath}</span>
            )}
          </div>
          <textarea
            value={yamlContent}
            onChange={(e) => setYamlContent(e.target.value)}
            className="w-full h-40 text-xs font-mono p-2 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded resize-y"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setEditingFilename(null)}
              className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={handleSaveYaml}
              disabled={saving}
              className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}

      {/* Create domain form */}
      {showCreate ? (
        <div className="border border-gray-200 dark:border-gray-700 rounded p-2 space-y-2">
          <input
            autoFocus
            placeholder="Domain name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <input
            placeholder="Description (optional)"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => { setShowCreate(false); setNewName(''); setNewDesc('') }}
              className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400"
            >
              Cancel
            </button>
            <button
              onClick={handleCreate}
              disabled={creating || !newName.trim()}
              className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
            >
              {creating ? 'Creating...' : 'Create'}
            </button>
          </div>
        </div>
      ) : (
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1 text-xs text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
        >
          <PlusIcon className="w-3.5 h-3.5" /> New Domain
        </button>
      )}
    </div>
  )
}
