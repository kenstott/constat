// Copyright (c) 2025 Kenneth Stott
// Canary: e670f707-3912-4f1d-a662-67838b0767e9
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Domain Panel — recursive tree view with toggle, expand/collapse, CRUD, drag-and-drop

import { useEffect, useState, useCallback, useRef } from 'react'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  PlusIcon,
  ArrowUpOnSquareIcon,
  Square3Stack3DIcon,
  MapIcon,
} from '@heroicons/react/24/outline'
import { useMutation } from '@apollo/client'
import { useSessionContext } from '@/contexts/SessionContext'
import { useArtifactContext } from '@/contexts/ArtifactContext'
import { useAuth } from '@/contexts/AuthContext'
import type { DomainTreeNode } from '@/types/api'
import { apolloClient } from '@/graphql/client'
import { SET_ACTIVE_DOMAINS } from '@/graphql/operations/sessions'
import {
  DOMAIN_TREE_QUERY,
  DOMAIN_CONTENT_QUERY,
  UPDATE_DOMAIN_CONTENT,
  DELETE_DOMAIN,
  UPDATE_DOMAIN,
  PROMOTE_DOMAIN,
  CREATE_DOMAIN,
  MOVE_DOMAIN_SOURCE,
  toDomainTreeNode,
  toDomainContent,
} from '@/graphql/operations/domains'
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline'
import { MermaidBlock } from '@/components/proof/MermaidBlock'

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
  onCreateChild,
  onShowComposition,
  onShowDiagram,
  togglingDomain,
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
  onCreateChild: (parentNode: DomainTreeNode) => void
  onShowComposition: (node: DomainTreeNode) => void
  onShowDiagram: (node: DomainTreeNode) => void
  togglingDomain: string | null
}) {
  const [expanded, setExpanded] = useState(depth < 2)
  const [menuOpen, setMenuOpen] = useState(false)
  const [renaming, setRenaming] = useState(false)
  const [renameValue, setRenameValue] = useState(node.name)
  const [dragOver, setDragOver] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const hasChildren = node.children.length > 0
  const isSystem = node.filename === 'root'
  const isSynthetic = node.filename === 'root' || node.filename === 'user'
  const isActive = isSystem || activeDomains.includes(node.filename)
  const resourceCount =
    node.databases.length + node.apis.length + node.documents.length
  const canModify = isAdmin || (node.tier !== 'system' && (!node.owner || node.owner === userId))
  const showLock = !canModify && !isSynthetic
  const canPromote = node.tier === 'user' && !isSynthetic && (isAdmin || !node.owner || node.owner === userId)

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

        {/* Checkbox — only system (root) is always active */}
        {togglingDomain === node.filename ? (
          <div className="w-3.5 h-3.5 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
        ) : (
          <input
            type="checkbox"
            checked={isActive}
            disabled={isSystem || togglingDomain !== null}
            onChange={() => !isSystem && onToggle(node.filename)}
            className={`w-3.5 h-3.5 rounded border-gray-300 text-primary-600 focus:ring-primary-500 ${isSystem || togglingDomain !== null ? 'opacity-50 cursor-default' : 'cursor-pointer'}`}
          />
        )}

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
              {canModify && !isSynthetic && (
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
              {canModify && (
                <button
                  onClick={() => {
                    setMenuOpen(false)
                    onCreateChild(node)
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                >
                  <PlusIcon className="w-3 h-3" /> Create Child
                </button>
              )}
              <button
                onClick={() => {
                  setMenuOpen(false)
                  onShowComposition(node)
                }}
                className="w-full text-left px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
              >
                <Square3Stack3DIcon className="w-3 h-3" /> Composition
              </button>
              <button
                onClick={() => {
                  setMenuOpen(false)
                  onShowDiagram(node)
                }}
                className="w-full text-left px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
              >
                <MapIcon className="w-3 h-3" /> Diagram
              </button>
              {canPromote && (
                <button
                  onClick={() => {
                    setMenuOpen(false)
                    onPromote(node.filename)
                  }}
                  className="w-full text-left px-3 py-1.5 text-xs text-blue-600 dark:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center gap-2"
                >
                  <ArrowUpOnSquareIcon className="w-3 h-3" /> Move to Root
                </button>
              )}
              {canModify && !isSynthetic && (
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

      {/* Description */}
      {expanded && node.description && !isSynthetic && (
        <p
          className="text-[11px] text-gray-500 dark:text-gray-400 truncate"
          style={{ paddingLeft: depth * 16 + 20 }}
          title={node.description}
        >
          {node.description}
        </p>
      )}

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
          onCreateChild={onCreateChild}
          onShowComposition={onShowComposition}
          onShowDiagram={onShowDiagram}
          togglingDomain={togglingDomain}
        />
      ))}
    </div>
  )
}

export default function DomainPanel() {
  const { session, updateSession } = useSessionContext()
  const { fetchDataSources, fetchPromptContext, promptContext } = useArtifactContext()
  const { isAdmin, userId } = useAuth()
  const [setActiveDomainsMutation] = useMutation(SET_ACTIVE_DOMAINS)
  const [tree, setTree] = useState<DomainTreeNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  // YAML editor
  const [editingFilename, setEditingFilename] = useState<string | null>(null)
  const [yamlContent, setYamlContent] = useState('')
  const [yamlPath, setYamlPath] = useState('')
  const [saving, setSaving] = useState(false)
  // Domain toggle loading
  const [togglingDomain, setTogglingDomain] = useState<string | null>(null)
  // Delete confirmation
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  // Create child domain
  const [createChildParent, setCreateChildParent] = useState<DomainTreeNode | null>(null)
  const [childName, setChildName] = useState('')
  const [childDesc, setChildDesc] = useState('')
  const [creatingChild, setCreatingChild] = useState(false)
  const activeDomains = session?.active_domains || []
  // Save session as domain
  const [promoting, setPromoting] = useState(false)
  const [promoteName, setPromoteName] = useState('')
  const [promoteDesc, setPromoteDesc] = useState('')
  const [promoteSubmitting, setPromoteSubmitting] = useState(false)
  // Composition modal
  const [compositionNode, setCompositionNode] = useState<DomainTreeNode | null>(null)
  // Diagram modal
  const [diagramNode, setDiagramNode] = useState<DomainTreeNode | null>(null)
  const [sessionDiagram, setSessionDiagram] = useState(false)
  const [diagramActiveOnly, setDiagramActiveOnly] = useState(false)
  const [compositionDomains, setCompositionDomains] = useState<string[]>([])
  const [savingComposition, setSavingComposition] = useState(false)

  // Build a Mermaid composition diagram centered on one or more focus nodes.
  // When allowedNodes is provided, only those nodes and edges between them are shown.
  const buildCompositionDiagram = useCallback((
    nodes: DomainTreeNode[],
    focusFilenames: string[],
    allowedNodes?: Set<string>,
  ): string => {
    // Collect all composition edges and display names
    const allEdges: [string, string][] = []
    const names = new Map<string, string>()
    const walk = (n: DomainTreeNode) => {
      names.set(n.filename, n.filename === 'root' ? 'System' : n.name)
      if (n.domains) {
        for (const child of n.domains) {
          allEdges.push([n.filename, child])
        }
      }
      n.children.forEach(walk)
    }
    nodes.forEach(walk)

    // Filter edges to allowed nodes when constrained
    const edges = allowedNodes
      ? allEdges.filter(([from, to]) => allowedNodes.has(from) && allowedNodes.has(to))
      : allEdges

    // BFS both directions from focus nodes
    const forward = new Map<string, string[]>()
    const backward = new Map<string, string[]>()
    for (const [from, to] of edges) {
      if (!forward.has(from)) forward.set(from, [])
      forward.get(from)!.push(to)
      if (!backward.has(to)) backward.set(to, [])
      backward.get(to)!.push(from)
    }

    const reachable = new Set<string>()
    const queue = [...focusFilenames.filter((f) => !allowedNodes || allowedNodes.has(f))]
    for (const f of queue) reachable.add(f)
    while (queue.length > 0) {
      const cur = queue.shift()!
      for (const next of forward.get(cur) || []) {
        if (!reachable.has(next)) { reachable.add(next); queue.push(next) }
      }
      for (const next of backward.get(cur) || []) {
        if (!reachable.has(next)) { reachable.add(next); queue.push(next) }
      }
    }

    // Build Mermaid lines
    const lines = ['graph TD']
    for (const id of reachable) {
      const label = names.get(id) || id
      const safeId = id.replace(/[^a-zA-Z0-9_-]/g, '_')
      lines.push(`  ${safeId}["${label}"]`)
    }
    const edgeSet = new Set<string>()
    for (const [from, to] of edges) {
      if (reachable.has(from) && reachable.has(to)) {
        const key = `${from}|${to}`
        if (!edgeSet.has(key)) {
          edgeSet.add(key)
          const safeFrom = from.replace(/[^a-zA-Z0-9_-]/g, '_')
          const safeTo = to.replace(/[^a-zA-Z0-9_-]/g, '_')
          lines.push(`  ${safeFrom} --> ${safeTo}`)
        }
      }
    }
    // Highlight focus nodes
    const focusSet = new Set(focusFilenames)
    for (const id of reachable) {
      if (focusSet.has(id)) {
        const safeId = id.replace(/[^a-zA-Z0-9_-]/g, '_')
        lines.push(`  style ${safeId} fill:#3B82F6,color:#fff`)
      }
    }

    return lines.join('\n')
  }, [])

  // Collect all domain filenames from tree (excluding synthetic nodes)
  const collectFilenames = useCallback((nodes: DomainTreeNode[]): string[] => {
    const result: string[] = []
    const walk = (n: DomainTreeNode) => {
      if (n.filename !== 'root' && n.filename !== 'user') result.push(n.filename)
      n.children.forEach(walk)
    }
    nodes.forEach(walk)
    return result
  }, [])

  const allDomainFilenames = collectFilenames(tree)

  const refreshTree = useCallback(async () => {
    try {
      const { data } = await apolloClient.query({ query: DOMAIN_TREE_QUERY, fetchPolicy: 'network-only' })
      setTree(data.domainTree.map(toDomainTreeNode))
    } catch (err) {
      setError(String(err))
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    apolloClient.query({ query: DOMAIN_TREE_QUERY, fetchPolicy: 'network-only' })
      .then(({ data }) => {
        if (!cancelled) setTree(data.domainTree.map(toDomainTreeNode))
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
      setTogglingDomain(filename)
      const current = session.active_domains || []
      const isSelected = current.includes(filename)
      const newDomains = isSelected
        ? current.filter((d) => d !== filename)
        : [...current, filename]

      try {
        await setActiveDomainsMutation({
          variables: { sessionId: session.session_id, domains: newDomains },
        })
        updateSession({ active_domains: newDomains })
        await fetchDataSources(session.session_id)
        // Refresh prompt context (domain activation may update session prompt)
        await fetchPromptContext(session.session_id)
      } catch (err: unknown) {
        console.error('Failed to toggle domain:', err)
        if (err && typeof err === 'object' && 'message' in err) {
          setError((err as { message: string }).message)
        }
      } finally {
        setTogglingDomain(null)
      }
    },
    [session, updateSession, fetchDataSources],
  )

  const handleEdit = useCallback(async (filename: string) => {
    try {
      const { data } = await apolloClient.query({ query: DOMAIN_CONTENT_QUERY, variables: { filename }, fetchPolicy: 'network-only' })
      const result = toDomainContent(data.domainContent)
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
      await apolloClient.mutate({ mutation: UPDATE_DOMAIN_CONTENT, variables: { filename: editingFilename, content: yamlContent } })
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
      await apolloClient.mutate({ mutation: DELETE_DOMAIN, variables: { filename } })
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
      await apolloClient.mutate({ mutation: UPDATE_DOMAIN, variables: { filename, name: newName } })
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree])

  const handlePromote = useCallback(async (filename: string) => {
    try {
      await apolloClient.mutate({ mutation: PROMOTE_DOMAIN, variables: { filename } })
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree])

  const handleCreateChild = useCallback(async () => {
    if (!childName.trim() || !createChildParent) return
    setCreatingChild(true)
    try {
      // Initial composition = active child domains under the parent
      const activeChildren = createChildParent.children
        .filter((c) => activeDomains.includes(c.filename))
        .map((c) => c.filename)
      await apolloClient.mutate({ mutation: CREATE_DOMAIN, variables: {
        name: childName.trim(),
        description: childDesc.trim(),
        parentDomain: createChildParent.filename,
        initialDomains: activeChildren,
      } })
      setChildName('')
      setChildDesc('')
      setCreateChildParent(null)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    } finally {
      setCreatingChild(false)
    }
  }, [childName, childDesc, createChildParent, activeDomains, refreshTree])

  const handleSaveComposition = useCallback(async () => {
    if (!compositionNode) return
    setSavingComposition(true)
    try {
      // Load current YAML, update domains list, save back
      const { data } = await apolloClient.query({ query: DOMAIN_CONTENT_QUERY, variables: { filename: compositionNode.filename }, fetchPolicy: 'network-only' })
      const result = toDomainContent(data.domainContent)
      const lines = result.content.split('\n')
      // Remove existing domains: block and rebuild
      let newLines: string[] = []
      let inDomains = false
      for (const line of lines) {
        if (/^domains:/.test(line)) {
          inDomains = true
          continue
        }
        if (inDomains && /^\s*-\s/.test(line)) continue
        if (inDomains) inDomains = false
        newLines.push(line)
      }
      // Append new domains list
      if (compositionDomains.length > 0) {
        newLines.push('domains:')
        for (const d of compositionDomains) {
          newLines.push(`  - ${d}`)
        }
      }
      await apolloClient.mutate({ mutation: UPDATE_DOMAIN_CONTENT, variables: { filename: compositionNode.filename, content: newLines.join('\n') } })
      setCompositionNode(null)
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    } finally {
      setSavingComposition(false)
    }
  }, [compositionNode, compositionDomains, refreshTree])

  const handlePromoteSession = useCallback(async () => {
    if (!promoteName.trim()) return
    setPromoteSubmitting(true)
    try {
      const sessionPrompt = promptContext?.systemPrompt || ''
      await apolloClient.mutate({ mutation: CREATE_DOMAIN, variables: {
        name: promoteName.trim(),
        description: promoteDesc.trim(),
        parentDomain: 'user',
        initialDomains: activeDomains,
        systemPrompt: sessionPrompt,
      } })
      setPromoting(false)
      setPromoteName('')
      setPromoteDesc('')
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    } finally {
      setPromoteSubmitting(false)
    }
  }, [promoteName, promoteDesc, activeDomains, refreshTree])

  const handleDrop = useCallback(async (item: DragItem, targetDomain: string) => {
    try {
      await apolloClient.mutate({ mutation: MOVE_DOMAIN_SOURCE, variables: {
        sourceType: item.type === 'database' ? 'databases' : item.type === 'api' ? 'apis' : 'documents',
        sourceName: item.name,
        fromDomain: item.sourceDomain,
        toDomain: targetDomain,
        sessionId: session?.session_id,
      } })
      await refreshTree()
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setError((err as { message: string }).message)
      }
    }
  }, [refreshTree, session])

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

      {/* Save session as domain */}
      {!promoting ? (
        <div className="flex items-center gap-3">
          <button
            onClick={() => setPromoting(true)}
            className="flex items-center gap-1.5 text-xs text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300"
          >
            <ArrowDownTrayIcon className="w-3.5 h-3.5" />
            Save Session as Domain
          </button>
          {activeDomains.length > 0 && (
            <button
              onClick={() => setSessionDiagram(true)}
              title="Session composition diagram"
              className="text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
            >
              <MapIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      ) : (
        <div className="border border-gray-200 dark:border-gray-700 rounded p-2 space-y-2">
          <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
            Save Session as Domain
          </span>
          <input
            autoFocus
            placeholder="Domain name"
            value={promoteName}
            onChange={(e) => setPromoteName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePromoteSession()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <input
            placeholder="Description (optional)"
            value={promoteDesc}
            onChange={(e) => setPromoteDesc(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handlePromoteSession()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => { setPromoting(false); setPromoteName(''); setPromoteDesc('') }}
              className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400"
            >
              Cancel
            </button>
            <button
              onClick={handlePromoteSession}
              disabled={promoteSubmitting || !promoteName.trim()}
              className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
            >
              {promoteSubmitting ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
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
              onCreateChild={(n) => { setCreateChildParent(n); setChildName(''); setChildDesc('') }}
              onShowComposition={(n) => { setCompositionNode(n); setCompositionDomains(n.domains || []) }}
              onShowDiagram={(n) => setDiagramNode(n)}
              togglingDomain={togglingDomain}
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

      {/* Create child domain form */}
      {createChildParent && (
        <div className="border border-gray-200 dark:border-gray-700 rounded p-2 space-y-2">
          <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
            New child of <span className="text-primary-600 dark:text-primary-400">{createChildParent.name}</span>
          </span>
          <input
            autoFocus
            placeholder="Domain name"
            value={childName}
            onChange={(e) => setChildName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateChild()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <input
            placeholder="Description (optional)"
            value={childDesc}
            onChange={(e) => setChildDesc(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreateChild()}
            className="w-full text-xs px-2 py-1.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded outline-none focus:ring-1 focus:ring-primary-500"
          />
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setCreateChildParent(null)}
              className="px-2 py-1 text-xs text-gray-600 dark:text-gray-400"
            >
              Cancel
            </button>
            <button
              onClick={handleCreateChild}
              disabled={creatingChild || !childName.trim()}
              className="px-3 py-1 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
            >
              {creatingChild ? 'Creating...' : 'Create'}
            </button>
          </div>
        </div>
      )}

      {/* Composition modal */}
      {compositionNode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-96 max-h-[60vh] overflow-y-auto p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                Composition: {compositionNode.name}
              </h3>
              <button
                onClick={() => setCompositionNode(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-lg leading-none"
              >
                &times;
              </button>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Select which domains this domain composes:
            </p>
            <div className="space-y-1 max-h-60 overflow-y-auto">
              {allDomainFilenames
                .filter((f) => f !== compositionNode.filename)
                .map((f) => (
                  <label
                    key={f}
                    className="flex items-center gap-2 text-xs px-2 py-1.5 rounded hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={compositionDomains.includes(f)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setCompositionDomains((prev) => [...prev, f])
                        } else {
                          setCompositionDomains((prev) => prev.filter((d) => d !== f))
                        }
                      }}
                      className="w-3.5 h-3.5 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <Square3Stack3DIcon className="w-3.5 h-3.5 text-gray-400" />
                    <span className="text-gray-800 dark:text-gray-200">{f}</span>
                  </label>
                ))}
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setCompositionNode(null)}
                className="px-3 py-1.5 text-xs bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
              >
                Cancel
              </button>
              <button
                onClick={handleSaveComposition}
                disabled={savingComposition}
                className="px-3 py-1.5 text-xs bg-primary-600 text-white rounded hover:bg-primary-700 disabled:opacity-50"
              >
                {savingComposition ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Composition diagram modal */}
      {diagramNode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-[600px] max-h-[80vh] overflow-auto p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                Composition: {diagramNode.name}
              </h3>
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={diagramActiveOnly}
                    onChange={(e) => setDiagramActiveOnly(e.target.checked)}
                    className="w-3 h-3 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  Active only
                </label>
                <button
                  onClick={() => { setDiagramNode(null); setDiagramActiveOnly(false) }}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-lg leading-none"
                >
                  &times;
                </button>
              </div>
            </div>
            <MermaidBlock chart={buildCompositionDiagram(
              tree,
              [diagramNode.filename],
              diagramActiveOnly ? new Set(['root', ...activeDomains]) : undefined,
            )} />
          </div>
        </div>
      )}

      {/* Session composition diagram modal */}
      {sessionDiagram && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-[600px] max-h-[80vh] overflow-auto p-4 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                Session Composition
              </h3>
              <button
                onClick={() => setSessionDiagram(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-lg leading-none"
              >
                &times;
              </button>
            </div>
            <MermaidBlock chart={buildCompositionDiagram(
              tree,
              ['root', ...activeDomains],
              new Set(['root', ...activeDomains]),
            )} />
          </div>
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

    </div>
  )
}
