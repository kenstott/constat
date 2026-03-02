// TreeWidget - Hierarchy editor with indent/outdent, add/rename/delete nodes

import { useState, useCallback } from 'react'
import {
  PlusIcon,
  TrashIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  PencilIcon,
} from '@heroicons/react/24/outline'

interface TreeNode {
  id: string
  label: string
  children: TreeNode[]
  expanded?: boolean
}

interface TreeWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: unknown
  onAnswer: (freeform: string, structured: TreeNode[]) => void
}

let nodeIdCounter = 0
function nextId(): string {
  return `node_${++nodeIdCounter}`
}

function flattenTree(nodes: TreeNode[], depth = 0): string {
  return nodes
    .map(n => {
      const indent = '  '.repeat(depth)
      const childText = n.children.length > 0 ? flattenTree(n.children, depth + 1) : ''
      return `${indent}- ${n.label}${childText ? '\n' + childText : ''}`
    })
    .join('\n')
}

function cloneTree(nodes: TreeNode[]): TreeNode[] {
  return nodes.map(n => ({
    ...n,
    children: cloneTree(n.children),
  }))
}

export function TreeWidget({ config, structuredValue, onAnswer }: TreeWidgetProps) {
  const initialNodes = (config.nodes as TreeNode[]) || []

  const [tree, setTree] = useState<TreeNode[]>(() => {
    if (structuredValue && Array.isArray(structuredValue)) return structuredValue as TreeNode[]
    if (initialNodes.length) return cloneTree(initialNodes)
    return [{ id: nextId(), label: 'Root', children: [], expanded: true }]
  })

  const [editingId, setEditingId] = useState<string | null>(null)
  const [editText, setEditText] = useState('')

  const emitAnswer = useCallback((newTree: TreeNode[]) => {
    const freeform = flattenTree(newTree)
    onAnswer(freeform, newTree)
  }, [onAnswer])

  // Find and update a node in the tree
  const updateNode = (nodes: TreeNode[], id: string, updater: (n: TreeNode) => TreeNode): TreeNode[] => {
    return nodes.map(n => {
      if (n.id === id) return updater(n)
      return { ...n, children: updateNode(n.children, id, updater) }
    })
  }

  // Remove a node from the tree
  const removeNode = (nodes: TreeNode[], id: string): TreeNode[] => {
    return nodes
      .filter(n => n.id !== id)
      .map(n => ({ ...n, children: removeNode(n.children, id) }))
  }

  // Add child to a node
  const addChild = (parentId: string) => {
    const newNode: TreeNode = { id: nextId(), label: 'New item', children: [], expanded: false }
    const newTree = updateNode(tree, parentId, n => ({
      ...n,
      expanded: true,
      children: [...n.children, newNode],
    }))
    setTree(newTree)
    emitAnswer(newTree)
    // Auto-edit the new node
    setEditingId(newNode.id)
    setEditText(newNode.label)
  }

  // Add sibling at root level
  const addRootNode = () => {
    const newNode: TreeNode = { id: nextId(), label: 'New item', children: [], expanded: false }
    const newTree = [...tree, newNode]
    setTree(newTree)
    emitAnswer(newTree)
    setEditingId(newNode.id)
    setEditText(newNode.label)
  }

  const startEdit = (id: string, label: string) => {
    setEditingId(id)
    setEditText(label)
  }

  const commitEdit = () => {
    if (!editingId || !editText.trim()) return
    const newTree = updateNode(tree, editingId, n => ({ ...n, label: editText.trim() }))
    setTree(newTree)
    setEditingId(null)
    setEditText('')
    emitAnswer(newTree)
  }

  const deleteNode = (id: string) => {
    const newTree = removeNode(tree, id)
    setTree(newTree)
    emitAnswer(newTree)
  }

  const toggleExpand = (id: string) => {
    const newTree = updateNode(tree, id, n => ({ ...n, expanded: !n.expanded }))
    setTree(newTree)
  }

  // Indent: move node to be a child of the previous sibling
  const indentNode = (nodes: TreeNode[], id: string): TreeNode[] => {
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].id === id && i > 0) {
        const node = nodes[i]
        const prevSibling = nodes[i - 1]
        const newNodes = nodes.filter((_, idx) => idx !== i)
        return newNodes.map(n =>
          n.id === prevSibling.id
            ? { ...n, expanded: true, children: [...n.children, node] }
            : n
        )
      }
      const updatedChildren = indentNode(nodes[i].children, id)
      if (updatedChildren !== nodes[i].children) {
        return nodes.map((n, idx) =>
          idx === i ? { ...n, children: updatedChildren } : n
        )
      }
    }
    return nodes
  }

  // Outdent: move node to be a sibling of its parent
  const outdentNode = (nodes: TreeNode[], id: string, _parent?: TreeNode[]): TreeNode[] => {
    for (let i = 0; i < nodes.length; i++) {
      const childIndex = nodes[i].children.findIndex(c => c.id === id)
      if (childIndex >= 0) {
        const child = nodes[i].children[childIndex]
        const newChildren = nodes[i].children.filter((_, idx) => idx !== childIndex)
        const parentIndex = i
        const result = [...nodes]
        result[parentIndex] = { ...result[parentIndex], children: newChildren }
        result.splice(parentIndex + 1, 0, child)
        return result
      }
      const updatedChildren = outdentNode(nodes[i].children, id, nodes)
      if (updatedChildren !== nodes[i].children) {
        return nodes.map((n, idx) =>
          idx === i ? { ...n, children: updatedChildren } : n
        )
      }
    }
    return nodes
  }

  const handleIndent = (id: string) => {
    const newTree = indentNode(tree, id)
    setTree(newTree)
    emitAnswer(newTree)
  }

  const handleOutdent = (id: string) => {
    const newTree = outdentNode(tree, id)
    setTree(newTree)
    emitAnswer(newTree)
  }

  const renderNode = (node: TreeNode, depth: number) => (
    <div key={node.id}>
      <div
        className="flex items-center gap-1 py-1 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded transition-colors group"
        style={{ paddingLeft: `${depth * 20 + 8}px` }}
      >
        {/* Expand toggle */}
        <button
          onClick={() => toggleExpand(node.id)}
          className="p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
        >
          {node.children.length > 0 ? (
            node.expanded ? (
              <ChevronDownIcon className="w-3.5 h-3.5" />
            ) : (
              <ChevronRightIcon className="w-3.5 h-3.5" />
            )
          ) : (
            <span className="w-3.5 h-3.5 inline-block" />
          )}
        </button>

        {/* Label or edit input */}
        {editingId === node.id ? (
          <input
            type="text"
            value={editText}
            onChange={(e) => setEditText(e.target.value)}
            onKeyDown={(e) => {
              e.stopPropagation()
              if (e.key === 'Enter') commitEdit()
              if (e.key === 'Escape') setEditingId(null)
            }}
            onKeyUp={(e) => e.stopPropagation()}
            onBlur={commitEdit}
            className="flex-1 px-2 py-0.5 text-sm rounded border border-primary-500 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:ring-1 focus:ring-primary-500"
            autoFocus
          />
        ) : (
          <span
            className="flex-1 text-sm text-gray-700 dark:text-gray-300 cursor-default"
            onDoubleClick={() => startEdit(node.id, node.label)}
          >
            {node.label}
          </span>
        )}

        {/* Action buttons (visible on hover) */}
        <div className="hidden group-hover:flex items-center gap-0.5 mr-2">
          <button
            onClick={() => startEdit(node.id, node.label)}
            className="p-0.5 text-gray-400 hover:text-primary-500 transition-colors"
            title="Rename"
          >
            <PencilIcon className="w-3 h-3" />
          </button>
          <button
            onClick={() => addChild(node.id)}
            className="p-0.5 text-gray-400 hover:text-green-500 transition-colors"
            title="Add child"
          >
            <PlusIcon className="w-3 h-3" />
          </button>
          <button
            onClick={() => handleIndent(node.id)}
            className="p-0.5 text-gray-400 hover:text-blue-500 transition-colors"
            title="Indent"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7" />
            </svg>
          </button>
          <button
            onClick={() => handleOutdent(node.id)}
            className="p-0.5 text-gray-400 hover:text-blue-500 transition-colors"
            title="Outdent"
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7" />
            </svg>
          </button>
          <button
            onClick={() => deleteNode(node.id)}
            className="p-0.5 text-gray-400 hover:text-red-500 transition-colors"
            title="Delete"
          >
            <TrashIcon className="w-3 h-3" />
          </button>
        </div>
      </div>
      {node.expanded && node.children.map(child => renderNode(child, depth + 1))}
    </div>
  )

  return (
    <div className="space-y-2">
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-2 max-h-80 overflow-y-auto">
        {tree.map(node => renderNode(node, 0))}
      </div>
      <button
        onClick={addRootNode}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/30 rounded-lg transition-colors"
      >
        <PlusIcon className="w-3.5 h-3.5" />
        Add item
      </button>
      <div className="text-xs text-gray-400 dark:text-gray-500">
        Double-click to rename. Hover for actions.
      </div>
    </div>
  )
}
