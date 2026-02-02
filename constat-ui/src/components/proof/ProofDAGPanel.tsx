// Proof DAG Panel - Floating panel for auditable mode fact resolution visualization

import { useState, useCallback } from 'react'
import { XMarkIcon, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline'

// Node status types matching server events
type NodeStatus = 'pending' | 'planning' | 'executing' | 'resolved' | 'failed' | 'blocked'

interface FactNode {
  id: string
  name: string
  description?: string
  status: NodeStatus
  value?: unknown
  source?: string
  confidence?: number
  tier?: number
  strategy?: string
  formula?: string
  reason?: string
  dependencies: string[]
  elapsed_ms?: number
}

interface ProofDAGPanelProps {
  isOpen: boolean
  onClose: () => void
  facts: Map<string, FactNode>
}

// Status symbols as per design doc
const STATUS_SYMBOLS: Record<NodeStatus, string> = {
  pending: '\u25CB',    // ○
  planning: '\u25D0',   // ◐
  executing: '\u25CF',  // ●
  resolved: '\u2713',   // ✓
  failed: '\u2717',     // ✗
  blocked: '\u2298',    // ⊘
}

const STATUS_COLORS: Record<NodeStatus, string> = {
  pending: 'text-gray-400',
  planning: 'text-yellow-500',
  executing: 'text-blue-500 animate-pulse',
  resolved: 'text-green-500',
  failed: 'text-red-500',
  blocked: 'text-orange-400',
}

function FactNodeCard({ node, onToggle, isExpanded }: {
  node: FactNode
  onToggle: () => void
  isExpanded: boolean
}) {
  return (
    <div
      className={`border rounded-lg p-3 cursor-pointer transition-all ${
        node.status === 'resolved'
          ? 'border-green-300 dark:border-green-700 bg-green-50 dark:bg-green-900/20'
          : node.status === 'failed'
          ? 'border-red-300 dark:border-red-700 bg-red-50 dark:bg-red-900/20'
          : node.status === 'executing'
          ? 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20'
          : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
      }`}
      onClick={onToggle}
    >
      <div className="flex items-center gap-2">
        <span className={`text-lg font-mono ${STATUS_COLORS[node.status]}`}>
          {STATUS_SYMBOLS[node.status]}
        </span>
        <span className="font-medium text-sm text-gray-900 dark:text-gray-100 truncate flex-1">
          {node.name}
        </span>
        {isExpanded ? (
          <ChevronDownIcon className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRightIcon className="w-4 h-4 text-gray-400" />
        )}
      </div>

      {isExpanded && (
        <div className="mt-3 space-y-2 text-xs">
          {node.description && (
            <div className="text-gray-600 dark:text-gray-400">{node.description}</div>
          )}
          {node.value !== undefined && (
            <div className="flex gap-2">
              <span className="text-gray-500">Value:</span>
              <span className="font-mono text-gray-900 dark:text-gray-100">
                {JSON.stringify(node.value)}
              </span>
            </div>
          )}
          {node.source && (
            <div className="flex gap-2">
              <span className="text-gray-500">Source:</span>
              <span className="text-gray-700 dark:text-gray-300">{node.source}</span>
            </div>
          )}
          {node.confidence !== undefined && (
            <div className="flex gap-2">
              <span className="text-gray-500">Confidence:</span>
              <span className="text-gray-700 dark:text-gray-300">
                {(node.confidence * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {node.tier !== undefined && (
            <div className="flex gap-2">
              <span className="text-gray-500">Tier:</span>
              <span className="text-gray-700 dark:text-gray-300">{node.tier}</span>
            </div>
          )}
          {node.formula && (
            <div className="flex gap-2">
              <span className="text-gray-500">Formula:</span>
              <span className="font-mono text-gray-700 dark:text-gray-300">{node.formula}</span>
            </div>
          )}
          {node.reason && (
            <div className="flex gap-2">
              <span className="text-gray-500">Reason:</span>
              <span className="text-red-600 dark:text-red-400">{node.reason}</span>
            </div>
          )}
          {node.dependencies.length > 0 && (
            <div className="flex gap-2">
              <span className="text-gray-500">Depends on:</span>
              <span className="text-gray-700 dark:text-gray-300">
                {node.dependencies.join(', ')}
              </span>
            </div>
          )}
          {node.elapsed_ms !== undefined && (
            <div className="flex gap-2">
              <span className="text-gray-500">Time:</span>
              <span className="text-gray-700 dark:text-gray-300">{node.elapsed_ms}ms</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function ProofDAGPanel({ isOpen, onClose, facts }: ProofDAGPanelProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())

  const toggleNode = useCallback((nodeId: string) => {
    setExpandedNodes((prev) => {
      const next = new Set(prev)
      if (next.has(nodeId)) {
        next.delete(nodeId)
      } else {
        next.add(nodeId)
      }
      return next
    })
  }, [])

  // Build dependency tree for visualization
  const nodes = Array.from(facts.values())
  const rootNodes = nodes.filter((n) => n.dependencies.length === 0)
  const childNodes = nodes.filter((n) => n.dependencies.length > 0)

  // Stats
  const resolvedCount = nodes.filter((n) => n.status === 'resolved').length
  const failedCount = nodes.filter((n) => n.status === 'failed').length
  const pendingCount = nodes.filter((n) => n.status !== 'resolved' && n.status !== 'failed').length

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl w-[600px] max-h-[80vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Proof Resolution
            </h2>
            <div className="flex gap-4 text-xs text-gray-500 mt-1">
              <span className="text-green-600">{resolvedCount} resolved</span>
              <span className="text-red-600">{failedCount} failed</span>
              <span className="text-blue-600">{pendingCount} in progress</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          >
            <XMarkIcon className="w-5 h-5" />
          </button>
        </div>

        {/* DAG Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {nodes.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <p>No facts being resolved yet.</p>
              <p className="text-sm mt-2">Facts will appear here as they are resolved.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Root nodes (no dependencies) */}
              {rootNodes.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                    Root Facts
                  </div>
                  <div className="space-y-2">
                    {rootNodes.map((node) => (
                      <FactNodeCard
                        key={node.id}
                        node={node}
                        isExpanded={expandedNodes.has(node.id)}
                        onToggle={() => toggleNode(node.id)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Dependent nodes */}
              {childNodes.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">
                    Derived Facts
                  </div>
                  <div className="space-y-2">
                    {childNodes.map((node) => (
                      <FactNodeCard
                        key={node.id}
                        node={node}
                        isExpanded={expandedNodes.has(node.id)}
                        onToggle={() => toggleNode(node.id)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

// Hook to manage proof facts from WebSocket events
export function useProofFacts() {
  const [facts, setFacts] = useState<Map<string, FactNode>>(new Map())
  const [isProving, setIsProving] = useState(false)

  const handleFactEvent = useCallback((eventType: string, data: Record<string, unknown>) => {
    const factName = data.fact_name as string
    if (!factName) return

    setFacts((prev) => {
      const next = new Map(prev)
      const existing = next.get(factName) || {
        id: factName,
        name: factName,
        status: 'pending' as NodeStatus,
        dependencies: [],
      }

      switch (eventType) {
        case 'fact_start':
          next.set(factName, {
            ...existing,
            description: data.fact_description as string | undefined,
            status: 'pending',
          })
          setIsProving(true)
          break

        case 'fact_planning':
          next.set(factName, {
            ...existing,
            status: 'planning',
          })
          break

        case 'fact_executing':
          next.set(factName, {
            ...existing,
            status: 'executing',
            formula: data.formula as string | undefined,
          })
          break

        case 'fact_resolved':
          next.set(factName, {
            ...existing,
            status: 'resolved',
            value: data.value,
            source: data.source as string | undefined,
            confidence: data.confidence as number | undefined,
            tier: data.tier as number | undefined,
            strategy: data.strategy as string | undefined,
            dependencies: (data.dependencies as string[]) || existing.dependencies,
            elapsed_ms: data.elapsed_ms as number | undefined,
          })
          break

        case 'fact_failed':
          next.set(factName, {
            ...existing,
            status: 'failed',
            reason: data.reason as string | undefined,
          })
          break

        case 'proof_complete':
          setIsProving(false)
          break
      }

      return next
    })
  }, [])

  const clearFacts = useCallback(() => {
    setFacts(new Map())
    setIsProving(false)
  }, [])

  return { facts, isProving, handleFactEvent, clearFacts }
}
