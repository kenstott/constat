// Proof DAG Panel - Floating panel for auditable mode fact resolution visualization
// Uses d3-dag for proper directed acyclic graph layout

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import * as d3dag from 'd3-dag'

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
  pending: '#9CA3AF',     // gray-400
  planning: '#EAB308',    // yellow-500
  executing: '#3B82F6',   // blue-500
  resolved: '#22C55E',    // green-500
  failed: '#EF4444',      // red-500
  blocked: '#FB923C',     // orange-400
}

const STATUS_BG_COLORS: Record<NodeStatus, string> = {
  pending: '#F3F4F6',     // gray-100
  planning: '#FEF9C3',    // yellow-100
  executing: '#DBEAFE',   // blue-100
  resolved: '#DCFCE7',    // green-100
  failed: '#FEE2E2',      // red-100
  blocked: '#FFEDD5',     // orange-100
}

// Node dimensions
const NODE_WIDTH = 160
const NODE_HEIGHT = 50
const NODE_RADIUS = 8

interface DagNode {
  id: string
  parentIds: string[]
  data: FactNode
}

function NodeTooltip({ node, position }: { node: FactNode; position: { x: number; y: number } }) {
  return (
    <div
      className="fixed z-[100] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-3 max-w-xs"
      style={{
        left: position.x + 10,
        top: position.y + 10,
      }}
    >
      <div className="space-y-2 text-xs">
        <div className="font-medium text-gray-900 dark:text-gray-100">{node.name}</div>
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
    </div>
  )
}

export function ProofDAGPanel({ isOpen, onClose, facts }: ProofDAGPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [hoveredNode, setHoveredNode] = useState<{ node: FactNode; position: { x: number; y: number } } | null>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 })

  // Convert facts Map to DAG structure
  const dagData = useMemo(() => {
    const nodes = Array.from(facts.values())
    if (nodes.length === 0) return null

    // Build node list with parent references (dependencies become parents in top-down view)
    // In our case, if A depends on B, then B is a parent of A (B must resolve before A)
    const dagNodes: DagNode[] = nodes.map(node => ({
      id: node.id,
      // Filter dependencies to only include nodes that exist in our facts
      parentIds: node.dependencies.filter(dep => facts.has(dep)),
      data: node,
    }))

    return dagNodes
  }, [facts])

  // Compute DAG layout
  const layout = useMemo(() => {
    if (!dagData || dagData.length === 0) return null

    try {
      // Create DAG using stratify (handles parent references)
      const stratify = d3dag.graphStratify()
      const graph = stratify(dagData)

      // Use Sugiyama layout for layered top-down display
      const layouter = d3dag.sugiyama()
        .nodeSize(() => [NODE_WIDTH + 40, NODE_HEIGHT + 60])

      // Apply layout
      layouter(graph)

      // Calculate bounds
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      for (const node of Array.from(graph.nodes())) {
        minX = Math.min(minX, node.x - NODE_WIDTH / 2)
        maxX = Math.max(maxX, node.x + NODE_WIDTH / 2)
        minY = Math.min(minY, node.y - NODE_HEIGHT / 2)
        maxY = Math.max(maxY, node.y + NODE_HEIGHT / 2)
      }

      const padding = 40
      const width = Math.max(600, maxX - minX + padding * 2)
      const height = Math.max(300, maxY - minY + padding * 2)
      const offsetX = -minX + padding
      const offsetY = -minY + padding

      return { graph, width, height, offsetX, offsetY }
    } catch (e) {
      console.error('DAG layout error:', e)
      return null
    }
  }, [dagData])

  // Update dimensions when layout changes
  useEffect(() => {
    if (layout) {
      setDimensions({ width: layout.width, height: layout.height })
    }
  }, [layout])

  // Stats
  const nodes = Array.from(facts.values())
  const resolvedCount = nodes.filter((n) => n.status === 'resolved').length
  const failedCount = nodes.filter((n) => n.status === 'failed').length
  const pendingCount = nodes.filter((n) => n.status !== 'resolved' && n.status !== 'failed').length

  if (!isOpen) return null

  // Render edge path with curve
  const renderEdge = (
    sourceX: number,
    sourceY: number,
    targetX: number,
    targetY: number,
    sourceStatus: NodeStatus,
    targetStatus: NodeStatus,
    key: string
  ) => {
    // Create curved path from source bottom to target top
    const startY = sourceY + NODE_HEIGHT / 2
    const endY = targetY - NODE_HEIGHT / 2
    const midY = (startY + endY) / 2

    const path = `M ${sourceX} ${startY} C ${sourceX} ${midY}, ${targetX} ${midY}, ${targetX} ${endY}`

    // Determine edge color based on resolution status
    const isResolved = sourceStatus === 'resolved'
    const strokeColor = isResolved ? STATUS_COLORS.resolved : '#CBD5E1'

    return (
      <g key={key}>
        {/* Edge path */}
        <path
          d={path}
          fill="none"
          stroke={strokeColor}
          strokeWidth={2}
          markerEnd="url(#arrowhead)"
          className={isResolved ? '' : 'opacity-50'}
        />
        {/* Animated flow indicator for executing edges */}
        {(sourceStatus === 'executing' || targetStatus === 'executing') && (
          <path
            d={path}
            fill="none"
            stroke={STATUS_COLORS.executing}
            strokeWidth={3}
            strokeDasharray="8 8"
            className="animate-dash"
          />
        )}
      </g>
    )
  }

  // Render node
  const renderNode = (
    nodeData: FactNode,
    x: number,
    y: number
  ) => {
    const status = nodeData.status
    const bgColor = STATUS_BG_COLORS[status]
    const borderColor = STATUS_COLORS[status]

    return (
      <g
        key={nodeData.id}
        transform={`translate(${x - NODE_WIDTH / 2}, ${y - NODE_HEIGHT / 2})`}
        className="cursor-pointer"
        onMouseEnter={(e) => {
          setHoveredNode({
            node: nodeData,
            position: { x: e.clientX, y: e.clientY }
          })
        }}
        onMouseLeave={() => setHoveredNode(null)}
      >
        {/* Node rectangle */}
        <rect
          width={NODE_WIDTH}
          height={NODE_HEIGHT}
          rx={NODE_RADIUS}
          ry={NODE_RADIUS}
          fill={bgColor}
          stroke={borderColor}
          strokeWidth={2}
          className={status === 'executing' ? 'animate-pulse' : ''}
        />
        {/* Status symbol */}
        <text
          x={12}
          y={NODE_HEIGHT / 2 + 5}
          fill={borderColor}
          fontSize={16}
          fontFamily="monospace"
        >
          {STATUS_SYMBOLS[status]}
        </text>
        {/* Node name */}
        <text
          x={32}
          y={NODE_HEIGHT / 2 + 4}
          fill="#1F2937"
          fontSize={12}
          fontWeight={500}
          className="select-none"
        >
          {nodeData.name.length > 14 ? nodeData.name.slice(0, 12) + '...' : nodeData.name}
        </text>
      </g>
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30">
      <div className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl max-w-[90vw] max-h-[85vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Proof Resolution DAG
            </h2>
            <div className="flex gap-4 text-xs text-gray-500 mt-1">
              <span className="flex items-center gap-1">
                <span className="text-green-600">{STATUS_SYMBOLS.resolved}</span>
                {resolvedCount} resolved
              </span>
              <span className="flex items-center gap-1">
                <span className="text-red-600">{STATUS_SYMBOLS.failed}</span>
                {failedCount} failed
              </span>
              <span className="flex items-center gap-1">
                <span className="text-blue-600">{STATUS_SYMBOLS.executing}</span>
                {pendingCount} in progress
              </span>
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
        <div className="flex-1 overflow-auto p-4">
          {nodes.length === 0 ? (
            <div className="text-center text-gray-500 py-8 min-w-[500px]">
              <p className="text-lg">{STATUS_SYMBOLS.pending} No facts being resolved yet.</p>
              <p className="text-sm mt-2">Facts will appear here as they are resolved.</p>
            </div>
          ) : layout ? (
            <svg
              ref={svgRef}
              width={dimensions.width}
              height={dimensions.height}
              className="block mx-auto"
            >
              {/* Definitions for markers */}
              <defs>
                <marker
                  id="arrowhead"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill="#94A3B8"
                  />
                </marker>
                <marker
                  id="arrowhead-resolved"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill={STATUS_COLORS.resolved}
                  />
                </marker>
              </defs>

              {/* Render edges first (below nodes) */}
              <g className="edges">
                {Array.from(layout.graph.links()).map((link) => {
                  const source = link.source
                  const target = link.target
                  const sourceData = (source.data as DagNode).data
                  const targetData = (target.data as DagNode).data

                  return renderEdge(
                    source.x + layout.offsetX,
                    source.y + layout.offsetY,
                    target.x + layout.offsetX,
                    target.y + layout.offsetY,
                    sourceData.status,
                    targetData.status,
                    `${source.data.id}-${target.data.id}`
                  )
                })}
              </g>

              {/* Render nodes */}
              <g className="nodes">
                {Array.from(layout.graph.nodes()).map((node) => {
                  const nodeData = (node.data as DagNode).data
                  return renderNode(
                    nodeData,
                    node.x + layout.offsetX,
                    node.y + layout.offsetY
                  )
                })}
              </g>
            </svg>
          ) : (
            <div className="text-center text-gray-500 py-8 min-w-[500px]">
              <p>Building graph layout...</p>
            </div>
          )}
        </div>

        {/* Legend */}
        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex flex-wrap gap-4 text-xs">
          {Object.entries(STATUS_SYMBOLS).map(([status, symbol]) => (
            <span key={status} className="flex items-center gap-1">
              <span style={{ color: STATUS_COLORS[status as NodeStatus] }}>{symbol}</span>
              <span className="text-gray-600 dark:text-gray-400 capitalize">{status}</span>
            </span>
          ))}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <span className="text-xs text-gray-500">Hover over nodes for details</span>
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>

      {/* Tooltip */}
      {hoveredNode && (
        <NodeTooltip node={hoveredNode.node} position={hoveredNode.position} />
      )}

      {/* CSS for animations */}
      <style>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -16;
          }
        }
        .animate-dash {
          animation: dash 0.5s linear infinite;
        }
      `}</style>
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
            dependencies: (data.dependencies as string[]) || existing.dependencies,
            status: 'pending',
          })
          setIsProving(true)
          break

        case 'fact_planning':
          next.set(factName, {
            ...existing,
            status: 'planning',
            tier: data.tier as number | undefined,
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
