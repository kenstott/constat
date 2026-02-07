// Proof DAG Panel - Floating panel for auditable mode fact resolution visualization
// Uses d3-dag for proper directed acyclic graph layout

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import * as d3dag from 'd3-dag'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

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
  isPlanningComplete?: boolean
  summary?: string | null  // LLM-generated proof summary
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
const NODE_WIDTH = 220
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
      className="fixed z-[100] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-xl p-3 max-w-md"
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
          <div>
            <span className="text-gray-500">Value:</span>
            {typeof node.value === 'string' && node.value.includes('|') ? (
              <div className="mt-1 overflow-x-auto">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ children }) => (
                      <table className="text-xs border-collapse">{children}</table>
                    ),
                    th: ({ children }) => (
                      <th className="border border-gray-300 dark:border-gray-600 px-2 py-1 bg-gray-100 dark:bg-gray-700 text-left">{children}</th>
                    ),
                    td: ({ children }) => (
                      <td className="border border-gray-300 dark:border-gray-600 px-2 py-1">{children}</td>
                    ),
                  }}
                >
                  {node.value}
                </ReactMarkdown>
              </div>
            ) : (
              <span className="font-mono text-gray-900 dark:text-gray-100 ml-2">
                {JSON.stringify(node.value)}
              </span>
            )}
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

export function ProofDAGPanel({ isOpen, onClose, facts, isPlanningComplete = false, summary }: ProofDAGPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [hoveredNode, setHoveredNode] = useState<{ node: FactNode; position: { x: number; y: number } } | null>(null)
  const [selectedNode, setSelectedNode] = useState<FactNode | null>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 })
  const [panelSize, setPanelSize] = useState({ width: 800, height: 600 })
  const [panelPosition, setPanelPosition] = useState<{ x: number; y: number } | null>(null)
  const [, setIsResizing] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  // Initialize panel size on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      setPanelSize({ width: window.innerWidth * 0.8, height: window.innerHeight * 0.8 })
    }
  }, [])

  // Handle drag
  const handleDragStart = useCallback((e: React.MouseEvent) => {
    // Only start drag if clicking on the header background, not buttons
    if ((e.target as HTMLElement).closest('button')) return

    e.preventDefault()
    setIsDragging(true)

    const panel = panelRef.current
    if (!panel) return

    const rect = panel.getBoundingClientRect()
    const startX = e.clientX
    const startY = e.clientY
    const startLeft = panelPosition?.x ?? rect.left
    const startTop = panelPosition?.y ?? rect.top

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX
      const deltaY = moveEvent.clientY - startY
      setPanelPosition({
        x: startLeft + deltaX,
        y: startTop + deltaY,
      })
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [panelPosition])

  // Handle resize
  const handleMouseDown = useCallback((e: React.MouseEvent, direction: 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w') => {
    e.preventDefault()
    setIsResizing(true)
    const startX = e.clientX
    const startY = e.clientY
    const startWidth = panelSize.width
    const startHeight = panelSize.height

    const handleMouseMove = (moveEvent: MouseEvent) => {
      const deltaX = moveEvent.clientX - startX
      const deltaY = moveEvent.clientY - startY

      setPanelSize(() => {
        let newWidth = startWidth
        let newHeight = startHeight

        if (direction.includes('e')) newWidth = Math.max(400, startWidth + deltaX)
        if (direction.includes('w')) newWidth = Math.max(400, startWidth - deltaX)
        if (direction.includes('s')) newHeight = Math.max(300, startHeight + deltaY)
        if (direction.includes('n')) newHeight = Math.max(300, startHeight - deltaY)

        return { width: newWidth, height: newHeight }
      })
    }

    const handleMouseUp = () => {
      setIsResizing(false)
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
  }, [panelSize.width, panelSize.height])

  // Convert facts Map to DAG structure
  // Recalculate when isPlanningComplete changes to ensure all dependencies are resolved
  const dagData = useMemo(() => {
    const nodes = Array.from(facts.values())
    if (nodes.length === 0) return null

    // Build node list with parent references (dependencies become parents in top-down view)
    // In our case, if A depends on B, then B is a parent of A (B must resolve before A)
    const dagNodes: DagNode[] = nodes.map(node => ({
      id: node.id,
      // Filter dependencies to only include nodes that exist in our facts
      // Once planning is complete, all nodes should exist so all edges connect
      parentIds: node.dependencies.filter(dep => facts.has(dep)),
      data: node,
    }))

    return dagNodes
  }, [facts, isPlanningComplete])

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

  // Find final inference node (highest tier or node that nothing depends on)
  const finalNode = useMemo(() => {
    if (nodes.length === 0) return null
    // Collect all node IDs that are dependencies
    const dependencyIds = new Set<string>()
    nodes.forEach((n) => n.dependencies.forEach((d) => dependencyIds.add(d)))
    // Find nodes that aren't dependencies of anything (leaf/final nodes)
    const finalNodes = nodes.filter((n) => !dependencyIds.has(n.id))
    // Among final nodes, prefer inference (I) nodes over premises (P)
    const inferenceNodes = finalNodes.filter((n) => n.id.startsWith('I'))
    if (inferenceNodes.length > 0) {
      // Return the one with highest tier, or highest number if no tier
      return inferenceNodes.reduce((best, curr) => {
        const currNum = parseInt(curr.id.slice(1)) || 0
        const bestNum = parseInt(best.id.slice(1)) || 0
        return (curr.tier ?? currNum) > (best.tier ?? bestNum) ? curr : best
      })
    }
    return finalNodes[0] || null
  }, [nodes])

  // Find penultimate node (the one that final node depends on - contains the actual result)
  const resultNode = useMemo(() => {
    if (!finalNode || finalNode.dependencies.length === 0) return finalNode
    // Get the first dependency of the final node (should be the result)
    const depId = finalNode.dependencies[0]
    return nodes.find((n) => n.id === depId) || finalNode
  }, [finalNode, nodes])

  const isProofComplete = pendingCount === 0 && resolvedCount > 0

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

  // Render node - always get fresh data from facts Map
  const renderNode = (
    nodeId: string,
    x: number,
    y: number
  ) => {
    // Get fresh node data from facts Map (not cached in layout)
    const nodeData = facts.get(nodeId)
    if (!nodeData) return null

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
        onClick={() => setSelectedNode(nodeData)}
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
          {nodeData.name.length > 22 ? nodeData.name.slice(0, 20) + '...' : nodeData.name}
        </text>
      </g>
    )
  }

  return (
    <div className={`fixed inset-0 z-40 ${panelPosition ? '' : 'flex items-center justify-center'} pointer-events-none`}>
      <div
        ref={panelRef}
        className="bg-white dark:bg-gray-900 rounded-xl shadow-2xl flex flex-col pointer-events-auto border border-gray-200 dark:border-gray-700 relative"
        style={{
          width: panelSize.width,
          height: panelSize.height,
          maxWidth: '95vw',
          maxHeight: '95vh',
          ...(panelPosition ? { position: 'absolute', left: panelPosition.x, top: panelPosition.y } : {}),
        }}
      >
        {/* Resize handles */}
        <div className="absolute -top-1 -left-1 w-3 h-3 cursor-nw-resize" onMouseDown={(e) => handleMouseDown(e, 'nw')} />
        <div className="absolute -top-1 -right-1 w-3 h-3 cursor-ne-resize" onMouseDown={(e) => handleMouseDown(e, 'ne')} />
        <div className="absolute -bottom-1 -left-1 w-3 h-3 cursor-sw-resize" onMouseDown={(e) => handleMouseDown(e, 'sw')} />
        <div className="absolute -bottom-1 -right-1 w-3 h-3 cursor-se-resize" onMouseDown={(e) => handleMouseDown(e, 'se')} />
        <div className="absolute top-0 left-3 right-3 h-1 cursor-n-resize" onMouseDown={(e) => handleMouseDown(e, 'n')} />
        <div className="absolute bottom-0 left-3 right-3 h-1 cursor-s-resize" onMouseDown={(e) => handleMouseDown(e, 's')} />
        <div className="absolute left-0 top-3 bottom-3 w-1 cursor-w-resize" onMouseDown={(e) => handleMouseDown(e, 'w')} />
        <div className="absolute right-0 top-3 bottom-3 w-1 cursor-e-resize" onMouseDown={(e) => handleMouseDown(e, 'e')} />
        {/* Header - draggable */}
        <div
          className={`flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
          onMouseDown={handleDragStart}
        >
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Proof
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
          ) : !isPlanningComplete ? (
            <div className="text-center text-gray-500 py-8 min-w-[500px]">
              <div className="animate-pulse">
                <p className="text-lg">{STATUS_SYMBOLS.planning} Planning proof...</p>
                <p className="text-sm mt-2">Analyzing dependencies and building resolution graph.</p>
                <p className="text-xs mt-4 text-gray-400">{nodes.length} facts identified</p>
              </div>
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
                  const sourceDagNode = source.data as DagNode
                  const targetDagNode = target.data as DagNode
                  // Get fresh status from facts Map
                  const sourceData = facts.get(sourceDagNode.id)
                  const targetData = facts.get(targetDagNode.id)
                  if (!sourceData || !targetData) return null

                  return renderEdge(
                    source.x + layout.offsetX,
                    source.y + layout.offsetY,
                    target.x + layout.offsetX,
                    target.y + layout.offsetY,
                    sourceData.status,
                    targetData.status,
                    `${sourceDagNode.id}-${targetDagNode.id}`
                  )
                })}
              </g>

              {/* Render nodes */}
              <g className="nodes">
                {Array.from(layout.graph.nodes()).map((node) => {
                  const dagNode = node.data as DagNode
                  return renderNode(
                    dagNode.id,
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

        {/* Confidence Summary - shown when proof is complete */}
        {isProofComplete && finalNode && finalNode.status === 'resolved' && (
          <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-green-50 dark:bg-green-900/20">
            <div className="flex items-center justify-center gap-3">
              <span className="text-green-600 dark:text-green-400 text-lg">{STATUS_SYMBOLS.resolved}</span>
              <span className="text-sm font-medium text-green-800 dark:text-green-200">
                Proven with {finalNode.confidence !== undefined ? `${(finalNode.confidence * 100).toFixed(0)}%` : 'high'} confidence
              </span>
              {finalNode.value !== undefined && (
                <span className="text-sm text-green-700 dark:text-green-300">
                  = <span className="font-mono">{typeof finalNode.value === 'string' ? finalNode.value : JSON.stringify(finalNode.value)}</span>
                </span>
              )}
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex flex-wrap gap-4 text-xs">
          <span className="flex items-center gap-1 font-medium text-gray-700 dark:text-gray-300">
            <span className="text-blue-600">P</span> = Premise
          </span>
          <span className="flex items-center gap-1 font-medium text-gray-700 dark:text-gray-300">
            <span className="text-purple-600">I</span> = Inference
          </span>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          {Object.entries(STATUS_SYMBOLS).map(([status, symbol]) => (
            <span key={status} className="flex items-center gap-1">
              <span style={{ color: STATUS_COLORS[status as NodeStatus] }}>{symbol}</span>
              <span className="text-gray-600 dark:text-gray-400 capitalize">{status}</span>
            </span>
          ))}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <span className="text-xs text-gray-500">Click nodes for details</span>
          <div className="flex gap-2">
            {resultNode && resultNode.status === 'resolved' && (
              <button
                onClick={() => setSelectedNode(resultNode)}
                className="px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
              >
                Show Final Result
              </button>
            )}
            <button
              onClick={() => setShowSummary(true)}
              disabled={!summary}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                summary
                  ? 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20'
                  : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
              }`}
              title={summary ? 'View LLM-generated summary' : 'Summary generating...'}
            >
              Summary
            </button>
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>

      {/* Tooltip */}
      {hoveredNode && !selectedNode && (
        <NodeTooltip node={hoveredNode.node} position={hoveredNode.position} />
      )}

      {/* Detail Panel - shown when node is clicked */}
      {selectedNode && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center bg-black/20" onClick={() => setSelectedNode(null)}>
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-2xl max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span style={{ color: STATUS_COLORS[selectedNode.status] }} className="text-lg">
                  {STATUS_SYMBOLS[selectedNode.status]}
                </span>
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">{selectedNode.name}</h3>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4 space-y-4">
              {selectedNode.description && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Description</span>
                  <p className="text-gray-700 dark:text-gray-300 mt-1">{selectedNode.description}</p>
                </div>
              )}
              {selectedNode.value !== undefined && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Value</span>
                  <div className="mt-1 overflow-x-auto">
                    {typeof selectedNode.value === 'string' && selectedNode.value.includes('|') ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          table: ({ children }) => (
                            <table className="text-sm border-collapse w-full">{children}</table>
                          ),
                          th: ({ children }) => (
                            <th className="border border-gray-300 dark:border-gray-600 px-3 py-2 bg-gray-100 dark:bg-gray-700 text-left font-medium">{children}</th>
                          ),
                          td: ({ children }) => (
                            <td className="border border-gray-300 dark:border-gray-600 px-3 py-2">{children}</td>
                          ),
                        }}
                      >
                        {selectedNode.value}
                      </ReactMarkdown>
                    ) : (
                      <pre className="font-mono text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                        {typeof selectedNode.value === 'string' ? selectedNode.value : JSON.stringify(selectedNode.value, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-4 text-sm">
                {selectedNode.source && (
                  <div>
                    <span className="text-xs font-medium text-gray-500 uppercase">Source</span>
                    <p className="text-gray-700 dark:text-gray-300 mt-1">{selectedNode.source}</p>
                  </div>
                )}
                {selectedNode.confidence !== undefined && (
                  <div>
                    <span className="text-xs font-medium text-gray-500 uppercase">Confidence</span>
                    <p className="text-gray-700 dark:text-gray-300 mt-1">{(selectedNode.confidence * 100).toFixed(0)}%</p>
                  </div>
                )}
                {selectedNode.tier !== undefined && (
                  <div>
                    <span className="text-xs font-medium text-gray-500 uppercase">Tier</span>
                    <p className="text-gray-700 dark:text-gray-300 mt-1">{selectedNode.tier}</p>
                  </div>
                )}
                {selectedNode.strategy && (
                  <div>
                    <span className="text-xs font-medium text-gray-500 uppercase">Strategy</span>
                    <p className="text-gray-700 dark:text-gray-300 mt-1">{selectedNode.strategy}</p>
                  </div>
                )}
              </div>
              {selectedNode.formula && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Formula</span>
                  <pre className="font-mono text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded mt-1">{selectedNode.formula}</pre>
                </div>
              )}
              {selectedNode.dependencies.length > 0 && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Dependencies</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {selectedNode.dependencies.map((dep) => (
                      <span key={dep} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">{dep}</span>
                    ))}
                  </div>
                </div>
              )}
              {selectedNode.reason && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Reason</span>
                  <p className="text-red-600 dark:text-red-400 mt-1">{selectedNode.reason}</p>
                </div>
              )}
              {selectedNode.elapsed_ms !== undefined && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Elapsed Time</span>
                  <p className="text-gray-700 dark:text-gray-300 mt-1">{selectedNode.elapsed_ms}ms</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Summary Panel - shown when summary button clicked */}
      {showSummary && summary && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center bg-black/20" onClick={() => setShowSummary(false)}>
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-2xl max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">Proof Summary</h3>
              <button
                onClick={() => setShowSummary(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="p-4">
              <p className="text-xs text-gray-500 mb-3 italic">LLM-generated summary of the proof derivation</p>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {summary}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        </div>
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
