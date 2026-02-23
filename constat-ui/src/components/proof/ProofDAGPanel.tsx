// Proof DAG Panel - Floating panel for auditable mode fact resolution visualization
// Uses d3-dag for proper directed acyclic graph layout

import { useState, useCallback, useMemo, useRef, useEffect } from 'react'
import { XMarkIcon, TableCellsIcon, ChevronDownIcon } from '@heroicons/react/24/outline'
import * as d3dag from 'd3-dag'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useUIStore } from '@/store/uiStore'
import { createSkillFromProof } from '@/api/skills'
import { MermaidBlock } from './MermaidBlock'
import { CodeViewer } from '@/components/artifacts/CodeViewer'

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
  attempt?: number
  code?: string
  validations?: string[]
  profile?: string[]
}

interface ProofDAGPanelProps {
  isOpen: boolean
  onClose: () => void
  facts: Map<string, FactNode>
  isPlanningComplete?: boolean
  summary?: string | null  // LLM-generated proof summary
  isSummaryGenerating?: boolean  // True while summary is being generated
  sessionId?: string
  onSkillCreated?: () => void
  onRedo?: (guidance?: string) => void
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
  resolved: '#DCFCE7',    // green-100 (for Inferences)
  failed: '#FEE2E2',      // red-100
  blocked: '#FFEDD5',     // orange-100
}

// Darker resolved color for Premises (P nodes)
const PREMISE_RESOLVED_BG = '#BBF7D0'  // green-200 (darker than green-100)

// Node dimensions
const NODE_WIDTH = 220
const NODE_HEIGHT = 50
const NODE_RADIUS = 8

// Helper to detect markdown table and count rows
function getTableRowCount(value: unknown): number | null {
  if (typeof value !== 'string') return null
  // Check if it looks like a markdown table (has | and header separator)
  if (!value.includes('|') || !value.includes('---')) return null
  // Count data rows (exclude header and separator)
  const lines = value.split('\n').filter(l => l.trim().startsWith('|'))
  return Math.max(0, lines.length - 2) // Subtract header and separator
}

// Helper to detect row count string like "14 rows", "14 records", or "(table_name) 14 rows"
function parseRowCountString(value: unknown): { count: number; tableName?: string } | null {
  if (typeof value !== 'string') return null

  // Match patterns like "(hr.employees) 15 rows" or "(table_name) 15 records"
  const prefixMatch = value.trim().match(/^\(([^)]+)\)\s*(\d+)\s*(rows?|records?)$/i)
  if (prefixMatch) {
    let tableName = prefixMatch[1]
    // Strip schema/database prefix (e.g., "hr.employees" -> "employees")
    if (tableName.includes('.')) {
      tableName = tableName.split('.').pop() || tableName
    }
    return { count: parseInt(prefixMatch[2], 10), tableName }
  }

  // Match simple patterns like "15 rows" or "15 records"
  const simpleMatch = value.trim().match(/^(\d+)\s*(rows?|records?)$/i)
  if (simpleMatch) {
    return { count: parseInt(simpleMatch[1], 10) }
  }

  return null
}

// Extract table name from fact node name (e.g., "I4: raise_recommendations" -> "raise_recommendations")
function extractTableName(nodeName: string): string | null {
  const match = nodeName.match(/^[PI]\d+:\s*(.+)$/)
  return match ? match[1].trim() : null
}

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
        {node.validations && node.validations.length > 0 && (
          <div>
            <span className="text-gray-500">Assertions:</span>
            <ul className="mt-1 space-y-0.5">
              {node.validations.map((v, i) => (
                <li key={i} className="flex items-center gap-1.5 text-green-600 dark:text-green-400">
                  <svg className="w-3 h-3 flex-shrink-0" viewBox="0 0 12 12" fill="none"><path d="M2 6l3 3 5-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
                  {v}
                </li>
              ))}
            </ul>
          </div>
        )}
        {node.profile && node.profile.length > 0 && (
          <div>
            <span className="text-gray-500">Data Profile:</span>
            <ul className="mt-1 space-y-0.5">
              {node.profile.map((p, i) => (
                <li key={i} className="flex items-center gap-1.5 text-gray-600 dark:text-gray-400">
                  <svg className="w-3 h-3 flex-shrink-0 text-blue-400" viewBox="0 0 12 12" fill="none"><circle cx="6" cy="6" r="4" stroke="currentColor" strokeWidth="1.5"/></svg>
                  {p}
                </li>
              ))}
            </ul>
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

export function ProofDAGPanel({ isOpen, onClose, facts, isPlanningComplete = false, summary, isSummaryGenerating = false, sessionId, onSkillCreated, onRedo }: ProofDAGPanelProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [hoveredNode, setHoveredNode] = useState<{ node: FactNode; position: { x: number; y: number } } | null>(null)
  const [selectedIdStack, setSelectedIdStack] = useState<string[]>([])
  // Derive selectedNode from live facts map so updates (code, elapsed_ms) are reflected
  const selectedNodeStack = selectedIdStack.map(id => facts.get(id)).filter((n): n is FactNode => !!n)
  const selectedNode = selectedNodeStack.length > 0 ? selectedNodeStack[selectedNodeStack.length - 1] : null
  const [showSkillForm, setShowSkillForm] = useState(false)
  const [skillName, setSkillName] = useState('')
  const [isSavingSkill, setIsSavingSkill] = useState(false)
  const [showRedoForm, setShowRedoForm] = useState(false)
  const [redoGuidance, setRedoGuidance] = useState('')
  const [codeExpanded, setCodeExpanded] = useState(false)
  const pushSelectedNode = (node: FactNode) => { setSelectedIdStack(prev => [...prev, node.id]); setCodeExpanded(false) }
  const popSelectedNode = () => { setSelectedIdStack(prev => prev.slice(0, -1)); setCodeExpanded(false) }
  const clearSelectedNodes = () => setSelectedIdStack([])
  const [dimensions, setDimensions] = useState({ width: 600, height: 400 })
  const [panelSize, setPanelSize] = useState(() => {
    try {
      const saved = localStorage.getItem('constat-proof-panel-geometry')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed.width && parsed.height) return { width: parsed.width, height: parsed.height }
      }
    } catch { /* ignore */ }
    return { width: typeof window !== 'undefined' ? window.innerWidth * 0.8 : 800, height: typeof window !== 'undefined' ? window.innerHeight * 0.8 : 600 }
  })
  const [panelPosition, setPanelPosition] = useState<{ x: number; y: number } | null>(() => {
    try {
      const saved = localStorage.getItem('constat-proof-panel-geometry')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed.x !== undefined && parsed.y !== undefined) return { x: parsed.x, y: parsed.y }
      }
    } catch { /* ignore */ }
    return null
  })
  const [, setIsResizing] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [showSummary, setShowSummary] = useState(false)

  // Persist panel geometry to localStorage when size or position changes
  useEffect(() => {
    const data: Record<string, number> = { width: panelSize.width, height: panelSize.height }
    if (panelPosition) {
      data.x = panelPosition.x
      data.y = panelPosition.y
    }
    localStorage.setItem('constat-proof-panel-geometry', JSON.stringify(data))
  }, [panelSize, panelPosition])

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
  const blockedCount = nodes.filter((n) => n.status === 'blocked').length
  const pendingCount = nodes.filter((n) => n.status !== 'resolved' && n.status !== 'failed' && n.status !== 'blocked').length

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

  const isProofComplete = pendingCount === 0 && (resolvedCount > 0 || failedCount > 0)

  // Compute critical path: longest dependency chain by elapsed_ms (or node count)
  const criticalPath = useMemo(() => {
    if (!isProofComplete || nodes.length === 0 || failedCount > 0) return new Set<string>()

    // Build adjacency: node -> its dependencies (parents)
    const nodeMap = new Map(nodes.map(n => [n.id, n]))

    // Topological order (BFS using in-degree)
    const inDegree = new Map<string, number>()
    const children = new Map<string, string[]>() // parent -> children that depend on it
    for (const n of nodes) {
      inDegree.set(n.id, n.dependencies.length)
      children.set(n.id, [])
    }
    for (const n of nodes) {
      for (const dep of n.dependencies) {
        children.get(dep)?.push(n.id)
      }
    }

    const topoOrder: string[] = []
    const queue = nodes.filter(n => n.dependencies.length === 0).map(n => n.id)
    while (queue.length > 0) {
      const cur = queue.shift()!
      topoOrder.push(cur)
      for (const child of (children.get(cur) || [])) {
        const deg = (inDegree.get(child) || 1) - 1
        inDegree.set(child, deg)
        if (deg === 0) queue.push(child)
      }
    }

    // DP: longest path to each node (sum of elapsed_ms, fallback to 1 per node)
    const longestTo = new Map<string, { cost: number; prev: string | null }>()
    for (const id of topoOrder) {
      const n = nodeMap.get(id)!
      const selfCost = n.elapsed_ms ?? 1
      if (n.dependencies.length === 0) {
        longestTo.set(id, { cost: selfCost, prev: null })
      } else {
        let bestCost = -1
        let bestPrev: string | null = null
        for (const dep of n.dependencies) {
          const depCost = longestTo.get(dep)?.cost ?? 0
          if (depCost > bestCost) {
            bestCost = depCost
            bestPrev = dep
          }
        }
        longestTo.set(id, { cost: bestCost + selfCost, prev: bestPrev })
      }
    }

    // Find the terminal node with longest path (use finalNode if available)
    let endNode = finalNode?.id || ''
    if (!endNode) {
      let maxCost = -1
      for (const [id, { cost }] of longestTo) {
        if (cost > maxCost) {
          maxCost = cost
          endNode = id
        }
      }
    }

    // Trace back
    const pathSet = new Set<string>()
    let cur: string | null = endNode
    while (cur) {
      pathSet.add(cur)
      cur = longestTo.get(cur)?.prev ?? null
    }

    return pathSet
  }, [nodes, isProofComplete, failedCount, finalNode])

  if (!isOpen) return null

  // Render edge path with curve
  const renderEdge = (
    sourceX: number,
    sourceY: number,
    targetX: number,
    targetY: number,
    sourceStatus: NodeStatus,
    targetStatus: NodeStatus,
    key: string,
    sourceId?: string,
    targetId?: string,
  ) => {
    // Create curved path from source bottom to target top
    const startY = sourceY + NODE_HEIGHT / 2
    const endY = targetY - NODE_HEIGHT / 2
    const midY = (startY + endY) / 2

    const path = `M ${sourceX} ${startY} C ${sourceX} ${midY}, ${targetX} ${midY}, ${targetX} ${endY}`

    // Critical path edge: both source and target on critical path
    const isCriticalEdge = sourceId && targetId && criticalPath.has(sourceId) && criticalPath.has(targetId)

    // Determine edge color based on resolution status
    const isResolved = sourceStatus === 'resolved'
    const strokeColor = isCriticalEdge ? '#D97706' : isResolved ? STATUS_COLORS.resolved : '#CBD5E1'
    const strokeW = isCriticalEdge ? 3 : 2

    return (
      <g key={key}>
        {/* Edge path */}
        <path
          d={path}
          fill="none"
          stroke={strokeColor}
          strokeWidth={strokeW}
          markerEnd={isCriticalEdge ? 'url(#arrowhead-critical)' : 'url(#arrowhead)'}
          className={isResolved || isCriticalEdge ? '' : 'opacity-50'}
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
    // Use darker shade for resolved Premises (P nodes)
    const isPremise = nodeId.startsWith('P')
    const bgColor = (status === 'resolved' && isPremise) ? PREMISE_RESOLVED_BG : STATUS_BG_COLORS[status]
    const borderColor = STATUS_COLORS[status]
    const isOnCriticalPath = criticalPath.has(nodeId)

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
        onClick={() => pushSelectedNode(nodeData)}
      >
        {/* Critical path glow */}
        {isOnCriticalPath && (
          <rect
            width={NODE_WIDTH + 4}
            height={NODE_HEIGHT + 4}
            x={-2}
            y={-2}
            rx={NODE_RADIUS + 1}
            ry={NODE_RADIUS + 1}
            fill="none"
            stroke="#D97706"
            strokeWidth={2}
            opacity={0.6}
          />
        )}
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
          className="select-none fill-gray-800 dark:fill-gray-100"
          fontSize={12}
          fontWeight={500}
        >
          {nodeData.name.length > 27 ? nodeData.name.slice(0, 24) + '...' : nodeData.name}
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
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Proof
          </h2>
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
              <div className="animate-pulse">
                <div className="flex justify-center mb-4">
                  <div className="relative">
                    <div className="w-12 h-12 border-4 border-gray-200 dark:border-gray-700 rounded-full" />
                    <div className="absolute top-0 left-0 w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  </div>
                </div>
                <p className="text-lg">{STATUS_SYMBOLS.planning} Generating proof plan...</p>
                <p className="text-sm mt-2">Analyzing the problem and identifying required facts.</p>
              </div>
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
                <marker
                  id="arrowhead-critical"
                  markerWidth="10"
                  markerHeight="7"
                  refX="9"
                  refY="3.5"
                  orient="auto"
                >
                  <polygon
                    points="0 0, 10 3.5, 0 7"
                    fill="#D97706"
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
                    `${sourceDagNode.id}-${targetDagNode.id}`,
                    sourceDagNode.id,
                    targetDagNode.id,
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

        {/* Legend + counts */}
        <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 flex flex-wrap items-center gap-4 text-xs">
          <span className="flex items-center gap-1 font-medium text-gray-700 dark:text-gray-300">
            <span className="text-blue-600">P</span> Premise
          </span>
          <span className="flex items-center gap-1 font-medium text-gray-700 dark:text-gray-300">
            <span className="text-purple-600">I</span> Inference
          </span>
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <span className="flex items-center gap-1">
            <span style={{ color: STATUS_COLORS.resolved }}>{STATUS_SYMBOLS.resolved}</span>
            <span className="text-gray-600 dark:text-gray-400">{resolvedCount} resolved</span>
          </span>
          <span className="flex items-center gap-1">
            <span style={{ color: STATUS_COLORS.failed }}>{STATUS_SYMBOLS.failed}</span>
            <span className="text-gray-600 dark:text-gray-400">{failedCount} failed</span>
          </span>
          {blockedCount > 0 && (
            <span className="flex items-center gap-1">
              <span style={{ color: STATUS_COLORS.blocked }}>{STATUS_SYMBOLS.blocked}</span>
              <span className="text-gray-600 dark:text-gray-400">{blockedCount} blocked</span>
            </span>
          )}
          <span className="flex items-center gap-1">
            <span style={{ color: STATUS_COLORS.pending }}>{STATUS_SYMBOLS.pending}</span>
            <span className="text-gray-600 dark:text-gray-400">{pendingCount} pending</span>
          </span>
          {criticalPath.size > 0 && (
            <>
              <span className="text-gray-300 dark:text-gray-600">|</span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 rounded" style={{ backgroundColor: '#D97706' }} />
                <span className="text-gray-600 dark:text-gray-400">Critical path</span>
              </span>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <span className="text-xs text-gray-500">Click nodes for details</span>
          <div className="flex gap-2 items-center">
            {isProofComplete && onRedo && (
              <button
                onClick={() => setShowRedoForm(true)}
                className="px-4 py-2 text-sm font-medium text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition-colors"
                title="Re-run proof with optional guidance"
              >
                Redo
              </button>
            )}
            {showRedoForm && onRedo && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40" onClick={() => { setShowRedoForm(false); setRedoGuidance('') }}>
                <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-[480px] max-w-[90vw]" onClick={(e) => e.stopPropagation()}>
                  <div className="px-5 py-4 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">Redo Proof</h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Provide guidance for the new proof attempt</p>
                  </div>
                  <form
                    className="px-5 py-4"
                    onSubmit={(e) => {
                      e.preventDefault()
                      onRedo(redoGuidance.trim() || undefined)
                      setShowRedoForm(false)
                      setRedoGuidance('')
                    }}
                  >
                    <textarea
                      value={redoGuidance}
                      onChange={(e) => setRedoGuidance(e.target.value)}
                      placeholder="What should be different this time? (optional)"
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-amber-500 resize-none"
                      rows={4}
                      autoFocus
                    />
                    <div className="flex justify-end gap-2 mt-4">
                      <button
                        type="button"
                        onClick={() => { setShowRedoForm(false); setRedoGuidance('') }}
                        className="px-4 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-lg transition-colors"
                      >
                        Prove
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
            {isProofComplete && sessionId && !showSkillForm && (
              <button
                onClick={() => setShowSkillForm(true)}
                className="px-4 py-2 text-sm font-medium text-emerald-600 dark:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/20 rounded-lg transition-colors"
              >
                Save as Skill
              </button>
            )}
            {showSkillForm && sessionId && (
              <form
                className="flex items-center gap-2"
                onSubmit={async (e) => {
                  e.preventDefault()
                  if (!skillName.trim() || isSavingSkill) return
                  setIsSavingSkill(true)
                  try {
                    await createSkillFromProof(sessionId, skillName.trim())
                    setShowSkillForm(false)
                    setSkillName('')
                    onSkillCreated?.()
                  } catch (err) {
                    console.error('Failed to save skill:', err)
                  } finally {
                    setIsSavingSkill(false)
                  }
                }}
              >
                <input
                  type="text"
                  value={skillName}
                  onChange={(e) => setSkillName(e.target.value)}
                  placeholder="Skill name..."
                  className="px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  autoFocus
                  disabled={isSavingSkill}
                />
                <button
                  type="submit"
                  disabled={!skillName.trim() || isSavingSkill}
                  className="px-3 py-1 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 disabled:bg-emerald-600/70 disabled:cursor-not-allowed rounded transition-colors flex items-center gap-1.5"
                >
                  {isSavingSkill ? (<><svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>Saving...</>) : 'Save'}
                </button>
                <button
                  type="button"
                  onClick={() => { setShowSkillForm(false); setSkillName('') }}
                  disabled={isSavingSkill}
                  className="px-2 py-1 text-sm text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                >
                  Cancel
                </button>
              </form>
            )}
            {resultNode && resultNode.status === 'resolved' && (
              <button
                onClick={() => pushSelectedNode(resultNode)}
                className="px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-colors"
              >
                Show Final Result
              </button>
            )}
            <button
              onClick={() => setShowSummary(true)}
              disabled={!summary}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center gap-2 ${
                summary
                  ? 'text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20'
                  : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
              }`}
              title={summary ? 'View LLM-generated summary' : isSummaryGenerating ? 'Generating summary...' : 'Summary not available'}
            >
              {isSummaryGenerating && (
                <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
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

      {/* Detail Panel - shown when node is clicked (stacked) */}
      {selectedNode && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center bg-black/20 pointer-events-auto" onClick={() => clearSelectedNodes()}>
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-2xl max-h-[80vh] overflow-auto pointer-events-auto"
            style={{
              // Offset each stacked modal slightly for visual effect
              transform: `translate(${(selectedNodeStack.length - 1) * 8}px, ${(selectedNodeStack.length - 1) * 8}px)`,
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-2">
                {selectedNodeStack.length > 1 && (
                  <button
                    onClick={popSelectedNode}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 mr-1"
                    title="Back to previous"
                  >
                    ←
                  </button>
                )}
                <span style={{ color: STATUS_COLORS[selectedNode.status] }} className="text-lg">
                  {STATUS_SYMBOLS[selectedNode.status]}
                </span>
                <h3 className="font-semibold text-gray-900 dark:text-gray-100">{selectedNode.name}</h3>
                {selectedNodeStack.length > 1 && (
                  <span className="text-xs text-gray-400 ml-2">({selectedNodeStack.length} deep)</span>
                )}
              </div>
              <button
                onClick={() => clearSelectedNodes()}
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
              {selectedNode.value !== undefined && (() => {
                const tableRowCount = getTableRowCount(selectedNode.value)
                const isTable = tableRowCount !== null
                const rowCountInfo = parseRowCountString(selectedNode.value)
                // Get table name from node name for materialized tables (e.g., "I4: results" -> "results")
                const materializedTableName = extractTableName(selectedNode.name)
                // Check if it's a database source reference like "(hr.employees) 15 rows"
                const isDbSourceRef = rowCountInfo !== null && rowCountInfo.tableName !== undefined
                // Check if it's a simple materialized table row count like "15 rows"
                const isClickableRowCount = rowCountInfo !== null && !rowCountInfo.tableName && materializedTableName !== null

                // For database source references, extract db and table name
                // Format: "(db.table) N rows" - tableName from rowCountInfo is "db.table"
                let dbName: string | undefined
                let dbTableName: string | undefined
                if (isDbSourceRef && rowCountInfo?.tableName) {
                  const parts = rowCountInfo.tableName.split('.')
                  if (parts.length >= 2) {
                    dbName = parts[0]
                    dbTableName = parts.slice(1).join('.')
                  } else {
                    // No dot - assume it's just a table name, try to get db from source
                    dbTableName = rowCountInfo.tableName
                    // Try to extract db name from source field (e.g., "database:hr")
                    const sourceMatch = selectedNode.source?.match(/^database:(\w+)$/i)
                    if (sourceMatch) {
                      dbName = sourceMatch[1]
                    }
                  }
                }

                return (
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-gray-500 uppercase">Value</span>
                      {isTable && (
                        <button
                          onClick={() => {
                            useUIStore.getState().openFullscreenArtifact({
                              type: 'proof_value',
                              name: selectedNode.name,
                              content: String(selectedNode.value),
                            })
                          }}
                          className="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                          title="View full table"
                        >
                          <TableCellsIcon className="w-3 h-3" />
                          {tableRowCount} rows
                        </button>
                      )}
                    </div>
                    <div className="mt-1 overflow-x-auto">
                      {isTable ? (
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
                          {String(selectedNode.value)}
                        </ReactMarkdown>
                      ) : isDbSourceRef && dbName && dbTableName ? (
                        <button
                          onClick={() => {
                            useUIStore.getState().openFullscreenArtifact({
                              type: 'database_table',
                              dbName,
                              tableName: dbTableName,
                            })
                          }}
                          className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                          title={`View ${dbName}.${dbTableName} table`}
                        >
                          <TableCellsIcon className="w-4 h-4" />
                          {rowCountInfo.count} rows
                        </button>
                      ) : isClickableRowCount ? (
                        <button
                          onClick={() => {
                            useUIStore.getState().openFullscreenArtifact({
                              type: 'table',
                              name: materializedTableName,
                            })
                          }}
                          className="flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline"
                          title={`View ${materializedTableName} table`}
                        >
                          <TableCellsIcon className="w-4 h-4" />
                          {rowCountInfo.count} rows
                        </button>
                      ) : (
                        <pre className="font-mono text-sm bg-gray-50 dark:bg-gray-900 p-3 rounded overflow-x-auto">
                          {typeof selectedNode.value === 'string' ? selectedNode.value : JSON.stringify(selectedNode.value, null, 2)}
                        </pre>
                      )}
                    </div>
                  </div>
                )
              })()}
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
                    {selectedNode.dependencies.map((dep) => {
                      const depNode = facts.get(dep)
                      if (depNode) {
                        return (
                          <button
                            key={dep}
                            onClick={() => pushSelectedNode(depNode)}
                            className="px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-sm hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors cursor-pointer"
                            title={`View ${dep}`}
                          >
                            {dep}
                          </button>
                        )
                      }
                      return (
                        <span key={dep} className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">{dep}</span>
                      )
                    })}
                  </div>
                </div>
              )}
              {selectedNode.reason && (
                <div>
                  <span className="text-xs font-medium text-gray-500 uppercase">Reason</span>
                  <p className="text-red-600 dark:text-red-400 mt-1">{selectedNode.reason}</p>
                </div>
              )}
              {(selectedNode.elapsed_ms !== undefined || selectedNode.attempt !== undefined) && (
                <div className="grid grid-cols-2 gap-4 text-sm">
                  {selectedNode.elapsed_ms !== undefined && (
                    <div>
                      <span className="text-xs font-medium text-gray-500 uppercase">Elapsed Time</span>
                      <p className="text-gray-700 dark:text-gray-300 mt-1">
                        {selectedNode.elapsed_ms >= 1000
                          ? `${(selectedNode.elapsed_ms / 1000).toFixed(1)}s`
                          : `${selectedNode.elapsed_ms}ms`}
                      </p>
                    </div>
                  )}
                  {selectedNode.attempt !== undefined && selectedNode.attempt > 1 && (
                    <div>
                      <span className="text-xs font-medium text-gray-500 uppercase">Retries</span>
                      <p className="text-amber-600 dark:text-amber-400 mt-1">{selectedNode.attempt - 1} {selectedNode.attempt === 2 ? 'retry' : 'retries'}</p>
                    </div>
                  )}
                </div>
              )}
              {selectedNode.code && (
                <div>
                  <button
                    onClick={() => setCodeExpanded(prev => !prev)}
                    className="flex items-center gap-1 text-xs font-medium text-gray-500 uppercase hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                  >
                    <ChevronDownIcon className={`w-3 h-3 transition-transform ${codeExpanded ? '' : '-rotate-90'}`} />
                    Code
                  </button>
                  {codeExpanded && (
                    <div className="mt-1">
                      <CodeViewer code={selectedNode.code} language="python" />
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Summary Panel - shown when summary button clicked */}
      {showSummary && summary && (
        <div className="fixed inset-0 z-[101] flex items-center justify-center bg-black/20 pointer-events-auto" onClick={() => setShowSummary(false)}>
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-2xl max-w-2xl max-h-[80vh] overflow-auto pointer-events-auto"
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
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    pre({ children }) {
                      return <pre className="has-[.mermaid-container]:!bg-transparent has-[.mermaid-container]:!border-none has-[.mermaid-container]:!p-0 has-[.mermaid-container]:!shadow-none">{children}</pre>
                    },
                    code({ className, children }) {
                      if (className === 'language-mermaid') {
                        return <MermaidBlock chart={String(children)} />
                      }
                      return <code className={className}>{children}</code>
                    },
                  }}
                >
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
            validations: data.validations as string[] | undefined,
            profile: data.profile as string[] | undefined,
          })
          break

        case 'fact_failed':
          next.set(factName, {
            ...existing,
            status: 'failed',
            reason: data.reason as string | undefined,
          })
          break

        case 'fact_blocked':
          next.set(factName, {
            ...existing,
            status: 'blocked',
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
