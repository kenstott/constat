// Glossary Panel — unified glossary view replacing EntityAccordion

import { useEffect, useMemo, useState, useCallback } from 'react'
import {
  ChevronRightIcon,
  ChevronDownIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  PlusIcon,
  SparklesIcon,
  ExclamationTriangleIcon,
  TrashIcon,
  ListBulletIcon,
} from '@heroicons/react/24/outline'
import { useGlossaryStore } from '@/store/glossaryStore'
import { getGlossaryTerm } from '@/api/sessions'
import type { GlossaryTerm, GlossaryEditorialStatus } from '@/types/api'

interface GlossaryPanelProps {
  sessionId: string
}

// Semantic type badge colors
const TYPE_COLORS: Record<string, string> = {
  concept: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  attribute: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
  action: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400',
  term: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
}

const STATUS_COLORS: Record<string, string> = {
  draft: 'text-gray-400 dark:text-gray-500',
  reviewed: 'text-yellow-500 dark:text-yellow-400',
  approved: 'text-green-500 dark:text-green-400',
}

const SCOPE_TABS = [
  { value: 'all', label: 'All' },
  { value: 'defined', label: 'Defined' },
  { value: 'self_describing', label: 'Self-describing' },
] as const

// Tree node type for hierarchy display
interface TreeNode {
  term: GlossaryTerm
  children: TreeNode[]
}

function buildTree(terms: GlossaryTerm[]): { roots: TreeNode[]; orphans: GlossaryTerm[] } {
  const byName = new Map<string, GlossaryTerm>()
  const byId = new Map<string, GlossaryTerm>()
  for (const t of terms) {
    byName.set(t.name.toLowerCase(), t)
    if (t.entity_id) byId.set(t.entity_id, t)
  }

  const nodeMap = new Map<string, TreeNode>()
  for (const t of terms) {
    nodeMap.set(t.name.toLowerCase(), { term: t, children: [] })
  }

  const roots: TreeNode[] = []
  const orphans: GlossaryTerm[] = []

  for (const t of terms) {
    const node = nodeMap.get(t.name.toLowerCase())!
    if (t.parent_id) {
      // parent_id could be an entity_id or a term ID — find parent
      const parentTerm = byId.get(t.parent_id)
      const parentNode = parentTerm ? nodeMap.get(parentTerm.name.toLowerCase()) : null
      if (parentNode) {
        parentNode.children.push(node)
      } else {
        roots.push(node)
      }
    } else {
      roots.push(node)
    }
  }

  return { roots, orphans }
}

// Connected resources display
function ConnectedResources({
  sessionId,
  termName,
}: {
  sessionId: string
  termName: string
}) {
  const [resources, setResources] = useState<Array<{
    entity_name: string
    entity_type: string
    sources: Array<{ document_name: string; source: string; section?: string }>
  }>>([])
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    getGlossaryTerm(sessionId, termName).then((data) => {
      if (!cancelled) {
        setResources(data.connected_resources || [])
        setLoaded(true)
      }
    }).catch(() => setLoaded(true))
    return () => { cancelled = true }
  }, [sessionId, termName])

  if (!loaded) return <div className="text-xs text-gray-400">Loading resources...</div>
  if (resources.length === 0) return null

  return (
    <div className="mt-1.5">
      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Connected Resources</div>
      {resources.map((r, i) => (
        <div key={i} className="text-xs text-gray-500 dark:text-gray-400 ml-2 mb-0.5">
          <span className="font-medium">{r.entity_name}</span>
          <span className="text-gray-400"> ({r.entity_type})</span>
          {r.sources.slice(0, 3).map((s, j) => (
            <div key={j} className="ml-2 text-gray-400">
              {s.document_name} {s.section ? `> ${s.section}` : ''}
            </div>
          ))}
        </div>
      ))}
    </div>
  )
}

// Inline definition editor
function DefineInline({
  name,
  sessionId,
  onDone,
}: {
  name: string
  sessionId: string
  onDone: () => void
}) {
  const [definition, setDefinition] = useState('')
  const { addDefinition } = useGlossaryStore()

  const handleSubmit = async () => {
    if (!definition.trim()) return
    await addDefinition(sessionId, name, definition.trim())
    onDone()
  }

  return (
    <div className="mt-1 space-y-1">
      <textarea
        value={definition}
        onChange={(e) => setDefinition(e.target.value)}
        placeholder="Business definition..."
        className="w-full text-xs p-1.5 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 resize-none"
        rows={2}
        autoFocus
      />
      <div className="flex gap-1">
        <button
          onClick={handleSubmit}
          className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Save
        </button>
        <button
          onClick={onDone}
          className="text-xs px-2 py-0.5 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

// Single glossary term item
function GlossaryItem({
  term,
  sessionId,
  depth = 0,
}: {
  term: GlossaryTerm
  sessionId: string
  depth?: number
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [isDefining, setIsDefining] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editDef, setEditDef] = useState('')
  const { updateTerm, refineTerm, deleteTerm } = useGlossaryStore()

  const typeColor =
    TYPE_COLORS[term.semantic_type || ''] ||
    'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'

  const statusColor = STATUS_COLORS[term.status || ''] || ''

  const isDefined = term.glossary_status === 'defined'

  const handleEditSave = async () => {
    if (!editDef.trim()) return
    await updateTerm(sessionId, term.name, { definition: editDef.trim() })
    setIsEditing(false)
  }

  const handleRefine = async () => {
    await refineTerm(sessionId, term.name)
  }

  const handleStatusChange = async (status: GlossaryEditorialStatus) => {
    await updateTerm(sessionId, term.name, { status })
  }

  const handleDelete = async () => {
    await deleteTerm(sessionId, term.name)
  }

  return (
    <div className="border-b border-gray-100 dark:border-gray-700 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center gap-2 py-2 px-1 text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
      >
        <ChevronRightIcon
          className={`w-3 h-3 text-gray-400 transition-transform flex-shrink-0 ${
            isOpen ? 'rotate-90' : ''
          }`}
        />
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300 flex-1 truncate">
          {term.display_name}
        </span>
        {term.semantic_type && (
          <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 ${typeColor}`}>
            {term.semantic_type}
          </span>
        )}
        {isDefined && term.status && (
          <span className={`text-xs flex-shrink-0 ${statusColor}`}>{term.status}</span>
        )}
        {term.domain && (
          <span className="text-xs px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 flex-shrink-0">
            {term.domain}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="pb-2 space-y-1.5" style={{ paddingLeft: `${depth * 16 + 24}px` }}>
          {/* Definition */}
          {isDefined && term.definition && !isEditing && (
            <p
              className="text-xs text-gray-600 dark:text-gray-400 italic cursor-pointer hover:text-gray-800 dark:hover:text-gray-200"
              onClick={() => {
                setEditDef(term.definition || '')
                setIsEditing(true)
              }}
            >
              &ldquo;{term.definition}&rdquo;
            </p>
          )}

          {/* Inline editor for existing definition */}
          {isEditing && (
            <div className="space-y-1">
              <textarea
                value={editDef}
                onChange={(e) => setEditDef(e.target.value)}
                className="w-full text-xs p-1.5 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 resize-none"
                rows={2}
              />
              <div className="flex gap-1">
                <button
                  onClick={handleEditSave}
                  className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded hover:bg-blue-600"
                >
                  Save
                </button>
                <button
                  onClick={() => setIsEditing(false)}
                  className="text-xs px-2 py-0.5 text-gray-500 hover:text-gray-700"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}

          {/* Define button for self-describing terms */}
          {!isDefined && !isDefining && (
            <button
              onClick={() => setIsDefining(true)}
              className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600"
            >
              <PlusIcon className="w-3 h-3" />
              Define
            </button>
          )}

          {isDefining && (
            <DefineInline
              name={term.name}
              sessionId={sessionId}
              onDone={() => setIsDefining(false)}
            />
          )}

          {/* Aliases */}
          {term.aliases && term.aliases.length > 0 && (
            <div className="text-xs text-gray-500 dark:text-gray-400">
              <span className="font-medium">Aliases:</span> {term.aliases.join(', ')}
            </div>
          )}

          {/* Provenance */}
          {isDefined && term.provenance && (
            <div className="text-xs text-gray-400 dark:text-gray-500">
              Provenance: {term.provenance}
            </div>
          )}

          {/* Connected resources */}
          <ConnectedResources sessionId={sessionId} termName={term.name} />

          {/* Actions for defined terms */}
          {isDefined && (
            <div className="flex gap-2 pt-1">
              <button
                onClick={handleRefine}
                className="flex items-center gap-1 text-xs text-purple-500 hover:text-purple-600"
                title="AI-assisted refinement"
              >
                <SparklesIcon className="w-3 h-3" />
                Refine
              </button>
              {term.status === 'draft' && (
                <button
                  onClick={() => handleStatusChange('reviewed')}
                  className="text-xs text-yellow-500 hover:text-yellow-600"
                >
                  Mark Reviewed
                </button>
              )}
              {term.status === 'reviewed' && (
                <button
                  onClick={() => handleStatusChange('approved')}
                  className="text-xs text-green-500 hover:text-green-600"
                >
                  Approve
                </button>
              )}
              <button
                onClick={handleDelete}
                className="text-xs text-red-400 hover:text-red-500"
              >
                Remove Definition
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Recursive tree node renderer
function TreeNodeView({
  node,
  sessionId,
  depth = 0,
}: {
  node: TreeNode
  sessionId: string
  depth?: number
}) {
  const [expanded, setExpanded] = useState(depth < 2)

  return (
    <div>
      <div className="flex items-center">
        {node.children.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="p-0.5"
            style={{ marginLeft: `${depth * 12}px` }}
          >
            {expanded ? (
              <ChevronDownIcon className="w-3 h-3 text-gray-400" />
            ) : (
              <ChevronRightIcon className="w-3 h-3 text-gray-400" />
            )}
          </button>
        )}
        <div className="flex-1" style={{ marginLeft: node.children.length === 0 ? `${depth * 12 + 16}px` : '0' }}>
          <GlossaryItem term={node.term} sessionId={sessionId} depth={0} />
        </div>
      </div>
      {expanded && node.children.map((child) => (
        <TreeNodeView
          key={child.term.name}
          node={child}
          sessionId={sessionId}
          depth={depth + 1}
        />
      ))}
    </div>
  )
}

// Taxonomy suggestions panel
function TaxonomySuggestionsPanel({ sessionId }: { sessionId: string }) {
  const { taxonomySuggestions, acceptTaxonomySuggestion, dismissTaxonomySuggestion } =
    useGlossaryStore()

  if (taxonomySuggestions.length === 0) return null

  return (
    <div className="border border-purple-200 dark:border-purple-800 rounded p-2 bg-purple-50 dark:bg-purple-900/20">
      <div className="text-xs font-medium text-purple-700 dark:text-purple-400 mb-1.5">
        Taxonomy Suggestions ({taxonomySuggestions.length})
      </div>
      {taxonomySuggestions.map((s, i) => (
        <div key={i} className="flex items-center gap-2 text-xs py-1 border-b border-purple-100 dark:border-purple-800 last:border-0">
          <div className="flex-1">
            <span className="text-gray-600 dark:text-gray-400">{s.child}</span>
            <span className="text-gray-400 mx-1">&rarr;</span>
            <span className="font-medium text-gray-700 dark:text-gray-300">{s.parent}</span>
            <span className="text-gray-400 ml-1">({s.confidence})</span>
            {s.reason && <div className="text-gray-400 text-xs mt-0.5">{s.reason}</div>}
          </div>
          <button
            onClick={() => acceptTaxonomySuggestion(sessionId, s)}
            className="text-xs px-1.5 py-0.5 bg-purple-500 text-white rounded hover:bg-purple-600"
          >
            Accept
          </button>
          <button
            onClick={() => dismissTaxonomySuggestion(s)}
            className="text-xs px-1.5 py-0.5 text-gray-500 hover:text-gray-700"
          >
            Reject
          </button>
        </div>
      ))}
    </div>
  )
}

// Deprecated terms section
function DeprecatedSection({
  sessionId,
}: {
  sessionId: string
}) {
  const { deprecatedTerms, deleteTerm } = useGlossaryStore()

  if (deprecatedTerms.length === 0) return null

  return (
    <div className="border border-amber-200 dark:border-amber-800 rounded p-2 bg-amber-50 dark:bg-amber-900/20">
      <div className="flex items-center gap-1 text-xs font-medium text-amber-700 dark:text-amber-400 mb-1.5">
        <ExclamationTriangleIcon className="w-3.5 h-3.5" />
        Deprecated ({deprecatedTerms.length})
      </div>
      {deprecatedTerms.map((t) => (
        <div key={t.name} className="flex items-center gap-2 text-xs py-1 border-b border-amber-100 dark:border-amber-800 last:border-0">
          <span className="text-gray-600 dark:text-gray-400 flex-1">{t.display_name}</span>
          <span className="text-gray-400 truncate max-w-[120px]">{t.definition}</span>
          <button
            onClick={() => deleteTerm(sessionId, t.name)}
            className="text-red-400 hover:text-red-500 flex-shrink-0"
            title="Delete deprecated term"
          >
            <TrashIcon className="w-3.5 h-3.5" />
          </button>
        </div>
      ))}
    </div>
  )
}

// Tree icon SVG inline
function TreeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
      <path d="M12 3v6M12 9H6M6 9v6M6 15H3M6 15h3M12 9h6M18 9v6M18 15h3M18 15h-3" />
    </svg>
  )
}

export default function GlossaryPanel({ sessionId }: GlossaryPanelProps) {
  const {
    terms,
    filters,
    viewMode,
    totalDefined,
    totalSelfDescribing,
    loading,
    fetchTerms,
    fetchDeprecated,
    suggestTaxonomy,
    setFilter,
    setViewMode,
  } = useGlossaryStore()

  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchTerms(sessionId)
    fetchDeprecated(sessionId)
  }, [sessionId, filters.scope, fetchTerms, fetchDeprecated])

  // Filter terms by search
  const filteredTerms = useMemo(() => {
    if (!search) return terms
    const q = search.toLowerCase()
    return terms.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.display_name.toLowerCase().includes(q) ||
        (t.definition && t.definition.toLowerCase().includes(q)) ||
        t.aliases.some((a) => a.toLowerCase().includes(q))
    )
  }, [terms, search])

  // Filter by status
  const displayTerms = useMemo(() => {
    if (!filters.status) return filteredTerms
    return filteredTerms.filter((t) => t.status === filters.status)
  }, [filteredTerms, filters.status])

  // Build tree for tree view
  const tree = useMemo(() => {
    if (viewMode !== 'tree') return null
    return buildTree(displayTerms)
  }, [displayTerms, viewMode])

  const handleScopeChange = useCallback(
    (scope: 'all' | 'defined' | 'self_describing') => {
      setFilter({ scope })
    },
    [setFilter]
  )

  return (
    <div className="space-y-2">
      {/* Header with view toggle and suggest taxonomy */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => setViewMode('list')}
          className={`p-1 rounded ${viewMode === 'list' ? 'bg-gray-200 dark:bg-gray-600' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
          title="List view"
        >
          <ListBulletIcon className="w-3.5 h-3.5 text-gray-500" />
        </button>
        <button
          onClick={() => setViewMode('tree')}
          className={`p-1 rounded ${viewMode === 'tree' ? 'bg-gray-200 dark:bg-gray-600' : 'hover:bg-gray-100 dark:hover:bg-gray-700'}`}
          title="Tree view"
        >
          <TreeIcon className="w-3.5 h-3.5 text-gray-500" />
        </button>
        <div className="flex-1" />
        <button
          onClick={() => suggestTaxonomy(sessionId)}
          className="flex items-center gap-1 text-xs text-purple-500 hover:text-purple-600 px-1.5 py-0.5 rounded hover:bg-purple-50 dark:hover:bg-purple-900/20"
          title="AI-suggested taxonomy"
        >
          <SparklesIcon className="w-3.5 h-3.5" />
          Taxonomy
        </button>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-400" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search glossary..."
          className="w-full pl-7 pr-7 py-1.5 text-xs border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 placeholder-gray-400"
        />
        {search && (
          <button
            onClick={() => setSearch('')}
            className="absolute right-2 top-1/2 -translate-y-1/2"
          >
            <XMarkIcon className="w-3.5 h-3.5 text-gray-400 hover:text-gray-600" />
          </button>
        )}
      </div>

      {/* Scope tabs */}
      <div className="flex gap-1">
        {SCOPE_TABS.map((tab) => (
          <button
            key={tab.value}
            onClick={() => handleScopeChange(tab.value)}
            className={`text-xs px-2 py-1 rounded ${
              filters.scope === tab.value
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                : 'text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
        {search && (
          <span className="text-xs text-gray-400 self-center ml-auto">
            {displayTerms.length} results
          </span>
        )}
      </div>

      {/* Stats */}
      <div className="text-xs text-gray-400 dark:text-gray-500">
        {totalDefined} defined, {totalSelfDescribing} self-describing
      </div>

      {/* Taxonomy suggestions */}
      <TaxonomySuggestionsPanel sessionId={sessionId} />

      {/* Deprecated terms */}
      <DeprecatedSection sessionId={sessionId} />

      {/* Terms list or tree */}
      {loading ? (
        <div className="text-xs text-gray-400 py-4 text-center">Loading glossary...</div>
      ) : displayTerms.length === 0 ? (
        <div className="text-xs text-gray-400 py-4 text-center">
          {search ? 'No matching terms' : 'No glossary terms'}
        </div>
      ) : viewMode === 'tree' && tree ? (
        <div className="max-h-96 overflow-y-auto">
          {tree.roots.map((node) => (
            <TreeNodeView
              key={node.term.name}
              node={node}
              sessionId={sessionId}
            />
          ))}
        </div>
      ) : (
        <div className="max-h-96 overflow-y-auto">
          {displayTerms.map((term) => (
            <GlossaryItem key={`${term.name}-${term.domain || ''}`} term={term} sessionId={sessionId} />
          ))}
        </div>
      )}
    </div>
  )
}
