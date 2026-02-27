// Glossary Panel — unified glossary view replacing EntityAccordion

import { useEffect, useMemo, useState, useCallback, useRef } from 'react'
import {
  ChevronRightIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  PlusIcon,
  MinusSmallIcon,
  PlusSmallIcon,
  SparklesIcon,
  ExclamationTriangleIcon,
  TrashIcon,
  ListBulletIcon,
  ArrowsPointingOutIcon,
  ArrowsPointingInIcon,
  ArrowUpIcon,
  PencilIcon,
  EyeIcon,
  EyeSlashIcon,
  CheckIcon,
} from '@heroicons/react/24/outline'
import { useGlossaryStore } from '@/store/glossaryStore'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import {
  getGlossaryTerm,
  createRelationship,
  updateRelationshipVerb,
  approveRelationship,
  deleteRelationship,
  updateGlossaryTerm,
  draftGlossaryDefinition,
  draftGlossaryAliases,
  listDomains,
  getDomainTree,
  moveDomainSource,
} from '@/api/sessions'
import type { DomainTreeNode } from '@/api/sessions'
import { forceSimulation, forceLink, forceManyBody, forceCollide, forceX, forceY } from 'd3-force'
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
  // Map all possible IDs to terms for parent lookup
  // parent_id can be a glossary_id (glossary term hash) or entity_id
  const byId = new Map<string, GlossaryTerm>()
  for (const t of terms) {
    if (t.glossary_id) byId.set(t.glossary_id, t)
    if (t.entity_id) byId.set(t.entity_id, t)
  }

  const nodeMap = new Map<string, TreeNode>()
  for (const t of terms) {
    nodeMap.set(t.name.toLowerCase(), { term: t, children: [] })
  }

  // Build term hierarchy within each domain
  const domainRoots = new Map<string, TreeNode[]>()
  const orphans: GlossaryTerm[] = []

  for (const t of terms) {
    const node = nodeMap.get(t.name.toLowerCase())!
    const bucket = t.domain || (t.glossary_status === 'self_describing' ? '(system)' : '(user)')
    if (t.parent_id) {
      const parentTerm = byId.get(t.parent_id)
      const parentNode = parentTerm ? nodeMap.get(parentTerm.name.toLowerCase()) : null
      if (parentNode) {
        parentNode.children.push(node)
      } else {
        if (!domainRoots.has(bucket)) domainRoots.set(bucket, [])
        domainRoots.get(bucket)!.push(node)
      }
    } else {
      if (!domainRoots.has(bucket)) domainRoots.set(bucket, [])
      domainRoots.get(bucket)!.push(node)
    }
  }

  // If only one domain (or no domains), flatten — no domain grouping needed
  const domains = [...domainRoots.keys()]
  if (domains.length <= 1) {
    return { roots: domainRoots.get(domains[0] || '(user)') || [], orphans }
  }

  const bucketLabel = (d: string) => d === '(system)' ? 'System' : d === '(user)' ? 'User' : d

  // Multiple domains: create domain folder nodes
  const roots: TreeNode[] = domains.sort().map(domain => ({
    term: {
      name: `__domain__${domain}`,
      display_name: bucketLabel(domain),
      glossary_status: 'defined' as const,
      aliases: [],
      connected_resources: [],
      cardinality: 'many',
      mention_count: 0,
    } as GlossaryTerm,
    children: domainRoots.get(domain) || [],
  }))

  return { roots, orphans }
}

// Deep link to a glossary term via URL path
function navigateToTerm(name: string) {
  console.log('[deep-link] navigateToTerm:', name)
  useUIStore.getState().navigateTo({ type: 'glossary_term', termName: name })
}

// Clickable term link
function TermLink({ name, displayName }: { name: string; displayName: string }) {
  return (
    <button
      onClick={() => navigateToTerm(name)}
      className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
    >
      {displayName}
    </button>
  )
}

// Clickable source link — deep links to tables, documents, and APIs
function SourceLink({ source, documentName, section }: { source: string; documentName: string; section?: string }) {
  const label = `${documentName}${section ? ` > ${section}` : ''}`
  const { navigateTo } = useUIStore.getState()

  // Strip source prefix (e.g., "schema:chinook.Track" → "chinook.Track", "api:countries.Breed" → "countries.Breed")
  const stripped = documentName.includes(':') ? documentName.split(':').slice(1).join(':') : documentName

  const handleClick = () => {
    console.log('[deep-link] SourceLink clicked:', { source, documentName, stripped })
    if (source === 'schema') {
      const parts = stripped.split('.')
      if (parts.length >= 2) {
        navigateTo({ type: 'table', dbName: parts[0], tableName: parts[1] })
      }
    } else if (source === 'document') {
      navigateTo({ type: 'document', documentName: stripped })
    } else if (source === 'api') {
      const parts = stripped.split('.')
      navigateTo({ type: 'api', apiName: parts[0] })
    }
  }

  return (
    <button
      onClick={handleClick}
      className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
    >
      {label}
    </button>
  )
}

// Inline-editable relationship row
function RelationshipRow({
  rel,
  sessionId,
  onDeleted,
  onUpdated,
  onApproved,
}: {
  rel: { id: string; subject: string; verb: string; object: string; user_edited?: boolean }
  sessionId: string
  onDeleted: (id: string) => void
  onUpdated: (id: string, verb: string) => void
  onApproved: (id: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [editVerb, setEditVerb] = useState(rel.verb)
  const [hover, setHover] = useState(false)

  const handleSave = async () => {
    if (!editVerb.trim() || editVerb.trim() === rel.verb) {
      setEditing(false)
      return
    }
    await updateRelationshipVerb(sessionId, rel.id, editVerb.trim())
    onUpdated(rel.id, editVerb.trim())
    setEditing(false)
  }

  const handleDelete = async () => {
    await deleteRelationship(sessionId, rel.id)
    onDeleted(rel.id)
  }

  const handleApprove = async () => {
    await approveRelationship(sessionId, rel.id)
    onApproved(rel.id)
  }

  return (
    <div
      className="group flex items-center text-xs text-gray-500 dark:text-gray-400 ml-2 py-0.5"
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <TermLink name={rel.subject} displayName={rel.subject} />
      {editing ? (
        <input
          value={editVerb}
          onChange={(e) => setEditVerb(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSave()
            if (e.key === 'Escape') { setEditing(false); setEditVerb(rel.verb) }
          }}
          onBlur={handleSave}
          className="mx-1 px-1 py-0 text-xs border border-blue-300 dark:border-blue-600 rounded bg-white dark:bg-gray-800 text-blue-500 w-24"
          autoFocus
        />
      ) : (
        <button
          onClick={() => { setEditVerb(rel.verb); setEditing(true) }}
          className="text-blue-500 dark:text-blue-400 mx-1 hover:underline cursor-pointer"
        >
          {rel.verb}
        </button>
      )}
      <TermLink name={rel.object} displayName={rel.object} />
      {rel.user_edited ? (
        <CheckIcon className="ml-1 w-3 h-3 text-green-500 flex-shrink-0" title="Approved — preserved during regeneration" />
      ) : hover && !editing ? (
        <button
          onClick={handleApprove}
          className="ml-1 text-green-400 hover:text-green-500 flex-shrink-0"
          title="Approve — preserve during regeneration"
        >
          <CheckIcon className="w-3 h-3" />
        </button>
      ) : null}
      {hover && !editing && (
        <button
          onClick={handleDelete}
          className="ml-1 text-red-400 hover:text-red-500 flex-shrink-0"
          title="Delete relationship"
        >
          <XMarkIcon className="w-3 h-3" />
        </button>
      )}
    </div>
  )
}

// Autocomplete dropdown for entity names
function EntityAutocomplete({
  value,
  onChange,
  placeholder,
  className,
  autoFocus,
}: {
  value: string
  onChange: (val: string) => void
  placeholder: string
  className?: string
  autoFocus?: boolean
}) {
  const { terms } = useGlossaryStore()
  const [focused, setFocused] = useState(false)

  const matches = useMemo(() => {
    const q = value.trim().toLowerCase()
    const filtered = q
      ? terms.filter(t => t.name.includes(q) || t.display_name.toLowerCase().includes(q))
      : terms
    return filtered.slice(0, 10)
  }, [value, terms])

  return (
    <div className="relative">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setTimeout(() => setFocused(false), 150)}
        placeholder={placeholder}
        className={className}
        autoFocus={autoFocus}
      />
      {focused && matches.length > 0 && (
        <div className="absolute z-50 top-full left-0 mt-0.5 min-w-[10rem] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg max-h-48 overflow-y-auto">
          {matches.map((t) => (
            <button
              key={t.name}
              onMouseDown={(e) => { e.preventDefault(); onChange(t.name); setFocused(false) }}
              className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 truncate"
            >
              {t.display_name}
              {t.semantic_type && (
                <span className="text-gray-400 ml-1">({t.semantic_type})</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// Verb picklist — shows existing verbs plus allows new
const COMMON_VERBS = [
  'contains', 'has', 'belongs_to',
  'manages', 'reports_to', 'is_type_of',
  'creates', 'processes', 'approves', 'places',
  'sends', 'receives', 'transfers',
  'drives', 'requires', 'enables',
  'precedes', 'follows', 'triggers',
  'references', 'works_in', 'participates_in', 'uses',
]

function VerbAutocomplete({
  value,
  onChange,
  existingVerbs,
  className,
}: {
  value: string
  onChange: (val: string) => void
  existingVerbs: string[]
  className?: string
}) {
  const [focused, setFocused] = useState(false)

  const allVerbs = useMemo(() => {
    const set = new Set([...existingVerbs, ...COMMON_VERBS])
    return Array.from(set).sort()
  }, [existingVerbs])

  const matches = useMemo(() => {
    if (!value.trim()) return allVerbs.slice(0, 8)
    const q = value.toLowerCase()
    return allVerbs.filter(v => v.includes(q)).slice(0, 8)
  }, [value, allVerbs])

  return (
    <div className="relative">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setFocused(true)}
        onBlur={() => setTimeout(() => setFocused(false), 150)}
        placeholder="verb"
        className={className}
      />
      {focused && matches.length > 0 && (
        <div className="absolute z-50 top-full left-0 mt-0.5 w-28 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg max-h-32 overflow-y-auto">
          {matches.map((v) => (
            <button
              key={v}
              onMouseDown={(e) => { e.preventDefault(); onChange(v); setFocused(false) }}
              className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 text-blue-500 truncate"
            >
              {v}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

// Add new relationship row — subject is locked to the current term
function AddRelationshipRow({
  sessionId,
  termName,
  onCreated,
}: {
  sessionId: string
  termName: string
  onCreated: (rel: { id: string; subject: string; verb: string; object: string; confidence: number }) => void
}) {
  const [open, setOpen] = useState(false)
  const [verb, setVerb] = useState('')
  const [object, setObject] = useState('')

  const handleSubmit = async () => {
    if (!verb.trim() || !object.trim()) return
    const result = await createRelationship(sessionId, termName, verb.trim(), object.trim())
    onCreated({
      id: result.id,
      subject: termName,
      verb: verb.trim(),
      object: object.trim(),
      confidence: 1.0,
    })
    setVerb('')
    setObject('')
    setOpen(false)
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-1 text-xs text-blue-500 hover:text-blue-600 ml-2 mt-1"
      >
        <PlusIcon className="w-3 h-3" />
        Add relationship
      </button>
    )
  }

  return (
    <div className="ml-2 mt-1 space-y-1">
      <div className="flex gap-1 items-center">
        <span className="text-xs text-gray-600 dark:text-gray-400 truncate max-w-[6rem]">{termName}</span>
        <VerbAutocomplete
          value={verb}
          onChange={setVerb}
          existingVerbs={[]}
          className="text-xs px-1.5 py-0.5 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-blue-500 w-20"
        />
        <EntityAutocomplete
          value={object}
          onChange={setObject}
          placeholder="Object"
          className="text-xs px-1.5 py-0.5 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 w-24"
          autoFocus
        />
        <button
          onClick={handleSubmit}
          className="text-xs px-1.5 py-0.5 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Add
        </button>
        <button
          onClick={() => setOpen(false)}
          className="text-xs text-gray-400 hover:text-gray-600"
        >
          <XMarkIcon className="w-3 h-3" />
        </button>
      </div>
    </div>
  )
}

// Connected resources display
function ConnectedResources({
  sessionId,
  termName,
}: {
  sessionId: string
  termName: string
}) {
  const refreshKey = useGlossaryStore((s) => s.refreshKey)
  const [detail, setDetail] = useState<{
    resources: Array<{
      entity_name: string
      entity_type: string
      sources: Array<{ document_name: string; source: string; section?: string }>
    }>
    parent: { name: string; display_name: string } | null
    parent_verb: string
    children: Array<{ name: string; display_name: string; parent_verb?: string }>
    relationships: Array<{ id: string; subject: string; verb: string; object: string; confidence: number }>
    cluster_siblings: string[]
    domain: string | null
    domain_path: string | null
  }>({ resources: [], parent: null, parent_verb: 'HAS_KIND', children: [], relationships: [], cluster_siblings: [], domain: null, domain_path: null })
  const [loaded, setLoaded] = useState(false)
  const [graphOpen, setGraphOpen] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    getGlossaryTerm(sessionId, termName).then((data) => {
      if (!cancelled) {
        console.log(`[ConnectedResources] ${termName}: relationships=${(data.relationships || []).length}, parent=${!!data.parent}, children=${(data.children || []).length}, resources=${(data.connected_resources || []).length}`)
        setDetail({
          resources: data.connected_resources || [],
          parent: data.parent || null,
          parent_verb: data.parent_verb || 'HAS_KIND',
          children: data.children || [],
          relationships: data.relationships || [],
          cluster_siblings: data.cluster_siblings || [],
          domain: data.domain || null,
          domain_path: data.domain_path || null,
        })
        setLoaded(true)
      }
    }).catch((err) => {
      console.error(`[ConnectedResources] ${termName}: fetch failed`, err)
      setLoaded(true)
    })
    return () => { cancelled = true }
  }, [sessionId, termName, refreshKey])

  if (!loaded) return <div className="text-xs text-gray-400">Loading resources...</div>

  const { resources, parent, parent_verb, children, relationships, cluster_siblings, domain, domain_path } = detail
  const hasConnections = !!(parent || children.length > 0 || relationships.length > 0 || cluster_siblings.length > 0)
  const hasContent = resources.length > 0 || hasConnections || !!domain
  if (!hasContent) return null

  return (
    <div className="mt-1.5 space-y-1.5">
      {domain && (
        <div className="text-xs text-gray-400 dark:text-gray-500">
          Domain: {domain_path || domain}
        </div>
      )}
      {parent && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">
            Parent <span className="text-gray-400 font-normal">({parent_verb})</span>
          </div>
          <div className="text-xs ml-2">
            <TermLink name={parent.name} displayName={parent.display_name} />
          </div>
        </div>
      )}
      {children.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Children</div>
          <div className="text-xs ml-2 flex flex-wrap gap-x-0.5">
            {children.map((c, i) => (
              <span key={i}>
                <TermLink name={c.name} displayName={c.display_name} />
                {i < children.length - 1 && <span className="text-gray-400">, </span>}
              </span>
            ))}
          </div>
        </div>
      )}
      {relationships.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Relationships</div>
          {relationships.map((r) => (
            <RelationshipRow
              key={r.id}
              rel={r}
              sessionId={sessionId}
              onDeleted={(id) => setDetail(prev => ({
                ...prev,
                relationships: prev.relationships.filter(x => x.id !== id),
              }))}
              onUpdated={(id, verb) => setDetail(prev => ({
                ...prev,
                relationships: prev.relationships.map(x => x.id === id ? { ...x, verb, user_edited: true } : x),
              }))}
              onApproved={(id) => setDetail(prev => ({
                ...prev,
                relationships: prev.relationships.map(x => x.id === id ? { ...x, user_edited: true } : x),
              }))}
            />
          ))}
        </div>
      )}
      <AddRelationshipRow
        sessionId={sessionId}
        termName={termName}
        onCreated={(rel) => setDetail(prev => ({
          ...prev,
          relationships: [...prev.relationships, rel],
        }))}
      />
      {cluster_siblings.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Cluster</div>
          <div className="text-xs ml-2 flex flex-wrap gap-x-0.5">
            {cluster_siblings.map((name, i) => (
              <span key={i}>
                <TermLink name={name} displayName={name} />
                {i < cluster_siblings.length - 1 && <span className="text-gray-400">, </span>}
              </span>
            ))}
          </div>
        </div>
      )}
      {resources.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Connected Resources</div>
          {resources.map((r, i) => (
            <div key={i} className="text-xs text-gray-500 dark:text-gray-400 ml-2 mb-0.5">
              <span className="font-medium">{r.entity_name}</span>
              <span className="text-gray-400"> ({r.entity_type})</span>
              {r.sources.map((s, j) => (
                <div key={j} className="ml-2">
                  <SourceLink source={s.source} documentName={s.document_name} section={s.section} />
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
      {hasConnections && (
        <div>
          <button
            onClick={() => setGraphOpen(!graphOpen)}
            className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ChevronRightIcon className={`w-3 h-3 transition-transform ${graphOpen ? 'rotate-90' : ''}`} />
            Graph
          </button>
          {graphOpen && <TermGraphInline sessionId={sessionId} termName={termName} />}
        </div>
      )}
    </div>
  )
}

// Alias editor — add, remove, edit aliases inline
function AliasEditor({ term, sessionId }: { term: GlossaryTerm; sessionId: string }) {
  const [adding, setAdding] = useState(false)
  const [newAlias, setNewAlias] = useState('')
  const [editIdx, setEditIdx] = useState<number | null>(null)
  const [editValue, setEditValue] = useState('')
  const [drafting, setDrafting] = useState(false)
  const { fetchTerms } = useGlossaryStore()

  const aliases = term.aliases || []

  const save = async (updated: string[]) => {
    await updateGlossaryTerm(sessionId, term.name, { aliases: updated })
    fetchTerms(sessionId)
  }

  const handleAdd = async () => {
    const val = newAlias.trim()
    if (!val) return
    await save([...aliases, val])
    setNewAlias('')
    setAdding(false)
  }

  const handleDelete = async (idx: number) => {
    await save(aliases.filter((_, i) => i !== idx))
  }

  const handleEditSave = async () => {
    if (editIdx === null) return
    const val = editValue.trim()
    if (!val || val === aliases[editIdx]) { setEditIdx(null); return }
    const updated = [...aliases]
    updated[editIdx] = val
    await save(updated)
    setEditIdx(null)
  }

  const handleDraft = async () => {
    setDrafting(true)
    try {
      const result = await draftGlossaryAliases(sessionId, term.name)
      if (result.aliases && result.aliases.length > 0) {
        // Merge with existing, dedup case-insensitive (keep first occurrence)
        const seen = new Set(aliases.map(a => a.toLowerCase()))
        const merged = [...aliases]
        for (const a of result.aliases) {
          if (!seen.has(a.toLowerCase())) {
            seen.add(a.toLowerCase())
            merged.push(a)
          }
        }
        await save(merged)
      }
    } catch (err) {
      console.error('Draft aliases failed:', err)
    } finally {
      setDrafting(false)
    }
  }

  return (
    <div className="text-xs text-gray-500 dark:text-gray-400">
      <span className="font-medium">Aliases:</span>
      {aliases.length === 0 && !adding && (
        <span className="text-gray-400 ml-1">none</span>
      )}
      {aliases.map((a, i) => (
        <span key={i} className="inline-flex items-center ml-1">
          {editIdx === i ? (
            <input
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleEditSave()
                if (e.key === 'Escape') setEditIdx(null)
              }}
              onBlur={handleEditSave}
              className="px-1 py-0 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 w-20"
              autoFocus
            />
          ) : (
            <span
              className="cursor-pointer hover:text-gray-700 dark:hover:text-gray-200"
              onClick={() => { setEditIdx(i); setEditValue(a) }}
            >
              {a}
            </span>
          )}
          <button
            onClick={() => handleDelete(i)}
            className="ml-0.5 text-red-400 hover:text-red-500"
            title="Remove alias"
          >
            <XMarkIcon className="w-2.5 h-2.5" />
          </button>
          {i < aliases.length - 1 && <span>,</span>}
        </span>
      ))}
      {adding ? (
        <span className="inline-flex items-center ml-1">
          <input
            value={newAlias}
            onChange={(e) => setNewAlias(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleAdd()
              if (e.key === 'Escape') setAdding(false)
            }}
            onBlur={() => { if (!newAlias.trim()) setAdding(false) }}
            className="px-1 py-0 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 w-20"
            placeholder="alias"
            autoFocus
          />
        </span>
      ) : (
        <>
          <button
            onClick={() => setAdding(true)}
            className="ml-1 text-blue-500 hover:text-blue-600"
            title="Add alias"
          >
            <PlusIcon className="w-3 h-3 inline" />
          </button>
          <button
            onClick={handleDraft}
            disabled={drafting}
            className="ml-1 inline-flex items-center gap-0.5 text-purple-500 hover:text-purple-600 disabled:opacity-50"
            title="AI-generate aliases"
          >
            <SparklesIcon className="w-3 h-3" />
            <span className="text-[10px]">{drafting ? 'Drafting...' : 'AI Draft'}</span>
          </button>
        </>
      )}
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
  const [drafting, setDrafting] = useState(false)
  const { addDefinition } = useGlossaryStore()

  const handleSubmit = async () => {
    if (!definition.trim()) return
    await addDefinition(sessionId, name, definition.trim())
    onDone()
  }

  const handleDraft = async () => {
    setDrafting(true)
    try {
      const result = await draftGlossaryDefinition(sessionId, name)
      if (result.draft) setDefinition(result.draft)
    } catch (err) {
      console.error('Draft generation failed:', err)
    } finally {
      setDrafting(false)
    }
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
          onClick={handleDraft}
          disabled={drafting}
          className="flex items-center gap-0.5 text-xs px-2 py-0.5 text-purple-500 hover:text-purple-600 disabled:opacity-50"
          title="AI-generate a draft definition"
        >
          <SparklesIcon className="w-3 h-3" />
          {drafting ? 'Drafting...' : 'AI Draft'}
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
  const { terms: allTerms, selectedName, updateTerm, refineTerm, deleteTerm, renameTerm } = useGlossaryStore()
  const isSelected = selectedName?.toLowerCase() === term.name.toLowerCase()
  const [isOpen, setIsOpen] = useState(false)
  const [isDefining, setIsDefining] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editDef, setEditDef] = useState('')
  const [isRenaming, setIsRenaming] = useState(false)
  const [renameTo, setRenameTo] = useState('')
  const isIgnored = term.ignored === true

  // Can rename: defined term with no entity (abstract only)
  const canRename = term.glossary_status === 'defined' && !term.entity_id

  const handleRename = async () => {
    const trimmed = renameTo.trim()
    if (!trimmed || trimmed.toLowerCase() === term.name.toLowerCase()) {
      setIsRenaming(false)
      return
    }
    await renameTerm(sessionId, term.name, trimmed)
    setIsRenaming(false)
  }

  // Auto-open when deep-linked via selectTerm
  useEffect(() => {
    if (isSelected && !isOpen) {
      setIsOpen(true)
    }
  }, [isSelected])

  const typeColor =
    TYPE_COLORS[term.semantic_type || ''] ||
    'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'

  const statusColor = STATUS_COLORS[term.status || ''] || ''

  const isDefined = term.glossary_status === 'defined'

  // Resolve parent display name from the terms list
  const parentName = useMemo(() => {
    if (!term.parent_id) return null
    const parent = allTerms.find(
      (t) => t.glossary_id === term.parent_id || t.entity_id === term.parent_id
    )
    return parent?.display_name || null
  }, [term.parent_id, allTerms])

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
    <div id={`glossary-term-${term.name}`} className="border-b border-gray-100 dark:border-gray-700 last:border-b-0">
      <div
        role="button"
        onClick={() => setIsOpen(!isOpen)}
        className="group w-full flex items-center gap-2 py-2 px-1 text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors cursor-pointer"
        style={{ paddingLeft: `${depth * 16 + 4}px` }}
      >
        <ChevronRightIcon
          className={`w-3 h-3 text-gray-400 transition-transform flex-shrink-0 ${
            isOpen ? 'rotate-90' : ''
          }`}
        />
        {isRenaming ? (
          <input
            autoFocus
            value={renameTo}
            onChange={(e) => setRenameTo(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleRename()
              if (e.key === 'Escape') setIsRenaming(false)
            }}
            onBlur={handleRename}
            onClick={(e) => e.stopPropagation()}
            className="text-sm font-medium text-gray-700 dark:text-gray-300 flex-1 bg-white dark:bg-gray-800 border border-blue-400 rounded px-1 py-0 outline-none min-w-0"
          />
        ) : (
          <span className={`text-sm font-medium flex-1 truncate ${isIgnored ? 'text-gray-400 dark:text-gray-500 line-through opacity-60' : 'text-gray-700 dark:text-gray-300'}`}>
            {term.display_name}
            {parentName && (
              <span className="text-xs font-normal text-gray-400 dark:text-gray-500 ml-1">
                &larr; {parentName}
              </span>
            )}
          </span>
        )}
        {canRename && !isRenaming && (
          <span
            role="button"
            onClick={(e) => { e.stopPropagation(); setRenameTo(term.display_name); setIsRenaming(true) }}
            className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 flex-shrink-0 text-gray-300 hover:text-gray-500 opacity-0 group-hover:opacity-100 transition-opacity"
            title="Rename term"
          >
            <PencilIcon className="w-3 h-3" />
          </span>
        )}
        <span
          role="button"
          onClick={(e) => { e.stopPropagation(); updateTerm(sessionId, term.name, { ignored: !isIgnored }) }}
          className={`p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 flex-shrink-0 transition-opacity ${isIgnored ? 'text-amber-500 opacity-100' : 'text-gray-300 hover:text-gray-500 opacity-0 group-hover:opacity-100'}`}
          title={isIgnored ? 'Show in graph & search' : 'Hide from graph & search'}
        >
          {isIgnored ? <EyeSlashIcon className="w-3 h-3" /> : <EyeIcon className="w-3 h-3" />}
        </span>
        {term.semantic_type && (
          <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 ${typeColor}`}>
            {term.semantic_type}
          </span>
        )}
        <DomainPromotePicker
          termName={term.name}
          currentDomain={term.domain || null}
          termStatus={term.status || undefined}
          sessionId={sessionId}
        />
        {isDefined && term.status && (
          <span className={`text-xs flex-shrink-0 ${statusColor}`}>{term.status}</span>
        )}
        {term.domain && (
          <span
            className="text-xs px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 flex-shrink-0"
            title={term.domain_path || term.domain}
          >
            {term.domain_path || term.domain}
          </span>
        )}
        <span
          role="button"
          onClick={(e) => { e.stopPropagation(); handleDelete() }}
          className="p-0.5 rounded hover:bg-red-100 dark:hover:bg-red-900/30 flex-shrink-0 text-gray-300 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
          title="Remove term"
        >
          <TrashIcon className="w-3 h-3" />
        </span>
      </div>

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

          {/* Aliases (editable) */}
          <AliasEditor term={term} sessionId={sessionId} />

          {/* Domain already shown as badge in header row */}

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
  const isDomainFolder = node.term.name.startsWith('__domain__')

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
              <MinusSmallIcon className="w-3 h-3 text-gray-400" />
            ) : (
              <PlusSmallIcon className="w-3 h-3 text-gray-400" />
            )}
          </button>
        )}
        {isDomainFolder ? (
          <div
            className="flex-1 text-xs font-semibold text-gray-600 dark:text-gray-300 py-1 px-1"
            style={{ marginLeft: node.children.length === 0 ? `${depth * 12 + 16}px` : '0' }}
          >
            {node.term.display_name}
            <span className="ml-1 text-gray-400 font-normal">({node.children.length})</span>
          </div>
        ) : (
          <div className="flex-1" style={{ marginLeft: node.children.length === 0 ? `${depth * 12 + 16}px` : '0' }}>
            <GlossaryItem term={node.term} sessionId={sessionId} depth={0} />
          </div>
        )}
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
  const { deprecatedTerms, deleteTerm, reconnectTerm, terms: allTerms } = useGlossaryStore()
  const [reconnecting, setReconnecting] = useState<string | null>(null)
  const [selectedParent, setSelectedParent] = useState('')
  const [reconnectVerb, setReconnectVerb] = useState<'HAS_ONE' | 'HAS_KIND' | 'HAS_MANY'>('HAS_KIND')

  if (deprecatedTerms.length === 0) return null

  const handleReconnect = async (name: string) => {
    if (!selectedParent) return
    await reconnectTerm(sessionId, name, { parent_id: selectedParent })
    // Also set the parent_verb via updateTerm
    const { updateTerm } = useGlossaryStore.getState()
    await updateTerm(sessionId, name, { parent_verb: reconnectVerb })
    setReconnecting(null)
    setSelectedParent('')
    setReconnectVerb('HAS_KIND')
  }

  return (
    <div className="border border-amber-200 dark:border-amber-800 rounded p-2 bg-amber-50 dark:bg-amber-900/20">
      <div className="flex items-center gap-1 text-xs font-medium text-amber-700 dark:text-amber-400 mb-1.5">
        <ExclamationTriangleIcon className="w-3.5 h-3.5" />
        Deprecated ({deprecatedTerms.length})
      </div>
      {deprecatedTerms.map((t) => (
        <div key={t.name} className="py-1 border-b border-amber-100 dark:border-amber-800 last:border-0">
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-600 dark:text-gray-400 flex-1">{t.display_name}</span>
            <span className="text-gray-400 truncate max-w-[120px]">{t.definition}</span>
            <button
              onClick={() => { setReconnecting(reconnecting === t.name ? null : t.name); setSelectedParent('') }}
              className="text-blue-400 hover:text-blue-500 flex-shrink-0 text-xs"
              title="Reconnect to parent"
            >
              Reconnect
            </button>
            <button
              onClick={() => deleteTerm(sessionId, t.name)}
              className="text-red-400 hover:text-red-500 flex-shrink-0"
              title="Delete deprecated term"
            >
              <TrashIcon className="w-3.5 h-3.5" />
            </button>
          </div>
          {reconnecting === t.name && (
            <div className="flex items-center gap-1 mt-1">
              <select
                value={reconnectVerb}
                onChange={(e) => setReconnectVerb(e.target.value as 'HAS_ONE' | 'HAS_KIND' | 'HAS_MANY')}
                className="text-xs py-0.5 px-1 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 w-20"
              >
                <option value="HAS_ONE">HAS_ONE</option>
                <option value="HAS_KIND">HAS_KIND</option>
                <option value="HAS_MANY">HAS_MANY</option>
              </select>
              <select
                value={selectedParent}
                onChange={(e) => setSelectedParent(e.target.value)}
                className="flex-1 text-xs py-0.5 px-1 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300"
              >
                <option value="">Select parent...</option>
                {allTerms.filter(at => at.name !== t.name).map(at => (
                  <option key={at.name} value={at.glossary_id || at.entity_id || at.name}>{at.display_name}</option>
                ))}
              </select>
              <button
                onClick={() => handleReconnect(t.name)}
                disabled={!selectedParent}
                className="text-xs px-1.5 py-0.5 rounded bg-blue-500 text-white disabled:opacity-50"
              >
                Save
              </button>
            </div>
          )}
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

// Domain move picker — assign/reassign/unassign a term's domain
// Any term can be moved between domains. Cascade moves descendants too.
// Collect all descendant terms (recursive via parent_id)
function getDescendants(termName: string, allTerms: GlossaryTerm[]): GlossaryTerm[] {
  const term = allTerms.find(t => t.name.toLowerCase() === termName.toLowerCase())
  if (!term) return []
  const parentIds = new Set<string>()
  if (term.glossary_id) parentIds.add(term.glossary_id)
  if (term.entity_id) parentIds.add(term.entity_id)

  const children = allTerms.filter(t => t.parent_id && parentIds.has(t.parent_id))
  const result: GlossaryTerm[] = []
  for (const child of children) {
    result.push(child)
    result.push(...getDescendants(child.name, allTerms))
  }
  return result
}

function DomainPromotePicker({
  termName,
  currentDomain,
  sessionId,
}: {
  termName: string
  currentDomain: string | null
  termStatus?: string | undefined
  sessionId: string
}) {
  const [open, setOpen] = useState(false)
  const [treeNodes, setTreeNodes] = useState<DomainTreeNode[]>([])
  const [loading, setLoading] = useState(false)
  const [canPromoteToSystem, setCanPromoteToSystem] = useState(false)
  const [confirmCascade, setConfirmCascade] = useState<{ target: string | null; descendants: GlossaryTerm[] } | null>(null)
  const { terms: allTerms, updateTerm } = useGlossaryStore()

  const selfTerm = allTerms.find(t => t.name.toLowerCase() === termName.toLowerCase())
  if (!selfTerm) return null

  const handleOpen = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setOpen(true)
    setLoading(true)
    const { useAuthStore } = await import('@/store/authStore')
    setCanPromoteToSystem(useAuthStore.getState().canWrite('tier_promote'))
    const activeDomains = useSessionStore.getState().session?.active_domains || []
    try {
      const tree = await getDomainTree()
      const filterTree = (nodes: DomainTreeNode[]): DomainTreeNode[] =>
        activeDomains.length > 0
          ? nodes.filter(n => activeDomains.includes(n.filename)).map(n => ({ ...n, children: filterTree(n.children) }))
          : nodes
      setTreeNodes(filterTree(tree))
    } catch {
      const { domains: allDomains } = await listDomains()
      const flat = (activeDomains.length > 0
        ? allDomains.filter(d => activeDomains.includes(d.filename))
        : allDomains
      ).map(d => ({ ...d, path: '', databases: [], apis: [], documents: [], children: [] }))
      setTreeNodes(flat)
    }
    setLoading(false)
  }

  const checkAndMove = (target: string | null) => {
    const descendants = getDescendants(termName, allTerms)
    if (descendants.length > 0) {
      setConfirmCascade({ target, descendants })
      return
    }
    executeMove(target, [])
  }

  const executeMove = async (target: string | null, descendants: GlossaryTerm[]) => {
    await updateTerm(sessionId, termName, { domain: target || '' })
    for (const d of descendants) {
      if (d.domain !== target) {
        await updateTerm(sessionId, d.name, { domain: target || '' })
      }
    }
    setOpen(false)
    setConfirmCascade(null)
  }

  return (
    <span className="flex-shrink-0" onClick={(e) => e.stopPropagation()}>
      <span
        role="button"
        onClick={handleOpen}
        className={`p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 inline-flex ${currentDomain ? 'text-blue-500' : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'}`}
        title={currentDomain ? `Move from ${currentDomain}` : 'Move to domain'}
      >
        <ArrowUpIcon className="w-3 h-3" />
      </span>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={(e) => { if (e.target === e.currentTarget) { setOpen(false); setConfirmCascade(null) } }}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm w-full mx-4 p-4 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            {confirmCascade ? (
              <>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Move with {confirmCascade.descendants.length} descendant{confirmCascade.descendants.length > 1 ? 's' : ''}?
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 ml-2 max-h-24 overflow-y-auto">
                  {confirmCascade.descendants.map(d => (
                    <div key={d.name}>{d.display_name}</div>
                  ))}
                </div>
                <div className="flex justify-end gap-2 pt-1">
                  <button
                    onClick={() => setConfirmCascade(null)}
                    className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Back
                  </button>
                  <button
                    onClick={() => executeMove(confirmCascade.target, confirmCascade.descendants)}
                    className="text-xs px-3 py-1.5 rounded bg-blue-500 text-white hover:bg-blue-600"
                  >
                    Move all
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Move "{selfTerm.display_name}"
                </div>
                {currentDomain && (
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Currently in: <span className="font-medium">{currentDomain}</span>
                  </div>
                )}
                {loading ? (
                  <div className="text-xs text-gray-400 py-2 text-center">Loading domains...</div>
                ) : (
                  <div className="space-y-1 max-h-48 overflow-y-auto">
                    {currentDomain && (
                      <button
                        onClick={() => checkAndMove(null)}
                        className="w-full text-left text-xs px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700 mb-1 pb-2"
                      >
                        <div className="font-medium">Unassign domain</div>
                      </button>
                    )}
                    {(() => {
                      const renderNode = (node: DomainTreeNode, depth: number = 0): React.ReactNode => (
                        <div key={node.filename}>
                          {node.filename !== currentDomain && (
                            <button
                              onClick={() => checkAndMove(node.filename)}
                              className="w-full text-left text-xs px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                              style={{ paddingLeft: `${depth * 12 + 12}px` }}
                            >
                              <div className="font-medium">{node.name}</div>
                              <div className="text-gray-400">{node.path || node.filename}</div>
                            </button>
                          )}
                          {node.children?.map(child => renderNode(child, depth + 1))}
                        </div>
                      )
                      return treeNodes.map(n => renderNode(n))
                    })()}
                    {canPromoteToSystem && currentDomain !== 'system' && (
                      <button
                        onClick={() => checkAndMove('system')}
                        className="w-full text-left text-xs px-3 py-2 rounded hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-600 dark:text-purple-400 border-t border-gray-100 dark:border-gray-700 mt-1 pt-2"
                      >
                        <div className="font-medium">Move to system</div>
                        <div className="text-purple-400 dark:text-purple-500">Available to all sessions</div>
                      </button>
                    )}
                    {treeNodes.length === 0 && !canPromoteToSystem && !currentDomain && (
                      <div className="text-xs text-gray-400 py-2 text-center">No domains available</div>
                    )}
                  </div>
                )}
                <div className="flex justify-end pt-1">
                  <button
                    onClick={() => setOpen(false)}
                    className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    Cancel
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </span>
  )
}

// Term neighborhood graph modal
interface PositionedNode {
  id: string
  label: string
  type: 'focal' | 'parent' | 'child' | 'relationship' | 'sibling'
  depth: number
  domain?: string | null
  x: number
  y: number
}

interface PositionedEdge {
  sourceId: string
  targetId: string
  label: string
  type: 'parent' | 'child' | 'relationship' | 'sibling'
}

const NODE_STYLES: Record<string, { fill: string; r: number }> = {
  focal: { fill: '#3b82f6', r: 24 },
  parent: { fill: '#a855f7', r: 18 },
  child: { fill: '#22c55e', r: 16 },
  relationship: { fill: '#9ca3af', r: 16 },
  sibling: { fill: '#f59e0b', r: 14 },
}

// Convex hull (Graham scan) for cluster backgrounds
function convexHull(points: { x: number; y: number }[]): { x: number; y: number }[] {
  if (points.length < 3) return points
  const pts = [...points].sort((a, b) => a.x - b.x || a.y - b.y)
  const cross = (o: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
  const lower: { x: number; y: number }[] = []
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p) }
  const upper: { x: number; y: number }[] = []
  for (const p of pts.reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p) }
  return lower.slice(0, -1).concat(upper.slice(0, -1))
}

// Expand hull points outward from centroid by padding
function expandHull(hull: { x: number; y: number }[], pad: number): { x: number; y: number }[] {
  if (hull.length === 0) return hull
  const cx = hull.reduce((s, p) => s + p.x, 0) / hull.length
  const cy = hull.reduce((s, p) => s + p.y, 0) / hull.length
  return hull.map(p => {
    const dx = p.x - cx, dy = p.y - cy
    const d = Math.sqrt(dx * dx + dy * dy) || 1
    return { x: p.x + (dx / d) * pad, y: p.y + (dy / d) * pad }
  })
}

const CLUSTER_COLORS = [
  'rgba(59,130,246,0.08)',   // blue
  'rgba(168,85,247,0.08)',   // purple
  'rgba(34,197,94,0.08)',    // green
  'rgba(249,115,22,0.08)',   // orange
  'rgba(236,72,153,0.08)',   // pink
  'rgba(20,184,166,0.08)',   // teal
  'rgba(234,179,8,0.08)',    // yellow
]

const CLUSTER_STROKES = [
  'rgba(59,130,246,0.25)',
  'rgba(168,85,247,0.25)',
  'rgba(34,197,94,0.25)',
  'rgba(249,115,22,0.25)',
  'rgba(236,72,153,0.25)',
  'rgba(20,184,166,0.25)',
  'rgba(234,179,8,0.25)',
]

// Tooltip for graph node hover — read-only detail card
function NodeTooltip({ sessionId, termName }: { sessionId: string; termName: string }) {
  const [data, setData] = useState<(GlossaryTerm & { grounded: boolean; connected_resources: Array<{ entity_name: string; entity_type: string; sources: Array<{ document_name: string; source: string }> }> }) | null>(null)

  useEffect(() => {
    let cancelled = false
    getGlossaryTerm(sessionId, termName)
      .then((d) => { if (!cancelled) setData(d) })
      .catch(() => {})
    return () => { cancelled = true }
  }, [sessionId, termName])

  if (!data) return <div className="text-[10px] text-gray-400 px-2 py-1">Loading...</div>

  return (
    <div className="space-y-1 max-w-[280px]">
      <div className="font-semibold text-[11px] text-gray-800 dark:text-gray-100">{data.display_name}</div>
      {data.definition && (
        <p className="text-[10px] text-gray-600 dark:text-gray-400 italic">&ldquo;{data.definition}&rdquo;</p>
      )}
      {data.aliases?.length > 0 && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Aliases:</span> {data.aliases.join(', ')}
        </div>
      )}
      {(data.domain_path || data.domain) && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Domain:</span> {data.domain_path || data.domain}
        </div>
      )}
      {data.provenance && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Provenance:</span> {data.provenance}
        </div>
      )}
      {data.parent && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Parent ({data.parent_verb || 'HAS_KIND'}):</span> {data.parent.display_name}
        </div>
      )}
      {data.children && data.children.length > 0 && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Children:</span> {data.children.map(c => c.display_name).join(', ')}
        </div>
      )}
      {data.relationships && data.relationships.length > 0 && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Relationships:</span>
          {data.relationships.map((r, i) => (
            <div key={i} className="ml-1">{r.subject} <span className="text-gray-400">{r.verb}</span> {r.object}</div>
          ))}
        </div>
      )}
      {data.connected_resources && data.connected_resources.length > 0 && (
        <div className="text-[10px] text-gray-500 dark:text-gray-400">
          <span className="font-medium">Resources:</span>
          {data.connected_resources.map((r, i) => (
            <div key={i} className="ml-1">{r.entity_name} <span className="text-gray-400">({r.entity_type})</span></div>
          ))}
        </div>
      )}
      {data.status && (
        <div className="text-[10px] text-gray-400">Status: {data.status}</div>
      )}
    </div>
  )
}

const GRAPH_W = 800
const GRAPH_H = 600
const GRAPH_MODAL_W = 1200
const GRAPH_MODAL_H = 900

// Domain filter for graph — compact multi-select dropdown
function GraphDomainFilter({
  domains,
  selected,
  onChange,
}: {
  domains: string[]
  selected: Set<string> | null
  onChange: (v: Set<string> | null) => void
}) {
  const [open, setOpen] = useState(false)
  const allSelected = !selected || selected.size === 0
  const label = allSelected ? 'All domains' : `${selected!.size} domain${selected!.size > 1 ? 's' : ''}`

  return (
    <div className="relative ml-1">
      <button
        onClick={() => setOpen(!open)}
        className={`text-[10px] py-0.5 px-1.5 border rounded flex items-center gap-0.5 ${
          allSelected
            ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400'
            : 'border-blue-300 dark:border-blue-700 bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400'
        }`}
      >
        {label}
        <ChevronRightIcon className={`w-2.5 h-2.5 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="absolute left-0 top-full mt-0.5 z-50 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 shadow-lg py-1 min-w-[120px] max-h-40 overflow-y-auto">
          <button
            onClick={() => { onChange(null); setOpen(false) }}
            className={`block w-full text-left text-[10px] px-2 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-700 ${allSelected ? 'text-blue-600 font-medium' : 'text-gray-600 dark:text-gray-400'}`}
          >
            All
          </button>
          {domains.map(d => {
            const checked = !selected || selected.has(d)
            return (
              <label key={d} className="flex items-center gap-1.5 px-2 py-0.5 text-[10px] text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => {
                    const next = new Set(selected || domains)
                    if (next.has(d)) next.delete(d)
                    else next.add(d)
                    onChange(next.size === domains.length || next.size === 0 ? null : next)
                  }}
                  className="w-2.5 h-2.5 accent-blue-500"
                />
                {d}
              </label>
            )
          })}
        </div>
      )}
    </div>
  )
}

function TermGraphInline({
  sessionId,
  termName,
}: {
  sessionId: string
  termName: string
}) {
  // Persist graph settings to localStorage
  const GRAPH_STORAGE_KEY = 'constat-graph-settings'
  const loadSetting = <T,>(key: string, fallback: T): T => {
    try {
      const raw = localStorage.getItem(GRAPH_STORAGE_KEY)
      if (raw) { const obj = JSON.parse(raw); if (key in obj) return obj[key] as T }
    } catch { /* ignore */ }
    return fallback
  }
  const saveSetting = (key: string, value: unknown) => {
    try {
      const raw = localStorage.getItem(GRAPH_STORAGE_KEY)
      const obj = raw ? JSON.parse(raw) : {}
      obj[key] = value
      localStorage.setItem(GRAPH_STORAGE_KEY, JSON.stringify(obj))
    } catch { /* ignore */ }
  }

  const [graph, setGraph] = useState<{ nodes: PositionedNode[]; edges: PositionedEdge[] } | null>(null)
  const [loading, setLoading] = useState(true)
  const [empty, setEmpty] = useState(false)
  const [depth, setDepth] = useState(() => loadSetting('depth', 1))
  const [fullscreen, setFullscreen] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })
  const [layout, setLayout] = useState<'force' | 'tree-v' | 'tree-h'>(() => loadSetting('layout', 'force'))
  const [nodeSpacing, setNodeSpacing] = useState(() => loadSetting('nodeSpacing', 100))
  const [showLeaves, setShowLeaves] = useState(() => loadSetting('showLeaves', true))
  const [showClusters, setShowClusters] = useState(() => loadSetting('showClusters', true))
  const [showDomains, setShowDomains] = useState(() => loadSetting('showDomains', true))
  const [forceIterations, setForceIterations] = useState(() => loadSetting('forceIterations', 200))
  const [graphHeight, setGraphHeight] = useState(() => loadSetting('graphHeight', 300))
  const simRef = useRef<ReturnType<typeof forceSimulation> | null>(null)
  const animFrameRef = useRef<number>(0)
  const shouldAnimateRef = useRef(false)
  const pinnedNodesRef = useRef<Map<string, { fx: number; fy: number }>>(new Map())
  const [graphDomains, setGraphDomains] = useState<string[]>([])
  const [graphDomainFilter, setGraphDomainFilter] = useState<Set<string> | null>(null)
  const [hoverNode, setHoverNode] = useState<{ name: string; x: number; y: number } | null>(null)
  const hoverTimeout = useRef<ReturnType<typeof setTimeout> | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerSize, setContainerSize] = useState<{ w: number; h: number } | null>(null)
  const [showExportMenu, setShowExportMenu] = useState(false)

  // Persist settings to localStorage on change
  useEffect(() => { saveSetting('depth', depth) }, [depth])
  useEffect(() => { saveSetting('layout', layout) }, [layout])
  useEffect(() => { saveSetting('nodeSpacing', nodeSpacing) }, [nodeSpacing])
  useEffect(() => { saveSetting('showLeaves', showLeaves) }, [showLeaves])
  useEffect(() => { saveSetting('showClusters', showClusters) }, [showClusters])
  useEffect(() => { saveSetting('showDomains', showDomains) }, [showDomains])
  useEffect(() => { saveSetting('forceIterations', forceIterations) }, [forceIterations])
  useEffect(() => { saveSetting('graphHeight', graphHeight) }, [graphHeight])

  // Track container size with ResizeObserver — debounced to avoid render loops
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let timer: ReturnType<typeof setTimeout> | null = null
    const obs = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect
      if (!r || r.width <= 0 || r.height <= 0) return
      const nw = Math.round(r.width)
      const nh = Math.round(r.height)
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => {
        setContainerSize(prev => {
          if (prev && Math.abs(prev.w - nw) < 10 && Math.abs(prev.h - nh) < 10) return prev
          return { w: nw, h: nh }
        })
      }, 200)
    })
    obs.observe(el)
    return () => { obs.disconnect(); if (timer) clearTimeout(timer) }
  }, [fullscreen, graph])

  const gw = containerSize?.w || (fullscreen ? GRAPH_MODAL_W : GRAPH_W)
  const gh = containerSize?.h || (fullscreen ? GRAPH_MODAL_H : GRAPH_H)

  // BFS fetch neighborhood up to `depth` levels, then simulate layout
  useEffect(() => {
    let cancelled = false
    cancelAnimationFrame(animFrameRef.current)
    setLoading(true)
    setEmpty(false)
    setGraph(null)
    graphIdRef.current += 1
    needsFitRef.current = true

    async function fetchNeighborhood() {
      const nodeMap = new Map<string, { id: string; label: string; type: PositionedNode['type']; depth: number; domain?: string | null }>()
      const rawEdges: { source: string; target: string; label: string; type: PositionedEdge['type'] }[] = []
      const edgeSet = new Set<string>()

      const addEdge = (src: string, tgt: string, label: string, type: PositionedEdge['type']) => {
        const key = `${src}|${tgt}|${label}`
        if (edgeSet.has(key)) return
        edgeSet.add(key)
        rawEdges.push({ source: src, target: tgt, label, type })
      }

      let queue: [string, number][] = [[termName, 0]]
      const visited = new Set<string>()

      while (queue.length > 0) {
        const frontier = queue.filter(([name]) => !visited.has(name))
        queue = []
        if (frontier.length === 0) break
        for (const [name] of frontier) visited.add(name)

        const results = await Promise.all(
          frontier.map(([name, d]) =>
            getGlossaryTerm(sessionId, name)
              .then((data) => ({ name, d, data }))
              .catch(() => null)
          )
        )

        if (cancelled) return

        for (const res of results) {
          if (!res) continue
          const { name, d, data } = res

          // Skip ignored terms (except the focal node)
          if (d > 0 && data.ignored) continue

          // Add the fetched node itself
          if (!nodeMap.has(name)) {
            nodeMap.set(name, {
              id: name,
              label: data.display_name || name,
              type: d === 0 ? 'focal' : 'relationship',
              depth: d,
              domain: data.domain,
            })
          }

          // Parent
          if (data.parent) {
            const pid = data.parent.name
            if (!nodeMap.has(pid)) {
              nodeMap.set(pid, { id: pid, label: data.parent.display_name, type: 'parent', depth: d + 1, domain: data.domain })
            }
            addEdge(pid, name, data.parent_verb || 'HAS_KIND', 'parent')
            if (d + 1 < depth) queue.push([pid, d + 1])
          }

          // Children
          for (const c of data.children || []) {
            if (!nodeMap.has(c.name)) {
              nodeMap.set(c.name, { id: c.name, label: c.display_name, type: 'child', depth: d + 1, domain: data.domain })
            }
            addEdge(name, c.name, c.parent_verb || 'HAS_KIND', 'child')
            if (d + 1 < depth) queue.push([c.name, d + 1])
          }

          // Relationships
          for (const r of data.relationships || []) {
            const partner = r.subject === name ? r.object : r.subject
            if (!nodeMap.has(partner)) {
              nodeMap.set(partner, { id: partner, label: partner, type: 'relationship', depth: d + 1, domain: data.domain })
            }
            addEdge(
              r.subject === name ? name : partner,
              r.subject === name ? partner : name,
              r.verb,
              'relationship',
            )
            if (d + 1 < depth) queue.push([partner, d + 1])
          }

          // Cluster siblings — only for the focal node (depth 0)
          if (d === 0 && showClusters) {
            for (const sib of data.cluster_siblings || []) {
              if (!nodeMap.has(sib)) {
                nodeMap.set(sib, { id: sib, label: sib, type: 'sibling', depth: 1, domain: data.domain })
              }
              addEdge(name, sib, 'cluster', 'sibling')
            }
          }
        }
      }

      if (cancelled) return

      let nodeArr = Array.from(nodeMap.values())

      // Collect unique domains found in the graph
      const domainSet = new Set<string>()
      for (const n of nodeArr) {
        if (n.domain) domainSet.add(n.domain)
      }
      if (!cancelled) setGraphDomains(Array.from(domainSet).sort())

      // Apply domain filter (focal node always kept)
      if (graphDomainFilter && graphDomainFilter.size > 0) {
        const kept = new Set(nodeArr.filter(n => n.type === 'focal' || !n.domain || graphDomainFilter.has(n.domain)).map(n => n.id))
        nodeArr = nodeArr.filter(n => kept.has(n.id))
        for (let i = rawEdges.length - 1; i >= 0; i--) {
          if (!kept.has(rawEdges[i].source) || !kept.has(rawEdges[i].target)) rawEdges.splice(i, 1)
        }
      }

      // Filter out leaf nodes (single-edge terminals) when toggle is off
      if (!showLeaves && nodeArr.length > 2) {
        const edgeCount = new Map<string, number>()
        for (const e of rawEdges) {
          edgeCount.set(e.source, (edgeCount.get(e.source) || 0) + 1)
          edgeCount.set(e.target, (edgeCount.get(e.target) || 0) + 1)
        }
        const leafIds = new Set(
          nodeArr.filter(n => n.type !== 'focal' && (edgeCount.get(n.id) || 0) <= 1).map(n => n.id)
        )
        if (leafIds.size < nodeArr.length - 1) {
          nodeArr = nodeArr.filter(n => !leafIds.has(n.id))
          // Remove edges referencing pruned nodes
          const kept = new Set(nodeArr.map(n => n.id))
          for (let i = rawEdges.length - 1; i >= 0; i--) {
            if (!kept.has(rawEdges[i].source) || !kept.has(rawEdges[i].target)) rawEdges.splice(i, 1)
          }
        }
      }

      if (nodeArr.length <= 1) {
        setEmpty(true)
        setLoading(false)
        return
      }

      let posNodes: PositionedNode[]
      const cx = gw / 2
      const cy = gh / 2

      if (layout === 'force') {
        // Force-directed layout
        const nodeCount = nodeArr.length
        const edgeCount = rawEdges.length
        const density = edgeCount / Math.max(nodeCount, 1)
        const linkDist = nodeSpacing * (nodeCount > 15 ? 0.67 : 1)
        const chargeStr = -(nodeSpacing * (density > 2 ? 5 : nodeCount > 15 ? 3 : 3.3))
        const collideExtra = nodeCount > 15 ? 8 : 12

        interface SimNode {
          id: string; label: string; type: PositionedNode['type']; depth: number; domain?: string | null
          x?: number; y?: number; vx?: number; vy?: number; fx?: number | null; fy?: number | null
        }
        const simNodes: SimNode[] = nodeArr.map((n) => ({ ...n }))
        const simEdges = rawEdges.map((e) => ({ source: e.source, target: e.target }))
        for (const n of simNodes) {
          if (n.type === 'focal' && n.depth === 0) { n.fx = cx; n.fy = cy }
        }

        // Compute domain cluster centroids for clustering force
        const domainNodes = new Map<string, SimNode[]>()
        for (const n of simNodes) {
          const d = n.domain || '__none__'
          if (!domainNodes.has(d)) domainNodes.set(d, [])
          domainNodes.get(d)!.push(n)
        }
        const hasClusters = domainNodes.size > 1

        // Custom clustering force — pull same-domain nodes toward their centroid
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const clusterForce: any = hasClusters ? Object.assign(
          (alpha: number) => {
            for (const [, nodes] of domainNodes) {
              if (nodes.length < 2) continue
              let mx = 0, my = 0
              for (const n of nodes) { mx += (n.x ?? 0); my += (n.y ?? 0) }
              mx /= nodes.length; my /= nodes.length
              const strength = 0.3 * alpha
              for (const n of nodes) {
                if (n.fx != null) continue
                n.vx = (n.vx ?? 0) + (mx - (n.x ?? 0)) * strength
                n.vy = (n.vy ?? 0) + (my - (n.y ?? 0)) * strength
              }
            }
          },
          { initialize: () => {} }
        ) : null

        const sim = forceSimulation<SimNode>(simNodes)
          .force('link', forceLink<SimNode, { source: string; target: string }>(simEdges).id((d) => d.id).distance(linkDist))
          .force('charge', forceManyBody().strength(chargeStr))
          .force('x', forceX<SimNode>(cx).strength(0.05))
          .force('y', forceY<SimNode>(cy).strength(0.05))
          .force('collide', forceCollide<SimNode>().radius((d) => {
            const baseR = NODE_STYLES[d.type]?.r || 16
            return Math.max(10, baseR - (d.depth * 2)) + collideExtra
          }))
        if (clusterForce) sim.force('cluster', clusterForce)
        sim.stop()

        // Restore pinned node positions from previous layout
        const pinned = pinnedNodesRef.current
        if (pinned.size > 0) {
          for (const n of simNodes) {
            const p = pinned.get(n.id)
            if (p) { n.x = p.fx; n.y = p.fy; n.fx = p.fx; n.fy = p.fy }
          }
        }

        sim.tick() // single tick for initial positions

        // Store simulation and trigger animation
        simRef.current = sim as any
        shouldAnimateRef.current = true

        posNodes = simNodes.map((n) => ({
          id: n.id, label: n.label, type: n.type, depth: n.depth, domain: n.domain,
          x: n.x ?? cx, y: n.y ?? cy,
        }))
      } else {
        // Tree layout (vertical or horizontal)
        const isHoriz = layout === 'tree-h'
        // Build adjacency from edges (source→target)
        const children = new Map<string, string[]>()
        const hasParent = new Set<string>()
        for (const e of rawEdges) {
          if (!children.has(e.source)) children.set(e.source, [])
          children.get(e.source)!.push(e.target)
          hasParent.add(e.target)
        }
        // Find roots (no incoming edges)
        const roots = nodeArr.filter(n => !hasParent.has(n.id))
        if (roots.length === 0 && nodeArr.length > 0) roots.push(nodeArr[0])

        // BFS assign levels + horizontal position within level
        const levelMap = new Map<string, number>()
        const levelNodes = new Map<number, string[]>()
        const queue: string[] = roots.map(r => r.id)
        for (const r of roots) levelMap.set(r.id, 0)
        let qi = 0
        while (qi < queue.length) {
          const nid = queue[qi++]
          const lvl = levelMap.get(nid)!
          if (!levelNodes.has(lvl)) levelNodes.set(lvl, [])
          levelNodes.get(lvl)!.push(nid)
          for (const child of children.get(nid) || []) {
            if (!levelMap.has(child)) {
              levelMap.set(child, lvl + 1)
              queue.push(child)
            }
          }
        }
        // Place unvisited nodes
        for (const n of nodeArr) {
          if (!levelMap.has(n.id)) {
            const lvl = (levelMap.get(roots[0]?.id) ?? 0) + 1
            levelMap.set(n.id, lvl)
            if (!levelNodes.has(lvl)) levelNodes.set(lvl, [])
            levelNodes.get(lvl)!.push(n.id)
          }
        }

        const spacing = isHoriz ? { level: nodeSpacing * 1.6, node: nodeSpacing * 0.6 } : { level: nodeSpacing * 0.8, node: nodeSpacing * 1.4 }
        const startX = 80
        const startY = 60

        posNodes = nodeArr.map((n) => {
          const lvl = levelMap.get(n.id) ?? 0
          const siblings = levelNodes.get(lvl) || [n.id]
          const idx = siblings.indexOf(n.id)
          const x = isHoriz ? startX + lvl * spacing.level : startX + idx * spacing.node + ((gw - siblings.length * spacing.node) / 2)
          const y = isHoriz ? startY + idx * spacing.node + ((gh - siblings.length * spacing.node) / 2) : startY + lvl * spacing.level
          return { id: n.id, label: n.label, type: n.type, depth: n.depth, domain: n.domain, x, y }
        })
      }

      const posEdges: PositionedEdge[] = rawEdges.map((e) => ({
        sourceId: e.source,
        targetId: e.target,
        label: e.label,
        type: e.type,
      }))

      if (!cancelled) {
        setGraph({ nodes: posNodes, edges: posEdges })
        setLoading(false)
      }
    }

    fetchNeighborhood().catch(() => {
      if (!cancelled) setLoading(false)
    })

    return () => { cancelled = true; cancelAnimationFrame(animFrameRef.current) }
  }, [sessionId, termName, depth, fullscreen, layout, nodeSpacing, showLeaves, showClusters, graphDomainFilter])

  // Fit-to-window after force animation completes (or immediately for tree layouts)
  const needsFitRef = useRef(false)
  const graphIdRef = useRef(0)
  const lastFitId = useRef(0)

  // For tree layouts: fit immediately when graph appears
  useEffect(() => {
    if (layout === 'force') return
    if (!graph || graph.nodes.length === 0 || !containerRef.current) return
    if (lastFitId.current === graphIdRef.current) return
    lastFitId.current = graphIdRef.current
    const padding = 60
    const xs = graph.nodes.map(n => n.x)
    const ys = graph.nodes.map(n => n.y)
    const minX = Math.min(...xs) - padding, maxX = Math.max(...xs) + padding
    const minY = Math.min(...ys) - padding, maxY = Math.max(...ys) + padding
    const cw = containerRef.current.clientWidth
    const ch = containerRef.current.clientHeight
    const newZoom = Math.min(cw / (maxX - minX), ch / (maxY - minY), 2)
    const gcx = (minX + maxX) / 2, gcy = (minY + maxY) / 2
    setZoom(newZoom)
    setPan({ x: (cw / 2) - gcx * newZoom, y: (ch / 2) - gcy * newZoom })
  }, [graph, layout])

  // Extract current positions from simulation into graph state
  const syncSimToGraph = useCallback(() => {
    const sim = simRef.current
    if (!sim) return
    const nodes = (sim as any).nodes() as Array<{ id: string; x: number; y: number; label: string; type: PositionedNode['type']; depth: number; domain?: string | null }>
    setGraph(prev => prev ? {
      ...prev,
      nodes: nodes.map(n => ({ id: n.id, label: n.label, type: n.type, depth: n.depth, domain: n.domain, x: n.x, y: n.y })),
    } : prev)
  }, [])

  // Auto-animate force layout whenever a new graph is created
  useEffect(() => {
    if (!shouldAnimateRef.current || !simRef.current || !graph) return
    shouldAnimateRef.current = false
    const sim = simRef.current
    const doFit = needsFitRef.current
    needsFitRef.current = false
    sim.alpha(1).restart()
    const tick = () => {
      if (sim.alpha() < 0.001) {
        sim.stop()
        syncSimToGraph()
        // Fit-to-window after animation converges
        if (doFit && containerRef.current) {
          const simNodes = (sim as any).nodes() as Array<{ x: number; y: number }>
          const padding = 60
          const xs = simNodes.map(n => n.x), ys = simNodes.map(n => n.y)
          const minX = Math.min(...xs) - padding, maxX = Math.max(...xs) + padding
          const minY = Math.min(...ys) - padding, maxY = Math.max(...ys) + padding
          const cw = containerRef.current.clientWidth
          const ch = containerRef.current.clientHeight
          const newZoom = Math.min(cw / (maxX - minX), ch / (maxY - minY), 2)
          const gcx = (minX + maxX) / 2, gcy = (minY + maxY) / 2
          setZoom(newZoom)
          setPan({ x: (cw / 2) - gcx * newZoom, y: (ch / 2) - gcy * newZoom })
        }
        return
      }
      syncSimToGraph()
      animFrameRef.current = requestAnimationFrame(tick)
    }
    animFrameRef.current = requestAnimationFrame(tick)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph])

  // Refine: animate N more iterations from current positions
  const refineLayout = useCallback(() => {
    const sim = simRef.current
    if (!sim) return
    sim.alpha(0.3).restart()
    const tick = () => {
      if (sim.alpha() < 0.001) {
        sim.stop()
        syncSimToGraph()
        return
      }
      syncSimToGraph()
      animFrameRef.current = requestAnimationFrame(tick)
    }
    animFrameRef.current = requestAnimationFrame(tick)
  }, [syncSimToGraph])

  // Cleanup animation on unmount
  useEffect(() => () => cancelAnimationFrame(animFrameRef.current), [])

  // Node drag state
  const [dragNode, setDragNode] = useState<string | null>(null)
  const dragNodeStart = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const dragNodeMoved = useRef(false)

  const handleNodeDragStart = useCallback((nodeId: string, e: React.PointerEvent) => {
    e.stopPropagation()
    ;(e.target as Element).setPointerCapture(e.pointerId)
    setDragNode(nodeId)
    dragNodeStart.current = { x: e.clientX, y: e.clientY }
    dragNodeMoved.current = false
  }, [])

  const handleNodeDragMove = useCallback((e: React.PointerEvent) => {
    if (!dragNode || !graph) return
    e.stopPropagation()
    const dx = (e.clientX - dragNodeStart.current.x) / zoom
    const dy = (e.clientY - dragNodeStart.current.y) / zoom
    if (Math.abs(dx) > 1 || Math.abs(dy) > 1) dragNodeMoved.current = true
    dragNodeStart.current = { x: e.clientX, y: e.clientY }
    setGraph(prev => {
      if (!prev) return prev
      return {
        ...prev,
        nodes: prev.nodes.map(n => n.id === dragNode ? { ...n, x: n.x + dx, y: n.y + dy } : n),
      }
    })
    // Update simulation node position and fix it
    const sim = simRef.current
    if (sim) {
      const simNodes = (sim as any).nodes() as Array<{ id: string; x: number; y: number; fx: number | null; fy: number | null }>
      const sn = simNodes.find(n => n.id === dragNode)
      if (sn) {
        sn.x += dx; sn.y += dy; sn.fx = sn.x; sn.fy = sn.y
        // Persist pin so it survives graph re-renders
        pinnedNodesRef.current.set(dragNode, { fx: sn.x, fy: sn.y })
      }
    }
  }, [dragNode, graph, zoom])

  const handleNodeDragEnd = useCallback(() => {
    setDragNode(null)
  }, [])

  const handleNodeClick = useCallback((name: string) => {
    if (dragNodeMoved.current) return
    navigateToTerm(name)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const fitToWindow = useCallback(() => {
    if (!graph || graph.nodes.length === 0 || !containerRef.current) return
    const padding = 60
    const xs = graph.nodes.map(n => n.x)
    const ys = graph.nodes.map(n => n.y)
    const minX = Math.min(...xs) - padding, maxX = Math.max(...xs) + padding
    const minY = Math.min(...ys) - padding, maxY = Math.max(...ys) + padding
    const cw = containerRef.current.clientWidth
    const ch = containerRef.current.clientHeight
    const newZoom = Math.min(cw / (maxX - minX), ch / (maxY - minY), 2)
    const gcx = (minX + maxX) / 2, gcy = (minY + maxY) / 2
    setZoom(newZoom)
    setPan({ x: (cw / 2) - gcx * newZoom, y: (ch / 2) - gcy * newZoom })
  }, [graph])

  const centerGraph = useCallback(() => {
    if (!graph || graph.nodes.length === 0 || !containerRef.current) return
    const xs = graph.nodes.map(n => n.x)
    const ys = graph.nodes.map(n => n.y)
    const gcx = (Math.min(...xs) + Math.max(...xs)) / 2
    const gcy = (Math.min(...ys) + Math.max(...ys)) / 2
    const cw = containerRef.current.clientWidth
    const ch = containerRef.current.clientHeight
    setPan({ x: (cw / 2) - gcx * zoom, y: (ch / 2) - gcy * zoom })
  }, [graph, zoom])

  const exportGraph = useCallback((format: 'svg' | 'png' | 'jpeg') => {
    const svg = svgRef.current
    if (!svg) return
    setShowExportMenu(false)
    const svgData = new XMLSerializer().serializeToString(svg)
    if (format === 'svg') {
      const blob = new Blob([svgData], { type: 'image/svg+xml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a'); a.href = url; a.download = `graph-${termName}.svg`; a.click()
      URL.revokeObjectURL(url)
      return
    }
    // Rasterize to canvas for PNG/JPEG
    const canvas = document.createElement('canvas')
    const rect = svg.getBoundingClientRect()
    const scale = 2 // 2x for retina
    canvas.width = rect.width * scale; canvas.height = rect.height * scale
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(scale, scale)
    const img = new Image()
    const blob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    img.onload = () => {
      if (format === 'jpeg') { ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, canvas.width, canvas.height) }
      ctx.drawImage(img, 0, 0, rect.width, rect.height)
      URL.revokeObjectURL(url)
      const mime = format === 'png' ? 'image/png' : 'image/jpeg'
      canvas.toBlob(b => {
        if (!b) return
        const dl = URL.createObjectURL(b)
        const a = document.createElement('a'); a.href = dl; a.download = `graph-${termName}.${format}`; a.click()
        URL.revokeObjectURL(dl)
      }, mime, 0.95)
    }
    img.src = url
  }, [termName])

  // Close export menu on outside click
  useEffect(() => {
    if (!showExportMenu) return
    const close = () => setShowExportMenu(false)
    const timer = setTimeout(() => document.addEventListener('click', close), 0)
    return () => { clearTimeout(timer); document.removeEventListener('click', close) }
  }, [showExportMenu])

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? 0.95 : 1.05
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    setZoom(z => {
      const nz = Math.max(0.1, Math.min(5, z * delta))
      setPan(p => ({
        x: mx - (mx - p.x) * (nz / z),
        y: my - (my - p.y) * (nz / z),
      }))
      return nz
    })
  }, [])

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest('circle')) return
    setDragging(true)
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y })
  }, [pan])

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (dragNode) { handleNodeDragMove(e); return }
    if (!dragging) return
    setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y })
  }, [dragging, dragStart, dragNode, handleNodeDragMove])

  const handlePointerUp = useCallback(() => { setDragging(false); if (dragNode) handleNodeDragEnd() }, [dragNode, handleNodeDragEnd])

  // Build a position lookup for edge rendering
  const posMap = useMemo(() => {
    if (!graph) return new Map<string, PositionedNode>()
    const m = new Map<string, PositionedNode>()
    for (const n of graph.nodes) m.set(n.id, n)
    return m
  }, [graph])

  // Compute cluster hulls per domain
  const clusterHulls = useMemo(() => {
    if (!graph) return []
    const byDomain = new Map<string, PositionedNode[]>()
    for (const n of graph.nodes) {
      const d = n.domain || '__none__'
      if (!byDomain.has(d)) byDomain.set(d, [])
      byDomain.get(d)!.push(n)
    }
    const hulls: { domain: string; path: string; fill: string; stroke: string; cx: number; cy: number }[] = []

    // Domain hulls — when enabled
    if (showDomains) {
      const domains = Array.from(byDomain.keys()).sort()
      for (let i = 0; i < domains.length; i++) {
        const domain = domains[i]
        const nodes = byDomain.get(domain)!
        if (nodes.length < 2) continue
        let pathD: string
        if (nodes.length === 2) {
          const centX = (nodes[0].x + nodes[1].x) / 2
          const centY = (nodes[0].y + nodes[1].y) / 2
          const ddx = nodes[1].x - nodes[0].x
          const ddy = nodes[1].y - nodes[0].y
          const halfDist = Math.sqrt(ddx * ddx + ddy * ddy) / 2
          const rx = halfDist + 30
          const ry = 30
          const angle = Math.atan2(ddy, ddx) * (180 / Math.PI)
          pathD = `M ${centX - rx},${centY} A ${rx},${ry} ${angle} 1,0 ${centX + rx},${centY} A ${rx},${ry} ${angle} 1,0 ${centX - rx},${centY} Z`
        } else {
          const points = nodes.map(n => ({ x: n.x, y: n.y }))
          const hull = convexHull(points)
          const expanded = expandHull(hull, 30)
          pathD = expanded.length > 0
            ? `M ${expanded.map(p => `${p.x},${p.y}`).join(' L ')} Z`
            : ''
        }
        hulls.push({
          domain: domain === '__none__' ? 'System' : domain,
          path: pathD,
          fill: CLUSTER_COLORS[i % CLUSTER_COLORS.length],
          stroke: CLUSTER_STROKES[i % CLUSTER_STROKES.length],
          cx: nodes.reduce((s, n) => s + n.x, 0) / nodes.length,
          cy: Math.min(...nodes.map(n => n.y)) - 20,
        })
      }
    }

    // Sibling cluster hull: focal node + all sibling nodes
    const focalNode = graph.nodes.find(n => n.type === 'focal')
    const siblingNodes = graph.nodes.filter(n => n.type === 'sibling')
    if (showClusters && focalNode && siblingNodes.length > 0) {
      const hullNodes = [focalNode, ...siblingNodes]
      const centX = hullNodes.reduce((s, n) => s + n.x, 0) / hullNodes.length
      const centY = hullNodes.reduce((s, n) => s + n.y, 0) / hullNodes.length
      let pathD: string
      if (hullNodes.length === 2) {
        // 2 points: draw an ellipse around them
        const dx = hullNodes[1].x - hullNodes[0].x
        const dy = hullNodes[1].y - hullNodes[0].y
        const halfDist = Math.sqrt(dx * dx + dy * dy) / 2
        const rx = halfDist + 30
        const ry = 30
        const angle = Math.atan2(dy, dx) * (180 / Math.PI)
        pathD = `M ${centX - rx},${centY} A ${rx},${ry} ${angle} 1,0 ${centX + rx},${centY} A ${rx},${ry} ${angle} 1,0 ${centX - rx},${centY} Z`
      } else {
        const points = hullNodes.map(n => ({ x: n.x, y: n.y }))
        const hull = convexHull(points)
        const expanded = expandHull(hull, 30)
        pathD = expanded.length > 0
          ? `M ${expanded.map(p => `${p.x},${p.y}`).join(' L ')} Z`
          : ''
      }
      if (pathD) {
        hulls.push({
          domain: 'cluster',
          path: pathD,
          fill: 'rgba(245,158,11,0.06)',
          stroke: 'rgba(245,158,11,0.2)',
          cx: centX,
          cy: Math.min(...hullNodes.map(n => n.y)) - 20,
        })
      }
    }

    return hulls
  }, [graph, showDomains, showClusters])

  const svgContent = graph && !empty ? (
    <g transform={`translate(${pan.x},${pan.y}) scale(${zoom})`}>
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#9ca3af" />
        </marker>
        <marker id="arrowhead-parent" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#a855f7" />
        </marker>
        <marker id="arrowhead-child" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#22c55e" />
        </marker>
      </defs>
      {/* Cluster hulls */}
      {clusterHulls.map((c, i) => (
        <g key={`cluster-${i}`}>
          <path d={c.path} fill={c.fill} stroke={c.stroke} strokeWidth={1.5} strokeDasharray="6 3" />
          <text
            x={c.cx} y={c.cy}
            textAnchor="middle"
            className="text-[9px] fill-gray-400 dark:fill-gray-500 pointer-events-none font-medium"
          >
            {c.domain}
          </text>
        </g>
      ))}
      {/* Edges */}
      {(() => {
        // Count parallel edges between each node pair so we can curve them
        const pairCount = new Map<string, number>()
        const pairIndex = new Map<string, number>()
        for (const e of graph.edges) {
          const key = [e.sourceId, e.targetId].sort().join('|')
          pairCount.set(key, (pairCount.get(key) || 0) + 1)
        }
        return graph.edges.map((e, i) => {
          const src = posMap.get(e.sourceId)
          const tgt = posMap.get(e.targetId)
          if (!src || !tgt) return null

          const pairKey = [e.sourceId, e.targetId].sort().join('|')
          const total = pairCount.get(pairKey) || 1
          const idx = pairIndex.get(pairKey) || 0
          pairIndex.set(pairKey, idx + 1)

          const srcR = Math.max(10, (NODE_STYLES[src.type]?.r || 16) - (src.depth * 2))
          const tgtR = Math.max(10, (NODE_STYLES[tgt.type]?.r || 16) - (tgt.depth * 2))
          const dx = tgt.x - src.x
          const dy = tgt.y - src.y
          const dist = Math.sqrt(dx * dx + dy * dy) || 1

          // Perpendicular offset for parallel edges
          const curvature = total > 1 ? (idx - (total - 1) / 2) * 30 : 0
          const nx = -dy / dist  // perpendicular unit vector
          const ny = dx / dist

          const startX = src.x + (dx / dist) * srcR
          const startY = src.y + (dy / dist) * srcR
          const endX = tgt.x - (dx / dist) * tgtR
          const endY = tgt.y - (dy / dist) * tgtR
          // Control point for quadratic bezier
          const cpX = (src.x + tgt.x) / 2 + nx * curvature
          const cpY = (src.y + tgt.y) / 2 + ny * curvature
          // Label position: point on the bezier at t=0.5
          const labelX = 0.25 * startX + 0.5 * cpX + 0.25 * endX
          const labelY = 0.25 * startY + 0.5 * cpY + 0.25 * endY

          const strokeColor = e.type === 'parent' ? '#a855f7' : e.type === 'child' ? '#22c55e' : e.type === 'sibling' ? '#f59e0b' : '#9ca3af'
          const isHierarchy = e.type === 'parent' || e.type === 'child'
          const dashArray = e.type === 'sibling' ? '2 3' : undefined
          const marker = isHierarchy
            ? `url(#arrowhead-${e.type})`
            : e.type === 'relationship' ? 'url(#arrowhead)' : undefined

          return (
            <g key={i}>
              {curvature === 0 ? (
                <line
                  x1={startX} y1={startY} x2={endX} y2={endY}
                  stroke={strokeColor} strokeWidth={1.5} strokeDasharray={dashArray}
                  markerEnd={marker}
                />
              ) : (
                <path
                  d={`M ${startX},${startY} Q ${cpX},${cpY} ${endX},${endY}`}
                  fill="none" stroke={strokeColor} strokeWidth={1.5} strokeDasharray={dashArray}
                  markerEnd={marker}
                />
              )}
              {e.type !== 'sibling' && (
              <text
                x={labelX} y={labelY - 5}
                textAnchor="middle"
                className="text-[8px] fill-gray-400 dark:fill-gray-500 pointer-events-none"
              >
                {e.label}
              </text>
              )}
            </g>
          )
        })
      })()}
      {/* Nodes */}
      {graph.nodes.map((n) => {
        const style = NODE_STYLES[n.type]
        const r = Math.max(10, style.r - (n.depth * 2))
        const isPinned = n.type !== 'focal' && pinnedNodesRef.current.has(n.id)
        return (
          <g
            key={n.id}
            transform={`translate(${n.x},${n.y})`}
            className="cursor-grab active:cursor-grabbing"
            onClick={() => handleNodeClick(n.id)}
            onPointerDown={(e) => handleNodeDragStart(n.id, e)}
            onPointerMove={dragNode === n.id ? handleNodeDragMove : undefined}
            onPointerUp={dragNode === n.id ? handleNodeDragEnd : undefined}
            onMouseEnter={(e) => {
              if (hoverTimeout.current) clearTimeout(hoverTimeout.current)
              setHoverNode({ name: n.id, x: e.clientX + 12, y: e.clientY - 10 })
            }}
            onMouseLeave={() => {
              hoverTimeout.current = setTimeout(() => setHoverNode(null), 150)
            }}
          >
            <circle r={r} fill={style.fill} className="hover:stroke-amber-400 hover:stroke-2" stroke={isPinned ? '#ef4444' : undefined} strokeWidth={isPinned ? 2 : undefined} />
            <text
              y={r + 12}
              textAnchor="middle"
              className="text-[10px] fill-gray-700 dark:fill-gray-300 pointer-events-none"
            >
              {n.label.length > 18 ? n.label.slice(0, 16) + '...' : n.label}
            </text>
            {n.type === 'focal' && (
              <text
                textAnchor="middle"
                dominantBaseline="central"
                className="text-[10px] fill-white font-semibold pointer-events-none"
              >
                {n.label.length > 10 ? n.label.slice(0, 8) + '..' : n.label}
              </text>
            )}
          </g>
        )
      })}
    </g>
  ) : null

  const graphContent = (
    <>
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <label className="text-[10px] text-gray-500 dark:text-gray-400 select-none">Depth</label>
        <input
          type="range"
          min={1}
          max={10}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          disabled={loading}
          className="w-20 h-1 accent-blue-500"
        />
        <span className="text-[10px] text-gray-500 dark:text-gray-400 w-3 text-center">{depth}</span>
        <select
          value={layout}
          onChange={(e) => setLayout(e.target.value as typeof layout)}
          className="text-[10px] py-0.5 px-1 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300"
        >
          <option value="force">Force</option>
          <option value="tree-v">Tree ↓</option>
          <option value="tree-h">Tree →</option>
        </select>
        {layout === 'force' && (
          <>
            <select
              value={forceIterations}
              onChange={(e) => setForceIterations(Number(e.target.value))}
              className="text-[10px] py-0.5 px-1 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300"
              title="Force iterations"
            >
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={400}>400</option>
              <option value={800}>800</option>
            </select>
            <button
              onClick={refineLayout}
              disabled={!simRef.current}
              className="text-[10px] px-1 py-0.5 rounded border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 disabled:opacity-30"
              title="Refine layout (run more iterations)"
            >
              ⟳
            </button>
          </>
        )}
        <button
          onClick={centerGraph}
          className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          title="Center graph"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
          </svg>
        </button>
        <button
          onClick={fitToWindow}
          className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          title="Fit to window"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M4 14h6v6M20 10h-6V4M14 10l7-7M3 21l7-7" />
          </svg>
        </button>
        <button
          onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }) }}
          className="text-[10px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
          title="Reset zoom"
        >
          1:1
        </button>
        <span className="text-[10px] text-gray-400">{Math.round(zoom * 100)}%</span>
        <span className="text-[10px] text-gray-500 dark:text-gray-400 ml-1 select-none">Spacing</span>
        <input
          type="range"
          min={40}
          max={250}
          value={nodeSpacing}
          onChange={(e) => setNodeSpacing(Number(e.target.value))}
          className="w-16 h-1 accent-blue-500"
        />
        <label className="flex items-center gap-0.5 text-[10px] text-gray-500 dark:text-gray-400 select-none ml-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showLeaves}
            onChange={(e) => setShowLeaves(e.target.checked)}
            className="w-3 h-3 accent-blue-500"
          />
          Leaves
        </label>
        <label className="flex items-center gap-0.5 text-[10px] text-gray-500 dark:text-gray-400 select-none ml-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showClusters}
            onChange={(e) => setShowClusters(e.target.checked)}
            className="w-3 h-3 accent-amber-500"
          />
          Clusters
        </label>
        <label className="flex items-center gap-0.5 text-[10px] text-gray-500 dark:text-gray-400 select-none ml-1 cursor-pointer">
          <input
            type="checkbox"
            checked={showDomains}
            onChange={(e) => setShowDomains(e.target.checked)}
            className="w-3 h-3 accent-purple-500"
          />
          Domains
        </label>
        {graphDomains.length > 1 && (
          <GraphDomainFilter
            domains={graphDomains}
            selected={graphDomainFilter}
            onChange={setGraphDomainFilter}
          />
        )}
        <div className="ml-auto flex items-center gap-1">
          <div className="relative">
            <button
              onClick={() => setShowExportMenu(v => !v)}
              className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              title="Export graph"
            >
              <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <path d="M12 5v14M19 12l-7 7-7-7" />
              </svg>
            </button>
            {showExportMenu && (
              <div className="absolute right-0 top-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-50 py-1 min-w-[80px]">
                {(['svg', 'png', 'jpeg'] as const).map(fmt => (
                  <button key={fmt} onClick={() => exportGraph(fmt)} className="block w-full text-left px-3 py-1 text-[11px] text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                    {fmt.toUpperCase()}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {fullscreen
              ? <ArrowsPointingInIcon className="w-3.5 h-3.5" />
              : <ArrowsPointingOutIcon className="w-3.5 h-3.5" />
            }
          </button>
        </div>
      </div>
      {loading ? (
        <div className="text-xs text-gray-400 py-4 text-center">Loading graph...</div>
      ) : empty || !graph ? (
        <div
          ref={containerRef}
          className={`overflow-hidden border border-gray-200 dark:border-gray-700 rounded ${fullscreen ? 'flex-1 min-h-0' : ''}`}
          style={fullscreen ? undefined : { height: `${graphHeight}px` }}
        >
          <svg width="100%" height="100%" viewBox="0 0 200 200" className="select-none">
            <circle cx="100" cy="100" r={24} fill="#3b82f6" />
            <text
              x="100" y="100"
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[10px] fill-white font-semibold pointer-events-none"
            >
              {termName.length > 10 ? termName.slice(0, 8) + '..' : termName}
            </text>
          </svg>
        </div>
      ) : (
        <>
          <div
            ref={containerRef}
            className={`overflow-hidden border border-gray-200 dark:border-gray-700 rounded cursor-grab active:cursor-grabbing ${fullscreen ? 'flex-1 min-h-0' : ''}`}
            style={fullscreen ? undefined : { height: `${graphHeight}px` }}
            onWheel={handleWheel}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerUp}
          >
            <svg ref={svgRef} width="100%" height="100%" className="select-none">
              {svgContent}
            </svg>
          </div>
          {/* Resize handle */}
          {!fullscreen && (
            <div
              className="h-1.5 cursor-ns-resize flex items-center justify-center hover:bg-gray-200 dark:hover:bg-gray-700 rounded-b"
              onPointerDown={(e) => {
                e.preventDefault()
                const startY = e.clientY
                const startH = graphHeight
                const onMove = (ev: PointerEvent) => {
                  const newH = Math.max(150, Math.min(800, startH + (ev.clientY - startY)))
                  setGraphHeight(newH)
                }
                const onUp = () => { document.removeEventListener('pointermove', onMove); document.removeEventListener('pointerup', onUp) }
                document.addEventListener('pointermove', onMove)
                document.addEventListener('pointerup', onUp)
              }}
            >
              <div className="w-8 h-0.5 bg-gray-300 dark:bg-gray-600 rounded" />
            </div>
          )}
          {hoverNode && (
            <div
              className="fixed z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 pointer-events-none"
              style={{ left: hoverNode.x, top: hoverNode.y, maxHeight: '260px', overflow: 'hidden' }}
            >
              <NodeTooltip sessionId={sessionId} termName={hoverNode.name} />
            </div>
          )}
          {/* Legend */}
          <div className="flex items-center gap-3 text-[10px] text-gray-400 mt-1">
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-blue-500" /> focal</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-purple-400" /> parent</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-green-400" /> child</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-gray-400" /> relationship</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-amber-500" /> sibling</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full border-2 border-red-500 bg-transparent" /> pinned</span>
          </div>
        </>
      )}
    </>
  )

  if (fullscreen) {
    return (
      <>
        <div className="mt-1" />
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={() => setFullscreen(false)}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-4 overflow-auto flex flex-col"
            style={{ width: '80vw', height: '80vh', resize: 'both', minWidth: '400px', minHeight: '300px' }}
            onClick={(e) => e.stopPropagation()}
          >
            {graphContent}
          </div>
        </div>
      </>
    )
  }

  return (
    <div className="mt-1">
      {graphContent}
    </div>
  )
}

// Modal to move a data source between domains
function SourceMovePicker({
  sourceName,
  sourceType,
  currentDomain,
  sessionId,
  onMoved,
}: {
  sourceName: string
  sourceType: 'databases' | 'apis' | 'documents'
  currentDomain: string
  sessionId: string
  onMoved: () => void
}) {
  const [open, setOpen] = useState(false)
  const [treeNodes, setTreeNodes] = useState<DomainTreeNode[]>([])
  const [loading, setLoading] = useState(false)

  const handleOpen = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setOpen(true)
    setLoading(true)
    try {
      const tree = await getDomainTree()
      setTreeNodes(tree)
    } catch {
      const { domains: allDomains } = await listDomains()
      setTreeNodes(allDomains.map(d => ({ ...d, path: '', databases: [], apis: [], documents: [], children: [] })))
    }
    setLoading(false)
  }

  const executeMove = async (targetDomain: string) => {
    await moveDomainSource({
      source_type: sourceType,
      source_name: sourceName,
      from_domain: currentDomain,
      to_domain: targetDomain,
      session_id: sessionId,
    })
    setOpen(false)
    onMoved()
  }

  const typeLabel = sourceType === 'databases' ? 'database' : sourceType === 'apis' ? 'API' : 'document'

  return (
    <span className="flex-shrink-0" onClick={(e) => e.stopPropagation()}>
      <span
        role="button"
        onClick={handleOpen}
        className="p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 inline-flex text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
        title={`Move ${typeLabel} to another domain`}
      >
        <ArrowUpIcon className="w-3 h-3" />
      </span>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm w-full mx-4 p-4 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Move {typeLabel} "{sourceName}"
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Currently in: <span className="font-medium">{currentDomain}</span>
            </div>
            {loading ? (
              <div className="text-xs text-gray-400 py-2 text-center">Loading domains...</div>
            ) : (
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {currentDomain !== 'system' && (
                  <button
                    onClick={() => executeMove('system')}
                    className="w-full text-left text-xs px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-700 mb-1 pb-2"
                  >
                    <div className="font-medium">Unassign (move to system)</div>
                  </button>
                )}
                {(() => {
                  const renderNode = (node: DomainTreeNode, depth: number = 0): React.ReactNode => (
                    <div key={node.filename}>
                      {node.filename !== currentDomain && node.filename !== 'system' && (
                        <button
                          onClick={() => executeMove(node.filename)}
                          className="w-full text-left text-xs px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                          style={{ paddingLeft: `${depth * 12 + 12}px` }}
                        >
                          <div className="font-medium">{node.name}</div>
                          <div className="text-gray-400">{node.path || node.filename}</div>
                        </button>
                      )}
                      {node.children?.map(child => renderNode(child, depth + 1))}
                    </div>
                  )
                  return treeNodes.map(n => renderNode(n))
                })()}
                {treeNodes.length === 0 && (
                  <div className="text-xs text-gray-400 py-2 text-center">No domains available</div>
                )}
              </div>
            )}
            <div className="flex justify-end pt-1">
              <button
                onClick={() => setOpen(false)}
                className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </span>
  )
}

// Domain filter checkbox tree
function DomainFilterTree({
  tree,
  selected,
  onChange,
  open,
  onToggle,
  sessionId,
  onTreeRefresh,
}: {
  tree: DomainTreeNode[]
  selected: string[]
  onChange: (selected: string[]) => void
  open: boolean
  onToggle: () => void
  sessionId: string
  onTreeRefresh?: () => void
}) {
  // Collect all filenames under a node (inclusive)
  const collectFilenames = useCallback((node: DomainTreeNode): string[] => {
    const result = [node.filename]
    for (const child of node.children || []) {
      result.push(...collectFilenames(child))
    }
    return result
  }, [])

  const allFilenames = useMemo(() => {
    const result: string[] = []
    for (const node of tree) result.push(...collectFilenames(node))
    return result
  }, [tree, collectFilenames])

  const toggleNode = useCallback((node: DomainTreeNode) => {
    const nodeFiles = collectFilenames(node)
    const allSelected = nodeFiles.every(f => selected.includes(f))
    if (allSelected) {
      // Deselect this node + children
      onChange(selected.filter(f => !nodeFiles.includes(f)))
    } else {
      // Select this node + children
      onChange([...new Set([...selected, ...nodeFiles])])
    }
  }, [selected, onChange, collectFilenames])

  const clearAll = useCallback(() => onChange([]), [onChange])

  const selectAll = useCallback(() => {
    onChange([...allFilenames])
  }, [onChange, allFilenames])

  const label = selected.length === 0
    ? 'All domains'
    : selected.length === 1
      ? (selected[0] === 'system' ? 'System (root)' : tree.find(n => n.filename === selected[0])?.name || selected[0])
      : `${selected.length} domains`

  const sourcesForNode = (node: DomainTreeNode): { name: string; type: 'databases' | 'apis' | 'documents'; label: string }[] => {
    const items: { name: string; type: 'databases' | 'apis' | 'documents'; label: string }[] = []
    for (const db of node.databases || []) items.push({ name: db, type: 'databases', label: 'db' })
    for (const api of node.apis || []) items.push({ name: api, type: 'apis', label: 'api' })
    for (const doc of node.documents || []) items.push({ name: doc, type: 'documents', label: 'doc' })
    return items
  }

  const renderNode = (node: DomainTreeNode, depth: number = 0): React.ReactNode => {
    const nodeFiles = collectFilenames(node)
    const allChecked = nodeFiles.every(f => selected.includes(f))
    const someChecked = !allChecked && nodeFiles.some(f => selected.includes(f))
    const sources = sourcesForNode(node)
    return (
      <div key={node.filename}>
        <label
          className="flex items-center gap-1.5 py-0.5 px-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-xs text-gray-700 dark:text-gray-300"
          style={{ paddingLeft: `${depth * 14 + 4}px` }}
        >
          <input
            type="checkbox"
            checked={allChecked}
            ref={(el) => { if (el) el.indeterminate = someChecked }}
            onChange={() => toggleNode(node)}
            className="w-3 h-3 rounded border-gray-300 dark:border-gray-600"
          />
          <span className="truncate">{node.name}</span>
          {(node.children?.length ?? 0) > 0 && (
            <span className="text-gray-400 text-[10px]">+{nodeFiles.length - 1}</span>
          )}
        </label>
        {sources.length > 0 && sources.map(src => (
          <div
            key={`${node.filename}:${src.type}:${src.name}`}
            className="flex items-center gap-1 py-0.5 text-[10px] text-gray-400 dark:text-gray-500"
            style={{ paddingLeft: `${depth * 14 + 24}px` }}
          >
            <span className="bg-gray-100 dark:bg-gray-700 rounded px-1">{src.label}</span>
            <span className="truncate">{src.name}</span>
            <SourceMovePicker
              sourceName={src.name}
              sourceType={src.type}
              currentDomain={node.filename}
              sessionId={sessionId}
              onMoved={() => onTreeRefresh?.()}
            />
          </div>
        ))}
        {node.children?.map(child => renderNode(child, depth + 1))}
      </div>
    )
  }

  return (
    <div className="relative">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between text-xs py-1 px-2 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700"
      >
        <span className="truncate">{label}</span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${open ? 'rotate-90' : ''}`} />
      </button>
      {open && (
        <div className="mt-1 border border-gray-200 dark:border-gray-700 rounded bg-white dark:bg-gray-800 shadow-lg max-h-48 overflow-y-auto py-1">
          <div className="flex items-center gap-2 px-2 pb-1 border-b border-gray-100 dark:border-gray-700 mb-1">
            <button onClick={clearAll} className="text-[10px] text-blue-500 hover:underline">Clear</button>
            <button onClick={selectAll} className="text-[10px] text-blue-500 hover:underline">All</button>
          </div>
          {tree.map(node => renderNode(node))}
        </div>
      )}
    </div>
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
    generateGlossary,
    deleteDrafts,
    generating,
    generationStage,
    generationPercent,
    setFilter,
    setViewMode,
  } = useGlossaryStore()

  const [showConfirm, setShowConfirm] = useState(false)
  const [showDeleteDrafts, setShowDeleteDrafts] = useState(false)
  const [deletingDrafts, setDeletingDrafts] = useState(false)
  const [fullscreen, setFullscreen] = useState(false)

  const [search, setSearch] = useState('')

  const [domainTree, setDomainTree] = useState<DomainTreeNode[]>([])
  const [domainFilterOpen, setDomainFilterOpen] = useState(false)
  const [showIgnored, setShowIgnored] = useState(true)

  useEffect(() => {
    fetchTerms(sessionId)
    fetchDeprecated(sessionId)
  }, [sessionId, filters.scope, filters.domain, fetchTerms, fetchDeprecated])

  const refreshDomainTree = useCallback(() => {
    getDomainTree()
      .then(setDomainTree)
      .catch(() => listDomains().then(r => setDomainTree(
        r.domains.map(d => ({ ...d, path: '', databases: [], apis: [], documents: [], children: [] }))
      )).catch(() => {}))
  }, [])

  useEffect(() => {
    refreshDomainTree()
  }, [refreshDomainTree])

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

  // Filter by status and ignored
  const displayTerms = useMemo(() => {
    let result = filteredTerms
    if (!showIgnored) {
      result = result.filter((t) => !t.ignored)
    }
    if (filters.status) {
      result = result.filter((t) => t.status === filters.status)
    }
    return result
  }, [filteredTerms, filters.status, showIgnored])

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

  const content = (
    <div className={fullscreen ? 'space-y-2 p-4 h-full flex flex-col' : 'space-y-2'}>
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
        <button
          onClick={() => setFullscreen(!fullscreen)}
          className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700"
          title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
        >
          {fullscreen
            ? <ArrowsPointingInIcon className="w-3.5 h-3.5 text-gray-500" />
            : <ArrowsPointingOutIcon className="w-3.5 h-3.5 text-gray-500" />
          }
        </button>
        <div className="flex-1" />
        {!generating && (
          <button
            onClick={() => setShowDeleteDrafts(true)}
            disabled={deletingDrafts}
            className="flex items-center gap-1 text-xs px-1.5 py-0.5 rounded text-red-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20"
            title="Delete all draft terms"
          >
            <TrashIcon className="w-3.5 h-3.5" />
            Drafts
          </button>
        )}
        {generating ? (
          <div className="flex items-center gap-1.5 text-xs text-purple-500">
            <SparklesIcon className="w-3.5 h-3.5 animate-spin" />
            <span className="truncate max-w-[8rem]">{generationStage || 'Starting...'}</span>
            <div className="w-16 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div className="h-full bg-purple-500 rounded-full transition-all"
                   style={{ width: `${generationPercent}%` }} />
            </div>
            <span className="text-[10px] text-gray-400">{generationPercent}%</span>
          </div>
        ) : (
          <button
            onClick={() => setShowConfirm(true)}
            className="flex items-center gap-1 text-xs px-1.5 py-0.5 rounded text-purple-500 hover:text-purple-600 hover:bg-purple-50 dark:hover:bg-purple-900/20"
            title="Generate definitions, taxonomy, and relationships"
          >
            <SparklesIcon className="w-3.5 h-3.5" />
            Taxonomy
          </button>
        )}
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
        <button
          onClick={() => setShowIgnored(!showIgnored)}
          className={`ml-auto p-1 rounded ${showIgnored ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-600' : 'text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
          title={showIgnored ? 'Hide ignored terms' : 'Show ignored terms'}
        >
          <EyeSlashIcon className="w-3.5 h-3.5" />
        </button>
        {search && (
          <span className="text-xs text-gray-400 self-center">
            {displayTerms.length} results
          </span>
        )}
      </div>

      {/* Domain filter */}
      {domainTree.length > 0 && (
        <DomainFilterTree
          tree={domainTree}
          selected={filters.domain?.split(',').filter(Boolean) || []}
          onChange={(selected) => setFilter({ domain: selected.length > 0 ? selected.join(',') : undefined })}
          open={domainFilterOpen}
          onToggle={() => setDomainFilterOpen(!domainFilterOpen)}
          sessionId={sessionId}
          onTreeRefresh={refreshDomainTree}
        />
      )}

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
        <div className={`overflow-y-auto ${fullscreen ? 'flex-1' : 'max-h-[calc(100vh-20rem)]'}`}>
          {tree.roots.map((node) => (
            <TreeNodeView
              key={node.term.name}
              node={node}
              sessionId={sessionId}
            />
          ))}
        </div>
      ) : (
        <div className={`overflow-y-auto ${fullscreen ? 'flex-1' : 'max-h-[calc(100vh-20rem)]'}`}>
          {displayTerms.map((term) => (
            <GlossaryItem key={`${term.name}-${term.domain || ''}`} term={term} sessionId={sessionId} />
          ))}
        </div>
      )}

      {/* Taxonomy confirmation dialog */}
      {showDeleteDrafts && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm w-full mx-4 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <TrashIcon className="w-5 h-5 text-red-500" />
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                Delete Draft Terms
              </h3>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              This will permanently delete all glossary terms with <span className="font-medium">draft</span> status.
              Reviewed and approved terms will not be affected.
            </p>
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setShowDeleteDrafts(false)}
                className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  setShowDeleteDrafts(false)
                  setDeletingDrafts(true)
                  await deleteDrafts(sessionId)
                  setDeletingDrafts(false)
                }}
                className="text-xs px-3 py-1.5 rounded bg-red-500 text-white hover:bg-red-600"
              >
                Delete Drafts
              </button>
            </div>
          </div>
        </div>
      )}
      {showConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm w-full mx-4 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <SparklesIcon className="w-5 h-5 text-purple-500" />
              <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                Generate Taxonomy
              </h3>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
              This will use AI to analyze all entities and:
            </p>
            <ul className="text-xs text-gray-600 dark:text-gray-400 space-y-1 ml-4 list-disc">
              <li>Generate definitions for each entity</li>
              <li>Build a parent/child hierarchy</li>
              <li>Extract subject-verb-object relationships</li>
            </ul>
            <p className="text-xs text-gray-500 dark:text-gray-500">
              Generated terms will be marked as <span className="font-medium">draft</span> (AI-authored).
              You can review, edit, and promote them to domain config.
            </p>
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setShowConfirm(false)}
                className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowConfirm(false)
                  generateGlossary(sessionId)
                }}
                className="text-xs px-3 py-1.5 rounded bg-purple-500 text-white hover:bg-purple-600"
              >
                Generate
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  if (fullscreen) {
    return (
      <div className="fixed inset-0 z-40 bg-white dark:bg-gray-900 overflow-hidden flex flex-col">
        {content}
      </div>
    )
  }

  return content
}
