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
} from '@heroicons/react/24/outline'
import { useGlossaryStore } from '@/store/glossaryStore'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import {
  getGlossaryTerm,
  createRelationship,
  updateRelationshipVerb,
  deleteRelationship,
  updateGlossaryTerm,
  draftGlossaryDefinition,
  draftGlossaryAliases,
  listDomains,
} from '@/api/sessions'
import type { DomainInfo } from '@/api/sessions'
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

  const roots: TreeNode[] = []
  const orphans: GlossaryTerm[] = []

  for (const t of terms) {
    const node = nodeMap.get(t.name.toLowerCase())!
    if (t.parent_id) {
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
}: {
  rel: { id: string; subject: string; verb: string; object: string }
  sessionId: string
  onDeleted: (id: string) => void
  onUpdated: (id: string, verb: string) => void
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
    children: Array<{ name: string; display_name: string }>
    relationships: Array<{ id: string; subject: string; verb: string; object: string; confidence: number }>
  }>({ resources: [], parent: null, children: [], relationships: [] })
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoaded(false)
    getGlossaryTerm(sessionId, termName).then((data) => {
      if (!cancelled) {
        console.log(`[ConnectedResources] ${termName}: relationships=${(data.relationships || []).length}, parent=${!!data.parent}, children=${(data.children || []).length}, resources=${(data.connected_resources || []).length}`)
        setDetail({
          resources: data.connected_resources || [],
          parent: data.parent || null,
          children: data.children || [],
          relationships: data.relationships || [],
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

  const [graphOpen, setGraphOpen] = useState(false)

  const { resources, parent, children, relationships } = detail
  const hasConnections = !!(parent || children.length > 0 || relationships.length > 0)
  const hasContent = resources.length > 0 || hasConnections
  if (!hasContent) return null

  return (
    <div className="mt-1.5 space-y-1.5">
      {parent && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Parent</div>
          <div className="text-xs ml-2">
            <TermLink name={parent.name} displayName={parent.display_name} />
          </div>
        </div>
      )}
      {children.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">Children</div>
          {children.map((c, i) => (
            <div key={i} className="text-xs ml-2">
              <TermLink name={c.name} displayName={c.display_name} />
            </div>
          ))}
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
                relationships: prev.relationships.map(x => x.id === id ? { ...x, verb } : x),
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
  const { terms: allTerms, selectedName, updateTerm, refineTerm, deleteTerm } = useGlossaryStore()
  const isSelected = selectedName?.toLowerCase() === term.name.toLowerCase()
  const [isOpen, setIsOpen] = useState(false)
  const [isDefining, setIsDefining] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editDef, setEditDef] = useState('')

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
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300 flex-1 truncate">
          {term.display_name}
          {parentName && (
            <span className="text-xs font-normal text-gray-400 dark:text-gray-500 ml-1">
              &larr; {parentName}
            </span>
          )}
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
          <span className="text-xs px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 flex-shrink-0">
            {term.domain}
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
              <MinusSmallIcon className="w-3 h-3 text-gray-400" />
            ) : (
              <PlusSmallIcon className="w-3 h-3 text-gray-400" />
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

// Domain promote picker — assign/remove a term's domain
// Promotion ladder: (none) → domain → system (admin only). No downgrade.
// Term must be approved before promotion is available.
// Collect all descendant terms (recursive via parent_id)
function getDescendants(termName: string, allTerms: GlossaryTerm[]): GlossaryTerm[] {
  // Find IDs that could be this term's parent reference
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

// Can this term be directly promoted? Only defined terms that are approved.
function canPromote(term: GlossaryTerm): boolean {
  return term.glossary_status === 'defined' && term.status === 'approved'
}

// Does this term block a parent's cascade promotion?
// Self-describing (entities) don't block — they're session-level and implicitly fine.
// Only unapproved defined terms block.
function blocksCascade(term: GlossaryTerm): boolean {
  if (term.glossary_status === 'self_describing') return false
  return term.status !== 'approved'
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
  const [domains, setDomains] = useState<DomainInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [canPromoteToSystem, setCanPromoteToSystem] = useState(false)
  const [menuPos, setMenuPos] = useState<{ top: number; left: number }>({ top: 0, left: 0 })
  const [confirmCascade, setConfirmCascade] = useState<{ target: string; descendants: GlossaryTerm[]; blocked: GlossaryTerm[] } | null>(null)
  const btnRef = useRef<HTMLSpanElement>(null)
  const { terms: allTerms, updateTerm } = useGlossaryStore()

  // Close on click outside (must be before early returns — hooks rules)
  useEffect(() => {
    if (!open && !confirmCascade) return
    const handler = () => { setOpen(false); setConfirmCascade(null) }
    document.addEventListener('click', handler)
    return () => document.removeEventListener('click', handler)
  }, [open, confirmCascade])

  // Already at system level — no further promotion
  if (currentDomain === 'system') return null

  // Only defined + approved terms can be promoted
  const selfTerm = allTerms.find(t => t.name.toLowerCase() === termName.toLowerCase())
  if (!selfTerm || !canPromote(selfTerm)) return null

  const handleOpen = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (open) { setOpen(false); return }
    if (btnRef.current) {
      const rect = btnRef.current.getBoundingClientRect()
      setMenuPos({ top: rect.bottom + 4, left: rect.right })
    }
    setOpen(true)
    setLoading(true)
    const { useAuthStore } = await import('@/store/authStore')
    setCanPromoteToSystem(useAuthStore.getState().canWrite('tier_promote'))
    if (!currentDomain) {
      const activeDomains = useSessionStore.getState().session?.active_domains || []
      const { domains: allDomains } = await listDomains()
      setDomains(
        activeDomains.length > 0
          ? allDomains.filter((d) => activeDomains.includes(d.filename))
          : allDomains
      )
    }
    setLoading(false)
  }

  const checkAndPromote = (target: string) => {
    const descendants = getDescendants(termName, allTerms)
    const blocked = descendants.filter(d => blocksCascade(d))
    if (blocked.length > 0) {
      setConfirmCascade({ target, descendants, blocked })
      return
    }
    executeCascade(target, descendants)
  }

  const executeCascade = async (target: string, descendants: GlossaryTerm[]) => {
    // Promote self
    await updateTerm(sessionId, termName, { domain: target })
    // Cascade to defined+approved descendants (skip self-describing and unapproved)
    for (const d of descendants) {
      if (canPromote(d) && d.domain !== target) {
        await updateTerm(sessionId, d.name, { domain: target })
      }
    }
    setOpen(false)
    setConfirmCascade(null)
  }

  const handleSelect = (filename: string) => checkAndPromote(filename)
  const handlePromoteSystem = () => checkAndPromote('system')

  const iconColor = currentDomain
    ? 'text-blue-500'
    : 'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300'

  return (
    <span className="flex-shrink-0" onClick={(e) => e.stopPropagation()}>
      <span
        ref={btnRef}
        role="button"
        onClick={handleOpen}
        className={`p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 inline-flex ${iconColor}`}
        title={currentDomain ? 'Promote to system' : 'Assign to domain'}
      >
        <ArrowUpIcon className="w-3 h-3" />
      </span>
      {open && !confirmCascade && (
        <div
          className="fixed z-50 min-w-[10rem] bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg max-h-48 overflow-y-auto"
          style={{ top: menuPos.top, left: menuPos.left, transform: 'translateX(-100%)' }}
          onClick={(e) => e.stopPropagation()}
        >
          {loading ? (
            <div className="text-xs text-gray-400 px-2 py-1">Loading...</div>
          ) : currentDomain ? (
            canPromoteToSystem ? (
              <button
                onClick={handlePromoteSystem}
                className="w-full text-left text-xs px-2 py-1 hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-600 dark:text-purple-400"
              >
                Promote to system
              </button>
            ) : (
              <div className="text-xs text-gray-400 px-2 py-1">Admin required to promote to system</div>
            )
          ) : domains.length === 0 && !canPromoteToSystem ? (
            <div className="text-xs text-gray-400 px-2 py-1">No domains available</div>
          ) : (
            <>
              {domains.map((d) => (
                <button
                  key={d.filename}
                  onClick={() => handleSelect(d.filename)}
                  className="w-full text-left text-xs px-2 py-1 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 truncate"
                >
                  {d.name}
                  <span className="text-gray-400 ml-1">({d.filename})</span>
                </button>
              ))}
              {canPromoteToSystem && (
                <button
                  onClick={handlePromoteSystem}
                  className="w-full text-left text-xs px-2 py-1 hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-600 dark:text-purple-400 border-t border-gray-100 dark:border-gray-700"
                >
                  Promote to system
                </button>
              )}
            </>
          )}
        </div>
      )}
      {confirmCascade && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
          onClick={(e) => { if (e.target === e.currentTarget) setConfirmCascade(null) }}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-sm w-full mx-4 p-4 space-y-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Promote with children
            </div>
            {confirmCascade.blocked.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs text-amber-600 dark:text-amber-400">
                  {confirmCascade.blocked.length} descendant{confirmCascade.blocked.length > 1 ? 's' : ''} not yet approved:
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400 ml-2 max-h-24 overflow-y-auto">
                  {confirmCascade.blocked.map(d => (
                    <div key={d.name}>{d.display_name} <span className="text-gray-400">({d.status || 'draft'})</span></div>
                  ))}
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  These will be skipped. Promote the rest?
                </div>
              </div>
            )}
            {confirmCascade.descendants.filter(d => canPromote(d)).length > 0 && (
              <div className="text-xs text-gray-500 dark:text-gray-400">
                {confirmCascade.descendants.filter(d => canPromote(d)).length} descendant{confirmCascade.descendants.filter(d => canPromote(d)).length > 1 ? 's' : ''} will also be promoted.
              </div>
            )}
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={() => setConfirmCascade(null)}
                className="text-xs px-3 py-1.5 rounded text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
              <button
                onClick={() => executeCascade(confirmCascade.target, confirmCascade.descendants)}
                className="text-xs px-3 py-1.5 rounded bg-blue-500 text-white hover:bg-blue-600"
              >
                Promote
              </button>
            </div>
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
  type: 'focal' | 'parent' | 'child' | 'relationship'
  depth: number
  x: number
  y: number
}

interface PositionedEdge {
  sourceId: string
  targetId: string
  label: string
  type: 'parent' | 'child' | 'relationship'
}

const NODE_STYLES: Record<string, { fill: string; r: number }> = {
  focal: { fill: '#3b82f6', r: 24 },
  parent: { fill: '#a855f7', r: 18 },
  child: { fill: '#22c55e', r: 16 },
  relationship: { fill: '#9ca3af', r: 16 },
}

const GRAPH_W = 460
const GRAPH_H = 300

function TermGraphInline({
  sessionId,
  termName,
}: {
  sessionId: string
  termName: string
}) {
  const [graph, setGraph] = useState<{ nodes: PositionedNode[]; edges: PositionedEdge[] } | null>(null)
  const [loading, setLoading] = useState(true)
  const [empty, setEmpty] = useState(false)
  const [depth, setDepth] = useState(1)

  // BFS fetch neighborhood up to `depth` levels, then simulate layout
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setEmpty(false)
    setGraph(null)

    async function fetchNeighborhood() {
      const nodeMap = new Map<string, { id: string; label: string; type: PositionedNode['type']; depth: number }>()
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

          // Add the fetched node itself
          if (!nodeMap.has(name)) {
            nodeMap.set(name, {
              id: name,
              label: data.display_name || name,
              type: d === 0 ? 'focal' : 'relationship',
              depth: d,
            })
          }

          // Parent
          if (data.parent) {
            const pid = data.parent.name
            if (!nodeMap.has(pid)) {
              nodeMap.set(pid, { id: pid, label: data.parent.display_name, type: 'parent', depth: d + 1 })
            }
            addEdge(pid, name, data.parent_verb || 'has', 'parent')
            if (d + 1 < depth) queue.push([pid, d + 1])
          }

          // Children
          for (const c of data.children || []) {
            if (!nodeMap.has(c.name)) {
              nodeMap.set(c.name, { id: c.name, label: c.display_name, type: 'child', depth: d + 1 })
            }
            addEdge(name, c.name, c.parent_verb || 'has', 'child')
            if (d + 1 < depth) queue.push([c.name, d + 1])
          }

          // Relationships
          for (const r of data.relationships || []) {
            const partner = r.subject === name ? r.object : r.subject
            if (!nodeMap.has(partner)) {
              nodeMap.set(partner, { id: partner, label: partner, type: 'relationship', depth: d + 1 })
            }
            addEdge(
              r.subject === name ? name : partner,
              r.subject === name ? partner : name,
              r.verb,
              'relationship',
            )
            if (d + 1 < depth) queue.push([partner, d + 1])
          }
        }
      }

      if (cancelled) return

      const nodeArr = Array.from(nodeMap.values())

      if (nodeArr.length <= 1) {
        setEmpty(true)
        setLoading(false)
        return
      }

      // Scale simulation parameters with node count
      const nodeCount = nodeArr.length
      const linkDist = nodeCount > 15 ? 100 : 150
      const chargeStr = nodeCount > 15 ? -300 : -500
      const collideExtra = nodeCount > 15 ? 8 : 12

      // Build simulation-friendly copies (d3 mutates these in place)
      interface SimNode {
        id: string
        label: string
        type: PositionedNode['type']
        depth: number
        x?: number
        y?: number
        fx?: number | null
        fy?: number | null
      }
      const simNodes: SimNode[] = nodeArr.map((n) => ({ ...n }))
      const simEdges = rawEdges.map((e) => ({ source: e.source, target: e.target }))

      const cx = GRAPH_W / 2
      const cy = GRAPH_H / 2
      for (const n of simNodes) {
        if (n.type === 'focal' && n.depth === 0) { n.fx = cx; n.fy = cy }
      }

      const sim = forceSimulation<SimNode>(simNodes)
        .force('link', forceLink<SimNode, { source: string; target: string }>(simEdges).id((d) => d.id).distance(linkDist))
        .force('charge', forceManyBody().strength(chargeStr))
        .force('x', forceX<SimNode>(cx).strength(0.05))
        .force('y', forceY<SimNode>(cy).strength(0.05))
        .force('collide', forceCollide<SimNode>().radius((d) => {
          const baseR = NODE_STYLES[d.type]?.r || 16
          return Math.max(10, baseR - (d.depth * 2)) + collideExtra
        }))
        .stop()

      for (let i = 0; i < 200; i++) sim.tick()

      // Produce clean positioned output (new objects, no d3 refs)
      const posNodes: PositionedNode[] = simNodes.map((n) => {
        const baseR = NODE_STYLES[n.type]?.r || 16
        const r = Math.max(10, baseR - (n.depth * 2))
        return {
          id: n.id,
          label: n.label,
          type: n.type,
          depth: n.depth,
          x: Math.max(r + 40, Math.min(GRAPH_W - r - 40, n.x ?? cx)),
          y: Math.max(r + 20, Math.min(GRAPH_H - r - 20, n.y ?? cy)),
        }
      })

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

    return () => { cancelled = true }
  }, [sessionId, termName, depth])

  const handleNodeClick = (name: string) => {
    navigateToTerm(name)
  }

  // Build a position lookup for edge rendering
  const posMap = useMemo(() => {
    if (!graph) return new Map<string, PositionedNode>()
    const m = new Map<string, PositionedNode>()
    for (const n of graph.nodes) m.set(n.id, n)
    return m
  }, [graph])

  return (
    <div className="mt-1">
      <div className="flex items-center gap-2 mb-1">
        <label className="text-[10px] text-gray-500 dark:text-gray-400 select-none">Depth</label>
        <input
          type="range"
          min={1}
          max={4}
          value={depth}
          onChange={(e) => setDepth(Number(e.target.value))}
          disabled={loading}
          className="flex-1 h-1 accent-blue-500"
        />
        <span className="text-[10px] text-gray-500 dark:text-gray-400 w-3 text-center">{depth}</span>
      </div>
      {loading ? (
        <div className="text-xs text-gray-400 py-4 text-center">Loading graph...</div>
      ) : empty || !graph ? (
        <svg width="100%" viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`} className="select-none">
          <g transform={`translate(${GRAPH_W / 2},${GRAPH_H / 2})`}>
            <circle r={24} fill="#3b82f6" />
            <text
              textAnchor="middle"
              dominantBaseline="central"
              className="text-[10px] fill-white font-semibold pointer-events-none"
            >
              {termName.length > 10 ? termName.slice(0, 8) + '..' : termName}
            </text>
          </g>
        </svg>
      ) : (
        <>
          <svg width="100%" viewBox={`0 0 ${GRAPH_W} ${GRAPH_H}`} className="select-none">
            <defs>
              <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                <polygon points="0 0, 8 3, 0 6" fill="#9ca3af" />
              </marker>
            </defs>
            {/* Edges */}
            {graph.edges.map((e, i) => {
              const src = posMap.get(e.sourceId)
              const tgt = posMap.get(e.targetId)
              if (!src || !tgt) return null

              const srcR = Math.max(10, (NODE_STYLES[src.type]?.r || 16) - (src.depth * 2))
              const tgtR = Math.max(10, (NODE_STYLES[tgt.type]?.r || 16) - (tgt.depth * 2))
              const dx = tgt.x - src.x
              const dy = tgt.y - src.y
              const dist = Math.sqrt(dx * dx + dy * dy) || 1
              const startX = src.x + (dx / dist) * srcR
              const startY = src.y + (dy / dist) * srcR
              const endX = tgt.x - (dx / dist) * tgtR
              const endY = tgt.y - (dy / dist) * tgtR
              const midX = (src.x + tgt.x) / 2
              const midY = (src.y + tgt.y) / 2

              return (
                <g key={i}>
                  <line
                    x1={startX} y1={startY} x2={endX} y2={endY}
                    stroke={e.type === 'parent' ? '#a855f7' : e.type === 'child' ? '#22c55e' : '#9ca3af'}
                    strokeWidth={1.5}
                    strokeDasharray={e.type === 'relationship' ? undefined : '4 3'}
                    markerEnd={e.type === 'relationship' ? 'url(#arrowhead)' : undefined}
                  />
                  <text
                    x={midX} y={midY - 5}
                    textAnchor="middle"
                    className="text-[9px] fill-gray-400 dark:fill-gray-500 pointer-events-none"
                  >
                    {e.label}
                  </text>
                </g>
              )
            })}
            {/* Nodes */}
            {graph.nodes.map((n) => {
              const style = NODE_STYLES[n.type]
              const r = Math.max(10, style.r - (n.depth * 2))
              return (
                <g
                  key={n.id}
                  transform={`translate(${n.x},${n.y})`}
                  className="cursor-pointer"
                  onClick={() => handleNodeClick(n.id)}
                >
                  <circle r={r} fill={style.fill} />
                  <title>{n.label}</title>
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
          </svg>
          {/* Legend */}
          <div className="flex items-center gap-3 text-[10px] text-gray-400 mt-1">
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-blue-500" /> focal</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-purple-400" /> parent</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-green-400" /> child</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-gray-400" /> relationship</span>
          </div>
        </>
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
