// Glossary Panel — unified glossary view replacing EntityAccordion

import { useEffect, useMemo, useState, useCallback } from 'react'
import {
  ChevronRightIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  PlusIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline'
import { useGlossaryStore } from '@/store/glossaryStore'
import type { GlossaryTerm, GlossaryEditorialStatus } from '@/types/api'

interface GlossaryPanelProps {
  sessionId: string
}

// Semantic type badge colors (reused from EntityAccordion)
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
}: {
  term: GlossaryTerm
  sessionId: string
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
        <div className="pl-6 pb-2 space-y-1.5">
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

export default function GlossaryPanel({ sessionId }: GlossaryPanelProps) {
  const {
    terms,
    filters,
    totalDefined,
    totalSelfDescribing,
    loading,
    fetchTerms,
    setFilter,
  } = useGlossaryStore()

  const [search, setSearch] = useState('')

  useEffect(() => {
    fetchTerms(sessionId)
  }, [sessionId, filters.scope, fetchTerms])

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

  const handleScopeChange = useCallback(
    (scope: 'all' | 'defined' | 'self_describing') => {
      setFilter({ scope })
    },
    [setFilter]
  )

  return (
    <div className="space-y-2">
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

      {/* Terms list */}
      {loading ? (
        <div className="text-xs text-gray-400 py-4 text-center">Loading glossary...</div>
      ) : displayTerms.length === 0 ? (
        <div className="text-xs text-gray-400 py-4 text-center">
          {search ? 'No matching terms' : 'No glossary terms'}
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
