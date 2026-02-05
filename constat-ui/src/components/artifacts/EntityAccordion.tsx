// Entity Accordion with alphabetical ordering and reference locations

import { useMemo, useState } from 'react'
import { ChevronRightIcon, MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline'
import type { Entity } from '@/types/api'

interface EntityAccordionProps {
  entities: Entity[]
}

// All possible entity types
const ENTITY_TYPES = [
  { value: 'table', label: 'Tables', color: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400' },
  { value: 'column', label: 'Columns', color: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400' },
  { value: 'concept', label: 'Concepts', color: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' },
  { value: 'action', label: 'Actions', color: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400' },
  { value: 'business_term', label: 'Business Terms', color: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400' },
  { value: 'api_endpoint', label: 'API Endpoints', color: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400' },
  { value: 'api_field', label: 'API Fields', color: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400' },
  { value: 'api_schema', label: 'API Schemas', color: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400' },
] as const

interface EntityItemProps {
  entity: Entity
}

function EntityItem({ entity }: EntityItemProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [showRefs, setShowRefs] = useState(false)

  // Get original name for display (if different from normalized name)
  const originalName: string | undefined =
    (typeof entity.original_name === 'string' ? entity.original_name : undefined) ||
    (typeof entity.metadata?.original_name === 'string' ? entity.metadata.original_name : undefined)
  const showOriginalName = originalName && originalName !== entity.name

  // Type badge color
  const typeColor = {
    table: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    column: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
    concept: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    action: 'bg-cyan-100 text-cyan-700 dark:bg-cyan-900/30 dark:text-cyan-400',
    business_term: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    api_endpoint: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
    api_field: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
    api_schema: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  }[entity.type] || 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'

  const references = entity.references || []

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
          {entity.name}
        </span>
        <span className={`text-xs px-1.5 py-0.5 rounded flex-shrink-0 ${typeColor}`}>
          {entity.type}
        </span>
        {entity.mention_count > 0 && (
          <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
            {entity.mention_count}x
          </span>
        )}
      </button>
      {isOpen && (
        <div className="pl-6 pb-2 space-y-2">
          {/* Original name if different from display name */}
          {showOriginalName && (
            <div className="text-xs text-gray-400 dark:text-gray-500">
              <span className="italic">Original: </span>
              <span className="font-mono text-gray-500 dark:text-gray-400">{originalName}</span>
            </div>
          )}
          {/* Sources - show aggregated high-level source names */}
          {references.length > 0 && (() => {
            // Extract high-level source from detailed paths
            // e.g., "api:catfacts.GET /breeds.coat" -> "catfacts (API)"
            // e.g., "Table: shipments" with section "Database: inventory" -> "inventory (DB)"
            // e.g., "Database: sales_data" -> "sales_data (DB)"
            // e.g., "business_rules" -> "business_rules"
            const getHighLevelSource = (ref: { document?: string; section?: string }): string => {
              const doc = ref.document || ''
              const section = ref.section || ''
              if (doc.startsWith('api:')) {
                const apiPart = doc.substring(4) // Remove "api:"
                const apiName = apiPart.split('.')[0] // Get first segment
                return `${apiName} (API)`
              }
              // Table: X with Database: Y in section -> Y (DB)
              if (doc.startsWith('Table: ') && section.startsWith('Database: ')) {
                return `${section.substring(10)} (DB)`
              }
              if (doc.startsWith('Database: ')) {
                return `${doc.substring(10)} (DB)`
              }
              if (doc.startsWith('__') && doc.includes('_metadata__')) {
                return 'Schema metadata'
              }
              return doc
            }
            const uniqueSources = [...new Set(references.map(r => getHighLevelSource(r)).filter(Boolean))]
            return (
              <div className="flex items-center gap-1 flex-wrap">
                <span className="text-xs text-gray-400 dark:text-gray-500">Sources:</span>
                {uniqueSources.map((src, idx) => (
                  <span
                    key={idx}
                    className="text-xs px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded"
                  >
                    {src}
                  </span>
                ))}
              </div>
            )
          })()}
          {/* References - collapsible */}
          {references.length === 0 ? (
            <p className="text-xs text-gray-400 dark:text-gray-500">
              No reference locations available
            </p>
          ) : (
            <div className="space-y-1">
              <button
                onClick={() => setShowRefs(!showRefs)}
                className="flex items-center gap-1 text-xs text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <ChevronRightIcon
                  className={`w-3 h-3 transition-transform ${showRefs ? 'rotate-90' : ''}`}
                />
                <span>Referenced in ({references.length})</span>
              </button>
              {showRefs && references.map((ref, idx) => (
                <div
                  key={idx}
                  className="text-xs text-gray-500 dark:text-gray-400 pl-4"
                >
                  <div className="flex items-center gap-2">
                    {ref.mention_text && (
                      <span className="font-mono text-blue-600 dark:text-blue-400">"{ref.mention_text}"</span>
                    )}
                    <span className="text-gray-400 dark:text-gray-500">in</span>
                    <span className="font-medium">{ref.document}</span>
                    {ref.section && (
                      <span className="text-gray-400 dark:text-gray-500">
                        ({ref.section})
                      </span>
                    )}
                    {ref.mentions > 1 && (
                      <span className="text-gray-400 dark:text-gray-500">
                        {ref.mentions}x
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function EntityAccordion({ entities }: EntityAccordionProps) {
  const [searchText, setSearchText] = useState('')
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set())

  // Get types that actually exist in the data
  const availableTypes = useMemo(() => {
    const types = new Set(entities.map(e => e.type))
    return ENTITY_TYPES.filter(t => types.has(t.value))
  }, [entities])

  // Filter entities based on search and type filters
  const filteredEntities = useMemo(() => {
    return entities.filter(entity => {
      // Type filter
      if (selectedTypes.size > 0 && !selectedTypes.has(entity.type)) {
        return false
      }
      // Text search
      if (searchText) {
        const search = searchText.toLowerCase()
        const nameMatch = entity.name.toLowerCase().includes(search)
        const originalMatch = (entity.original_name || entity.metadata?.original_name || '')
          .toString().toLowerCase().includes(search)
        const refMatch = entity.references?.some(ref =>
          ref.document?.toLowerCase().includes(search) ||
          ref.mention_text?.toLowerCase().includes(search)
        )
        return nameMatch || originalMatch || refMatch
      }
      return true
    })
  }, [entities, searchText, selectedTypes])

  // Sort filtered entities alphabetically by name
  const sortedEntities = useMemo(() => {
    return [...filteredEntities].sort((a, b) =>
      a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    )
  }, [filteredEntities])

  // Group by first letter for large lists
  const groupedEntities = useMemo(() => {
    if (sortedEntities.length < 20) {
      return null // Don't group for small lists
    }
    const groups: Record<string, Entity[]> = {}
    for (const entity of sortedEntities) {
      const firstLetter = entity.name[0]?.toUpperCase() || '#'
      if (!groups[firstLetter]) {
        groups[firstLetter] = []
      }
      groups[firstLetter].push(entity)
    }
    return groups
  }, [sortedEntities])

  const toggleType = (type: string) => {
    const newTypes = new Set(selectedTypes)
    if (newTypes.has(type)) {
      newTypes.delete(type)
    } else {
      newTypes.add(type)
    }
    setSelectedTypes(newTypes)
  }

  const clearFilters = () => {
    setSearchText('')
    setSelectedTypes(new Set())
  }

  const hasFilters = searchText || selectedTypes.size > 0

  if (entities.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        No entities extracted yet
      </p>
    )
  }

  return (
    <div className="space-y-2">
      {/* Filter controls */}
      <div className="space-y-2">
        {/* Search input */}
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search entities..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="w-full pl-8 pr-8 py-1.5 text-sm border border-gray-200 dark:border-gray-700 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
          />
          {searchText && (
            <button
              onClick={() => setSearchText('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <XMarkIcon className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Type filter chips */}
        {availableTypes.length > 1 && (
          <div className="flex flex-wrap gap-1">
            {availableTypes.map(typeInfo => {
              const isSelected = selectedTypes.has(typeInfo.value)
              const count = entities.filter(e => e.type === typeInfo.value).length
              return (
                <button
                  key={typeInfo.value}
                  onClick={() => toggleType(typeInfo.value)}
                  className={`text-[10px] px-1.5 py-0.5 rounded-full transition-all ${
                    isSelected
                      ? typeInfo.color + ' ring-1 ring-offset-1 ring-gray-400 dark:ring-gray-500'
                      : 'bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {typeInfo.label} ({count})
                </button>
              )
            })}
            {hasFilters && (
              <button
                onClick={clearFilters}
                className="text-[10px] px-1.5 py-0.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                Clear
              </button>
            )}
          </div>
        )}

        {/* Results count */}
        {hasFilters && (
          <p className="text-xs text-gray-400 dark:text-gray-500">
            Showing {sortedEntities.length} of {entities.length} entities
          </p>
        )}
      </div>

      {/* Entity list */}
      {sortedEntities.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">
          No entities match your filters
        </p>
      ) : !groupedEntities ? (
        // For small lists, just show flat list
        <div className="space-y-0">
          {sortedEntities.map((entity) => (
            <EntityItem key={entity.id} entity={entity} />
          ))}
        </div>
      ) : (
        // For large lists, show grouped by letter
        <div className="space-y-2">
          {Object.entries(groupedEntities).map(([letter, letterEntities]) => (
            <div key={letter}>
              <div className="text-xs font-semibold text-gray-400 dark:text-gray-500 px-1 py-1 sticky top-0 bg-white dark:bg-gray-900">
                {letter}
              </div>
              <div className="space-y-0">
                {letterEntities.map((entity) => (
                  <EntityItem key={entity.id} entity={entity} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}