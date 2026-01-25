// Entity Accordion with alphabetical ordering and reference locations

import { useMemo, useState } from 'react'
import { ChevronRightIcon } from '@heroicons/react/24/outline'
import type { Entity } from '@/types/api'

interface EntityAccordionProps {
  entities: Entity[]
}

interface EntityItemProps {
  entity: Entity
}

function EntityItem({ entity }: EntityItemProps) {
  const [isOpen, setIsOpen] = useState(false)

  // Type badge color
  const typeColor = {
    table: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    column: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
    concept: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
    business_term: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
    api_endpoint: 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400',
    api_field: 'bg-pink-100 text-pink-700 dark:bg-pink-900/30 dark:text-pink-400',
    api_schema: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  }[entity.type] || 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'

  const references = entity.references || []
  const sources = entity.sources || []

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
          {entity.original_name && entity.original_name !== entity.name && (
            <div className="text-xs text-gray-400 dark:text-gray-500">
              <span className="italic">Original: </span>
              <span className="font-mono text-gray-500 dark:text-gray-400">{entity.original_name}</span>
            </div>
          )}
          {/* Check metadata for original_name as fallback */}
          {!entity.original_name && entity.metadata?.original_name && entity.metadata.original_name !== entity.name && (
            <div className="text-xs text-gray-400 dark:text-gray-500">
              <span className="italic">Original: </span>
              <span className="font-mono text-gray-500 dark:text-gray-400">{String(entity.metadata.original_name)}</span>
            </div>
          )}
          {/* Sources */}
          {sources.length > 0 && (
            <div className="flex items-center gap-1 flex-wrap">
              <span className="text-xs text-gray-400 dark:text-gray-500">Sources:</span>
              {sources.map((src, idx) => (
                <span
                  key={idx}
                  className="text-xs px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded"
                >
                  {src}
                </span>
              ))}
            </div>
          )}
          {/* References */}
          {references.length === 0 ? (
            <p className="text-xs text-gray-400 dark:text-gray-500">
              No reference locations available
            </p>
          ) : (
            <div className="space-y-1">
              <span className="text-xs text-gray-400 dark:text-gray-500">Referenced in:</span>
              {references.map((ref, idx) => (
                <div
                  key={idx}
                  className="text-xs text-gray-500 dark:text-gray-400 pl-2"
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
  // Sort entities alphabetically by name
  const sortedEntities = useMemo(() => {
    return [...entities].sort((a, b) =>
      a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    )
  }, [entities])

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

  if (entities.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        No entities extracted yet
      </p>
    )
  }

  // For small lists, just show flat list
  if (!groupedEntities) {
    return (
      <div className="space-y-0">
        {sortedEntities.map((entity) => (
          <EntityItem key={entity.id} entity={entity} />
        ))}
      </div>
    )
  }

  // For large lists, show grouped by letter
  return (
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
  )
}