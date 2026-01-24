// Artifact Accordion component

import { ReactNode } from 'react'
import { ChevronDownIcon } from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'

interface AccordionSectionProps {
  id: string
  title: string
  count?: number
  icon?: ReactNode
  children: ReactNode
}

export function AccordionSection({ id, title, count, icon, children }: AccordionSectionProps) {
  const { expandedArtifactSections, toggleArtifactSection } = useUIStore()
  const isExpanded = expandedArtifactSections.includes(id)

  return (
    <div className="border-b border-gray-200 dark:border-gray-700">
      <button
        onClick={() => toggleArtifactSection(id)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
      >
        {icon && <span className="text-gray-500 dark:text-gray-400">{icon}</span>}
        <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
          {title}
        </span>
        {count !== undefined && count > 0 && (
          <span className="text-xs text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-700 px-2 py-0.5 rounded-full">
            {count}
          </span>
        )}
        <ChevronDownIcon
          className={`w-4 h-4 text-gray-400 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
        />
      </button>
      {isExpanded && (
        <div className="px-4 py-3 bg-white dark:bg-gray-800">{children}</div>
      )}
    </div>
  )
}