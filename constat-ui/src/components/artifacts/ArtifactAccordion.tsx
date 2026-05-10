// Copyright (c) 2025 Kenneth Stott
// Canary: db4cdae5-6389-4af0-9ad2-b5e1fd9d7712
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Artifact Accordion component

import { ReactNode } from 'react'
import { ChevronDownIcon } from '@heroicons/react/24/outline'
import { useReactiveVar } from '@apollo/client'
import { expandedSectionsVar, toggleSection } from '@/graphql/ui-state'

interface AccordionSectionProps {
  id: string
  title: string
  count?: number
  icon?: ReactNode
  action?: ReactNode
  command?: string  // Slash command for this section (e.g., "/tables")
  children: ReactNode
}

export function AccordionSection({ id, title, count, icon, action, command, children }: AccordionSectionProps) {
  const expandedArtifactSections = useReactiveVar(expandedSectionsVar)
  const isExpanded = expandedArtifactSections.includes(id)

  return (
    <div id={`section-${id}`} className="border-b border-gray-200 dark:border-gray-700">
      <div className="flex items-center">
        <button
          onClick={() => toggleSection(id)}
          className="flex-1 flex items-center gap-2 px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
        >
          {icon && <span className="text-gray-500 dark:text-gray-400">{icon}</span>}
          <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
            {title}
          </span>
          {/* Command hint - fixed width for consistent alignment */}
          <span className={`w-20 text-[10px] font-mono px-1.5 py-0.5 rounded text-right ${
            command
              ? 'text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-700'
              : 'text-transparent bg-transparent'
          }`}>
            {command || ''}
          </span>
          {/* Count badge */}
          <span className={`min-w-[2rem] text-center text-xs px-2 py-0.5 rounded-full ${
            count !== undefined && count > 0
              ? 'text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-gray-700'
              : 'text-gray-300 dark:text-gray-600 bg-gray-50 dark:bg-gray-800'
          }`}>
            {count ?? 0}
          </span>
          <ChevronDownIcon
            className={`w-4 h-4 text-gray-400 transition-transform ${
              isExpanded ? 'rotate-180' : ''
            }`}
          />
        </button>
        <div className="w-14 pr-2 flex justify-end">{action}</div>
      </div>
      {isExpanded && (
        <div className="px-4 py-3 bg-white dark:bg-gray-800">{children}</div>
      )}
    </div>
  )
}