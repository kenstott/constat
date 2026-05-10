// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState, useMemo } from 'react'
import { useQuery } from '@apollo/client'
import { BookOpenIcon } from '@heroicons/react/24/outline'
import { useSessionContext } from '@/contexts/SessionContext'
import { useActiveDomains } from '@/hooks/useDomains'
import { HANDBOOK_QUERY } from '@/graphql/operations/handbook'
import { HandbookSection, type HandbookSectionData } from './HandbookSection'
import { SkeletonLoader } from '@/components/common/SkeletonLoader'

/** Ordered section keys matching the spec. */
const SECTION_ORDER = [
  'overview',
  'data_sources',
  'key_entities',
  'glossary',
  'learned_rules',
  'common_patterns',
  'agents_skills',
  'known_limitations',
] as const

/** Human-readable labels for sidebar nav. */
const SECTION_LABELS: Record<string, string> = {
  overview: 'Overview',
  data_sources: 'Data Sources',
  key_entities: 'Key Entities',
  glossary: 'Glossary',
  learned_rules: 'Learned Rules',
  common_patterns: 'Common Patterns',
  agents_skills: 'Agents & Skills',
  known_limitations: 'Known Limitations',
}

interface HandbookSectionsMap {
  [key: string]: {
    title: string
    content: Array<{
      key: string
      display: string
      metadata?: Record<string, unknown> | null
      editable: boolean
    }>
    last_updated?: string | null
  }
}

export function DomainHandbookPage() {
  const { sessionId } = useSessionContext()
  const { activeDomains } = useActiveDomains()
  const [selectedDomain, setSelectedDomain] = useState<string | undefined>(undefined)
  const [activeSection, setActiveSection] = useState<string>(SECTION_ORDER[0])

  const effectiveDomain = selectedDomain ?? (activeDomains.length > 0 ? activeDomains[0] : undefined)

  const { data, loading, error } = useQuery(HANDBOOK_QUERY, {
    variables: { sessionId: sessionId!, domain: effectiveDomain },
    skip: !sessionId,
    fetchPolicy: 'cache-and-network',
  })

  const sections: HandbookSectionsMap = useMemo(() => {
    if (!data?.handbook?.sections) return {}
    // sections comes as JSON scalar — may already be parsed
    const raw = data.handbook.sections
    if (typeof raw === 'string') {
      try { return JSON.parse(raw) } catch { return {} }
    }
    return raw as HandbookSectionsMap
  }, [data])

  const orderedSections = useMemo(() => {
    return SECTION_ORDER.filter((key) => key in sections).map((key) => ({
      key,
      data: toSectionData(sections[key]),
    }))
  }, [sections])

  const handleSectionClick = (key: string) => {
    setActiveSection(key)
    const el = document.getElementById(`section-${key}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  if (!sessionId) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
        No active session. Create or select a session to view the handbook.
      </div>
    )
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 border-r border-gray-200 dark:border-gray-700 p-4 space-y-4 overflow-y-auto">
        {/* Domain selector */}
        <div>
          <label
            htmlFor="handbook-domain"
            className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
          >
            Domain
          </label>
          <select
            id="handbook-domain"
            value={effectiveDomain ?? ''}
            onChange={(e) => setSelectedDomain(e.target.value || undefined)}
            className="w-full text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
          >
            {activeDomains.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </div>

        {/* Section nav */}
        <nav aria-label="Handbook sections">
          <ul className="space-y-1">
            {SECTION_ORDER.map((key) => {
              const inData = key in sections
              return (
                <li key={key}>
                  <button
                    onClick={() => handleSectionClick(key)}
                    disabled={!inData && !loading}
                    className={`w-full text-left text-sm px-2 py-1.5 rounded transition-colors ${
                      activeSection === key
                        ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium'
                        : inData
                          ? 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                          : 'text-gray-400 dark:text-gray-600 cursor-not-allowed'
                    }`}
                  >
                    {SECTION_LABELS[key] ?? key}
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center gap-3 mb-2">
          <BookOpenIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            Domain Handbook
            {data?.handbook?.domain && (
              <span className="ml-2 text-base font-normal text-gray-500 dark:text-gray-400">
                — {data.handbook.domain}
              </span>
            )}
          </h1>
        </div>

        {/* Summary */}
        {data?.handbook?.summary && (
          <p className="text-sm text-gray-700 dark:text-gray-300 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 rounded-lg p-3">
            {data.handbook.summary}
          </p>
        )}

        {/* Loading */}
        {loading && orderedSections.length === 0 && (
          <div className="space-y-4" data-testid="handbook-loading">
            <SkeletonLoader lines={5} />
            <SkeletonLoader lines={3} />
            <SkeletonLoader lines={4} />
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
            Failed to load handbook: {error.message}
          </div>
        )}

        {/* Sections */}
        {orderedSections.map(({ key, data: sectionData }) => (
          <HandbookSection
            key={key}
            sectionKey={key}
            section={sectionData}
            sessionId={sessionId!}
            defaultExpanded={key === 'overview'}
          />
        ))}

        {/* Empty state */}
        {!loading && !error && orderedSections.length === 0 && (
          <div className="flex flex-col items-center justify-center h-48 text-gray-400 dark:text-gray-500">
            <BookOpenIcon className="h-10 w-10 mb-2" />
            <p className="text-sm">No handbook data available for this domain.</p>
          </div>
        )}

        {/* Generated timestamp */}
        {data?.handbook?.generatedAt && (
          <p className="text-xs text-gray-400 dark:text-gray-500 text-right pt-2">
            Generated: {new Date(data.handbook.generatedAt).toLocaleString()}
          </p>
        )}
      </main>
    </div>
  )
}

/** Convert raw section JSON to the typed data structure. */
function toSectionData(raw: HandbookSectionsMap[string]): HandbookSectionData {
  return {
    title: raw.title,
    content: (raw.content ?? []).map((e) => ({
      key: e.key,
      display: e.display,
      metadata: e.metadata ?? null,
      editable: e.editable ?? false,
    })),
    lastUpdated: raw.last_updated ?? null,
  }
}
