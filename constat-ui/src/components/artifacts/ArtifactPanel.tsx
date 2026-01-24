// Artifact Panel container

import { useEffect } from 'react'
import {
  ChartBarIcon,
  TableCellsIcon,
  CodeBracketIcon,
  LightBulbIcon,
  TagIcon,
  StarIcon,
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { AccordionSection } from './ArtifactAccordion'
import { TableViewer } from './TableViewer'
import { CodeViewer } from './CodeViewer'

export function ArtifactPanel() {
  const { session } = useSessionStore()
  const {
    artifacts,
    tables,
    facts,
    entities,
    selectedTable,
    selectTable,
    fetchArtifacts,
    fetchTables,
    fetchFacts,
    fetchEntities,
  } = useArtifactStore()

  // Fetch data when session changes
  useEffect(() => {
    if (session) {
      fetchArtifacts(session.session_id)
      fetchTables(session.session_id)
      fetchFacts(session.session_id)
      fetchEntities(session.session_id)
    }
  }, [session, fetchArtifacts, fetchTables, fetchFacts, fetchEntities])

  // Visualizations: charts, images, HTML reports, etc.
  const visualArtifacts = artifacts.filter((a) =>
    ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image'].includes(a.artifact_type)
  )
  const codeArtifacts = artifacts.filter((a) =>
    ['code', 'output'].includes(a.artifact_type)
  )
  // Key artifacts: marked as important results to keep
  const keyArtifacts = artifacts.filter((a) => a.is_key_result)

  if (!session) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          No session active
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Key Artifacts */}
      <AccordionSection
        id="artifacts"
        title="Artifacts"
        count={keyArtifacts.length}
        icon={<StarIcon className="w-4 h-4" />}
      >
        {keyArtifacts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No key artifacts yet</p>
        ) : (
          <div className="space-y-3">
            {keyArtifacts.map((artifact) => (
              <div
                key={artifact.id}
                className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg"
              >
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {artifact.title || artifact.name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {artifact.artifact_type} · Step {artifact.step_number}
                </p>
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* Visualizations */}
      <AccordionSection
        id="visualizations"
        title="Visualizations"
        count={visualArtifacts.length}
        icon={<ChartBarIcon className="w-4 h-4" />}
      >
        {visualArtifacts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No visualizations yet</p>
        ) : (
          <div className="space-y-3">
            {visualArtifacts.map((artifact) => (
              <div
                key={artifact.id}
                className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
              >
                <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {artifact.title || artifact.name}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Step {artifact.step_number}
                </p>
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* Tables */}
      <AccordionSection
        id="tables"
        title="Tables"
        count={tables.length}
        icon={<TableCellsIcon className="w-4 h-4" />}
      >
        {tables.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No tables yet</p>
        ) : (
          <div className="space-y-3">
            {/* Table list */}
            <div className="flex flex-wrap gap-2">
              {tables.map((table) => (
                <button
                  key={table.name}
                  onClick={() =>
                    selectTable(selectedTable === table.name ? null : table.name)
                  }
                  className={`px-3 py-1.5 text-xs rounded-md transition-colors ${
                    selectedTable === table.name
                      ? 'bg-primary-100 dark:bg-primary-900 text-primary-700 dark:text-primary-300'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
                  }`}
                >
                  {table.name}
                  <span className="ml-1.5 text-gray-400">({table.row_count})</span>
                </button>
              ))}
            </div>

            {/* Selected table viewer */}
            {selectedTable && <TableViewer tableName={selectedTable} />}
          </div>
        )}
      </AccordionSection>

      {/* Code */}
      <AccordionSection
        id="code"
        title="Code"
        count={codeArtifacts.length}
        icon={<CodeBracketIcon className="w-4 h-4" />}
      >
        {codeArtifacts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No code yet</p>
        ) : (
          <div className="space-y-3">
            {codeArtifacts.map((artifact) => (
              <div key={artifact.id}>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  Step {artifact.step_number}: {artifact.name}
                </p>
                <CodeViewer
                  code="# Code will be loaded when selected"
                  language="python"
                />
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* Facts */}
      <AccordionSection
        id="facts"
        title="Facts"
        count={facts.length}
        icon={<LightBulbIcon className="w-4 h-4" />}
      >
        {facts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No facts yet</p>
        ) : (
          <div className="space-y-2">
            {facts.map((fact) => (
              <div
                key={fact.name}
                className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-start justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {fact.name}
                  </span>
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {fact.source}
                  </span>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  {String(fact.value)}
                </p>
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* Entities */}
      <AccordionSection
        id="entities"
        title="Entities"
        count={entities.length}
        icon={<TagIcon className="w-4 h-4" />}
      >
        {entities.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">
            No entities extracted yet
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {entities.map((entity) => (
              <span
                key={entity.id}
                className="px-2 py-1 text-xs bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-md"
                title={entity.type}
              >
                {entity.name}
              </span>
            ))}
          </div>
        )}
      </AccordionSection>
    </div>
  )
}