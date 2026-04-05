// Copyright (c) 2025 Kenneth Stott
// Canary: 2b4c8d1e-f3a5-4e9b-8c7d-6f0a1b2c3d4e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import type { ApiEndpointInfo } from '@/types/api'

interface ApiEndpointPanelProps {
  endpoints: ApiEndpointInfo[]
  expandedEndpoint: string | null
  setExpandedEndpoint: (name: string | null) => void
}

export function ApiEndpointPanel({ endpoints, expandedEndpoint, setExpandedEndpoint }: ApiEndpointPanelProps) {
  const renderEndpoint = (ep: ApiEndpointInfo) => (
    <div key={ep.name}>
      <button
        onClick={() => setExpandedEndpoint(expandedEndpoint === ep.name ? null : ep.name)}
        className={`w-full text-left text-xs px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
          expandedEndpoint === ep.name
            ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
            : 'text-gray-600 dark:text-gray-400'
        }`}
      >
        <div className="flex items-center gap-1.5">
          {expandedEndpoint === ep.name ? (
            <ChevronDownIcon className="w-2.5 h-2.5 flex-shrink-0" />
          ) : (
            <ChevronRightIcon className="w-2.5 h-2.5 flex-shrink-0" />
          )}
          <span className="font-medium">{ep.name}</span>
          {ep.return_type && (
            <span className="font-mono text-[10px] text-purple-600 dark:text-purple-400">{ep.return_type}</span>
          )}
        </div>
        {ep.description && (
          <p className="text-[10px] text-gray-400 mt-0.5 ml-4">{ep.description}</p>
        )}
      </button>
      {expandedEndpoint === ep.name && ep.fields.length > 0 && (
        <div className="ml-6 mt-1 mb-2 border-l-2 border-gray-200 dark:border-gray-700 pl-2 space-y-0.5">
          <p className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Fields</p>
          {ep.fields.map((f) => (
            <div key={f.name} className="text-xs flex items-baseline gap-1.5">
              <span className="font-medium text-gray-700 dark:text-gray-300">{f.name}</span>
              <span className="font-mono text-[10px] text-purple-600 dark:text-purple-400">{f.type}</span>
              {f.is_required && <span className="text-[9px] text-red-500">required</span>}
              {f.description && <span className="text-gray-400 truncate">{f.description}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  )

  const renderSection = (label: string, items: ApiEndpointInfo[]) =>
    items.length > 0 ? (
      <div key={label}>
        <p className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
          {label} <span className="font-normal">({items.length})</span>
        </p>
        <div className="space-y-0.5">{items.map(renderEndpoint)}</div>
      </div>
    ) : null

  const gqlKinds: Record<string, string> = {
    graphql_query: 'Queries', graphql_mutation: 'Mutations',
    graphql_subscription: 'Subscriptions', graphql_type: 'Types',
  }
  const gqlGroups = Object.entries(gqlKinds)
    .map(([kind, label]) => ({ label, items: endpoints.filter((ep) => ep.kind === kind) }))
    .filter((g) => g.items.length > 0)

  const restOps = endpoints.filter((ep) => ep.kind === 'rest' || (!ep.kind?.startsWith('graphql_') && !ep.kind?.includes('/') && ep.http_method))
  const restTypes = endpoints.filter((ep) => ep.kind === 'openapi/model')
  const restOther = endpoints.filter((ep) => !ep.kind?.startsWith('graphql_') && ep.kind !== 'rest' && ep.kind !== 'openapi/model' && !ep.http_method)
  const methodOrder = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
  const restMethods = [...new Set(restOps.map((ep) => ep.http_method || 'OTHER'))]
    .sort((a, b) => (methodOrder.indexOf(a) === -1 ? 99 : methodOrder.indexOf(a)) - (methodOrder.indexOf(b) === -1 ? 99 : methodOrder.indexOf(b)))
  const restGroups = [
    ...restMethods.map((method) => ({ label: method, items: restOps.filter((ep) => (ep.http_method || 'OTHER') === method) })),
    ...(restTypes.length > 0 ? [{ label: 'Types', items: restTypes }] : []),
    ...(restOther.length > 0 ? [{ label: 'Other', items: restOther }] : []),
  ].filter((g) => g.items.length > 0)

  return (
    <div className="space-y-2">
      {gqlGroups.map((g) => renderSection(g.label, g.items))}
      {restGroups.map((g) => renderSection(g.label, g.items))}
    </div>
  )
}
