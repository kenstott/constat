// Copyright (c) 2025 Kenneth Stott
// Canary: 80d5d7d2-ccfd-4991-bb96-6f16a42470f2
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useEffect, useState } from 'react'
import {
  AcademicCapIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon,
  ArrowsRightLeftIcon,
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  PlusIcon,
  BeakerIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline'
import { AccordionSection } from '../ArtifactAccordion'
import { SkeletonLoader } from '../../common/SkeletonLoader'
import { DomainBadge } from '../../common/DomainBadge'
import { ScopeBadge } from '../../common/ScopeBadge'
import RegressionPanel from '../RegressionPanel'
import { apolloClient } from '@/graphql/client'
import { COMPACT_LEARNINGS } from '@/graphql/operations/learnings'
import { MOVE_DOMAIN_RULE } from '@/graphql/operations/domains'
import {
  FINE_TUNE_JOBS_QUERY, FINE_TUNE_PROVIDERS_QUERY, START_FINE_TUNE_JOB,
  CANCEL_FINE_TUNE_JOB, DELETE_FINE_TUNE_JOB, toFineTuneJob, toFineTuneProvider,
} from '@/graphql/operations/fine-tune'
import { useLearnings } from '@/hooks/useLearnings'
import { useTestableDomains } from '@/hooks/useTesting'
import { downloadSimpleExemplars } from '@/api/sessions'
import type { FineTuneJob, FineTuneProvider } from '@/types/api'

export interface ImprovementsSectionProps {
  sessionId: string
  improvementVisible: boolean
  canWrite: boolean
  canSeeSection: (section: string) => boolean
  domainList: { filename: string; name: string }[]
  addRule: (summary: string) => Promise<void>
  updateRule: (id: string, summary: string) => Promise<void>
  deleteRule: (id: string) => Promise<void>
  deleteLearning: (id: string) => Promise<void>
  openModal: (type: 'rule') => void
}

export default function ImprovementsSection({
  sessionId,
  improvementVisible,
  canSeeSection,
  domainList,
  updateRule,
  deleteRule,
  deleteLearning,
  openModal,
}: ImprovementsSectionProps) {
  // Hooks
  const { learnings, rules, loading: learningsLoading } = useLearnings()
  const { domains: testableDomains } = useTestableDomains()
  const regressionQuestionCount = testableDomains.reduce((n: number, d: { question_count: number }) => n + d.question_count, 0)

  // Collapse state
  const [improvementCollapsed, setImprovementCollapsed] = useState(() => localStorage.getItem('constat-improvement-collapsed') === 'true')

  // Tab / editing state
  const [learningsTab, setLearningsTab] = useState<'rules' | 'pending' | 'export' | 'fine-tune'>('rules')
  const [editingRule, setEditingRule] = useState<{ id: string; summary: string } | null>(null)
  const [compacting, setCompacting] = useState(false)
  const [movingRule, setMovingRule] = useState<string | null>(null)

  // Export state
  const [exportFormat, setExportFormat] = useState<'messages' | 'alpaca' | 'sharegpt'>('messages')
  const [exportInclude, setExportInclude] = useState<Set<string>>(new Set(['corrections', 'rules']))
  const [exportMinConfidence, setExportMinConfidence] = useState(0.6)
  const [exporting, setExporting] = useState(false)

  // Fine-tune state
  const [ftJobs, setFtJobs] = useState<FineTuneJob[]>([])
  const [ftProviders, setFtProviders] = useState<FineTuneProvider[]>([])
  const [ftShowForm, setFtShowForm] = useState(false)
  const [ftName, setFtName] = useState('')
  const [ftProvider, setFtProvider] = useState('')
  const [ftBaseModel, setFtBaseModel] = useState('')
  const [ftTaskTypes, setFtTaskTypes] = useState<Set<string>>(new Set(['sql_generation']))
  const [ftDomain, setFtDomain] = useState('')
  const [ftInclude, setFtInclude] = useState<Set<string>>(new Set(['corrections', 'rules']))
  const [ftMinConf, setFtMinConf] = useState(0.6)
  const [ftSubmitting, setFtSubmitting] = useState(false)

  // Auto-refresh fine-tune jobs when any are training
  useEffect(() => {
    const hasTraining = ftJobs.some(j => j.status === 'training')
    if (!hasTraining || learningsTab !== 'fine-tune') return
    const interval = setInterval(() => {
      apolloClient.query({ query: FINE_TUNE_JOBS_QUERY, fetchPolicy: 'network-only' })
        .then((r) => setFtJobs(r.data.fineTuneJobs.map(toFineTuneJob)))
        .catch(() => {})
    }, 30000)
    return () => clearInterval(interval)
  }, [ftJobs, learningsTab])

  // Handlers
  const handleUpdateRule = async () => {
    if (!editingRule || !editingRule.summary.trim()) return
    await updateRule(editingRule.id, editingRule.summary.trim())
    setEditingRule(null)
  }

  const handleDeleteRule = async (ruleId: string) => {
    await deleteRule(ruleId)
  }

  const handleDeleteLearning = async (learningId: string) => {
    await deleteLearning(learningId)
  }

  const handleMoveRule = async (ruleId: string, toDomain: string) => {
    await apolloClient.mutate({ mutation: MOVE_DOMAIN_RULE, variables: { ruleId, toDomain } })
    setMovingRule(null)
    apolloClient.refetchQueries({ include: ['Learnings'] })
  }

  return (
    <>
      {/* --- Improvement sub-group --- */}
      {improvementVisible && (
      <button
        onClick={() => {
          const newVal = !improvementCollapsed
          setImprovementCollapsed(newVal)
          localStorage.setItem('constat-improvement-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Improvement
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${improvementCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {improvementVisible && !improvementCollapsed && (
      <>

      {/* Learnings */}
      {canSeeSection('learnings') && (
      <AccordionSection
        id="learnings"
        title="Learnings"
        count={learnings.length + rules.length}
        icon={<AcademicCapIcon className="w-4 h-4" />}
        command="/learnings"
        action={
          <div className="flex items-center gap-1">
            {learnings.length >= 2 && (
              <button
                onClick={async () => {
                  setCompacting(true)
                  try {
                    const { data } = await apolloClient.mutate({ mutation: COMPACT_LEARNINGS })
                    const result = data.compactLearnings
                    if (result.status === 'success') {
                      apolloClient.refetchQueries({ include: ['Learnings'] })
                    }
                  } finally {
                    setCompacting(false)
                  }
                }}
                disabled={compacting}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                title="Compact learnings into rules"
              >
                <ArrowPathIcon className={`w-4 h-4 ${compacting ? 'animate-spin' : ''}`} />
              </button>
            )}
            <button
              onClick={() => openModal('rule')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add rule"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
        }
      >
        {learnings.length === 0 && rules.length === 0 ? (
          learningsLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No learnings yet</p>
        ) : (
          <div className="space-y-3">
            {/* Tab bar */}
            <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setLearningsTab('rules')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'rules'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Rules ({rules.length})
              </button>
              <button
                onClick={() => setLearningsTab('pending')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'pending'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Pending ({learnings.length})
              </button>
              <button
                onClick={() => setLearningsTab('export')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'export'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Export
              </button>
              <button
                onClick={() => {
                  setLearningsTab('fine-tune')
                  apolloClient.query({ query: FINE_TUNE_JOBS_QUERY, fetchPolicy: 'network-only' })
                    .then((r) => setFtJobs(r.data.fineTuneJobs.map(toFineTuneJob)))
                    .catch(() => {})
                  apolloClient.query({ query: FINE_TUNE_PROVIDERS_QUERY, fetchPolicy: 'network-only' })
                    .then((r) => {
                      const p = r.data.fineTuneProviders.map(toFineTuneProvider)
                      setFtProviders(p)
                      if (p.length > 0 && !ftProvider) {
                        setFtProvider(p[0].name)
                        if (p[0].models.length > 0) setFtBaseModel(p[0].models[0])
                      }
                    }).catch(() => {})
                }}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'fine-tune'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Fine-Tune
              </button>
            </div>

            {/* Rules tab */}
            {learningsTab === 'rules' && rules.length > 0 && (
              <div className="space-y-2">
                {rules.map((rule: { id: string; summary: string; confidence: number; source_count: number; domain?: string; scope?: { level?: string; data_sources?: Array<{ name?: string; type?: string }>; domain?: string }; tags: string[] }) => (
                  <div
                    key={rule.id}
                    className="p-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg group"
                  >
                    {editingRule?.id === rule.id ? (
                      <div className="space-y-2">
                        <textarea
                          value={editingRule.summary || ''}
                          onChange={(e) => setEditingRule({ ...editingRule!, summary: e.target.value })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
                          rows={2}
                          autoFocus
                        />
                        <div className="flex gap-1">
                          <button
                            onClick={handleUpdateRule}
                            className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded"
                            title="Save"
                          >
                            <CheckIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setEditingRule(null)}
                            className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                            title="Cancel"
                          >
                            <XMarkIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-start justify-between gap-2">
                          <p className="text-sm text-gray-700 dark:text-gray-300 flex-1">
                            {rule.summary}
                          </p>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={() => setEditingRule({ id: rule.id, summary: rule.summary || '' })}
                              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                              title="Edit rule"
                            >
                              <PencilIcon className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => setMovingRule(movingRule === rule.id ? null : rule.id)}
                              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                              title="Move to domain"
                            >
                              <ArrowsRightLeftIcon className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => handleDeleteRule(rule.id)}
                              className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                              title="Delete rule"
                            >
                              <TrashIcon className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                        {/* Move-to-domain picker */}
                        {movingRule === rule.id && (
                          <div className="mt-1 flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 rounded px-2 py-1">
                            <span className="text-[11px] text-gray-600 dark:text-gray-400">Move to:</span>
                            <select
                              autoFocus
                              className="text-[11px] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5"
                              defaultValue=""
                              onChange={(e) => { if (e.target.value) handleMoveRule(rule.id, e.target.value) }}
                            >
                              <option value="" disabled>Select domain...</option>
                              {domainList.filter((d) => d.filename !== (rule.domain || '')).map((d) => (
                                <option key={d.filename} value={d.filename}>{d.name}</option>
                              ))}
                            </select>
                            <button onClick={() => setMovingRule(null)} className="text-[11px] text-gray-400 hover:text-gray-600">Cancel</button>
                          </div>
                        )}
                        <div className="mt-1 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
                          <span className="px-1.5 py-0.5 bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 rounded">
                            {Math.round(rule.confidence * 100)}% confidence
                          </span>
                          <span>{rule.source_count} sources</span>
                          <DomainBadge domain={rule.domain || 'user'} />
                          <ScopeBadge scope={rule.scope} />
                          {rule.tags.length > 0 && (
                            <span className="text-gray-300 dark:text-gray-600">
                              {rule.tags.join(', ')}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
            {learningsTab === 'rules' && rules.length === 0 && (
              learningsLoading ? <SkeletonLoader lines={2} /> :
              <p className="text-sm text-gray-500 dark:text-gray-400">No rules yet. Compact pending learnings to generate rules.</p>
            )}

            {/* Pending tab */}
            {learningsTab === 'pending' && learnings.length > 0 && (
              <div className="space-y-2">
                {learnings.map((learning: { id: string; content: string; category: string; scope?: { level?: string; data_sources?: Array<{ name?: string; type?: string }>; domain?: string }; applied_count: number }) => (
                  <div
                    key={learning.id}
                    className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm text-gray-700 dark:text-gray-300 flex-1">
                        {learning.content}
                      </p>
                      <button
                        onClick={() => handleDeleteLearning(learning.id)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Delete learning"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
                      <span className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded">
                        {learning.category}
                      </span>
                      <DomainBadge domain="user" />
                      <ScopeBadge scope={learning.scope} />
                      {learning.applied_count > 0 && (
                        <span>Applied {learning.applied_count}x</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {learningsTab === 'pending' && learnings.length === 0 && (
              <p className="text-sm text-gray-500 dark:text-gray-400">No pending learnings.</p>
            )}

            {/* Export tab */}
            {learningsTab === 'export' && (
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Format</label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as typeof exportFormat)}
                    className="mt-1 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="messages">OpenAI Messages</option>
                    <option value="alpaca">Alpaca</option>
                    <option value="sharegpt">ShareGPT</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Include</label>
                  <div className="mt-1 space-y-1">
                    {(['corrections', 'rules', 'glossary'] as const).map((item) => (
                      <label key={item} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <input
                          type="checkbox"
                          checked={exportInclude.has(item)}
                          onChange={(e) => {
                            const next = new Set(exportInclude)
                            if (e.target.checked) next.add(item); else next.delete(item)
                            setExportInclude(next)
                          }}
                          className="rounded text-primary-600"
                        />
                        {item === 'corrections' ? `Corrections (${learnings.length})` :
                         item === 'rules' ? `Rules (${rules.length})` :
                         'Glossary terms'}
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">
                    Min rule confidence: {Math.round(exportMinConfidence * 100)}%
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={exportMinConfidence * 100}
                    onChange={(e) => setExportMinConfidence(Number(e.target.value) / 100)}
                    className="mt-1 w-full"
                  />
                </div>

                <button
                  onClick={async () => {
                    setExporting(true)
                    try {
                      await downloadSimpleExemplars({
                        format: exportFormat,
                        include: Array.from(exportInclude),
                        min_confidence: exportMinConfidence,
                      })
                    } catch (err) {
                      console.error('Export failed:', err)
                    } finally {
                      setExporting(false)
                    }
                  }}
                  disabled={exporting || exportInclude.size === 0}
                  className="w-full px-3 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  {exporting ? 'Downloading...' : 'Download JSONL'}
                </button>
              </div>
            )}

            {/* Fine-Tune tab */}
            {learningsTab === 'fine-tune' && (
              <div className="space-y-3">
                {/* Job list */}
                {ftJobs.length > 0 && (
                  <div className="space-y-2">
                    {ftJobs.map((job) => (
                      <div
                        key={job.id}
                        className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg group"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                                {job.name}
                              </span>
                              <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${
                                job.status === 'ready' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' :
                                job.status === 'training' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 animate-pulse' :
                                job.status === 'failed' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' :
                                'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-gray-500 dark:text-gray-400">
                              {job.provider}/{job.base_model} · {job.exemplar_count} examples
                              {job.task_types.length > 0 && ` · ${job.task_types.join(', ')}`}
                            </div>
                          </div>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            {job.status === 'training' && (
                              <button
                                onClick={async () => {
                                  await apolloClient.mutate({ mutation: CANCEL_FINE_TUNE_JOB, variables: { modelId: job.id } })
                                  setFtJobs(ftJobs.map(j => j.id === job.id ? { ...j, status: 'failed' as const } : j))
                                }}
                                className="p-1 text-gray-400 hover:text-yellow-600 rounded"
                                title="Cancel"
                              >
                                <XMarkIcon className="w-3 h-3" />
                              </button>
                            )}
                            <button
                              onClick={async () => {
                                await apolloClient.mutate({ mutation: DELETE_FINE_TUNE_JOB, variables: { modelId: job.id } })
                                setFtJobs(ftJobs.filter(j => j.id !== job.id))
                              }}
                              className="p-1 text-gray-400 hover:text-red-500 rounded"
                              title="Delete"
                            >
                              <TrashIcon className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* New job form */}
                {ftShowForm ? (
                  <div className="space-y-2 p-2 border border-gray-200 dark:border-gray-700 rounded-lg">
                    <input
                      type="text"
                      placeholder="Model name (e.g., sales-sql-v1)"
                      value={ftName}
                      onChange={(e) => setFtName(e.target.value)}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Provider</label>
                        <select
                          value={ftProvider}
                          onChange={(e) => {
                            setFtProvider(e.target.value)
                            const prov = ftProviders.find(p => p.name === e.target.value)
                            if (prov && prov.models.length > 0) setFtBaseModel(prov.models[0])
                          }}
                          className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        >
                          {ftProviders.length === 0 && <option value="">No providers (set API keys)</option>}
                          {ftProviders.map(p => (
                            <option key={p.name} value={p.name}>{p.name}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Base Model</label>
                        <select
                          value={ftBaseModel}
                          onChange={(e) => setFtBaseModel(e.target.value)}
                          className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        >
                          {(ftProviders.find(p => p.name === ftProvider)?.models || []).map((m: string) => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Task Types</label>
                      <div className="mt-0.5 flex flex-wrap gap-1">
                        {['sql_generation', 'python_analysis', 'planning', 'summarization'].map(tt => (
                          <label key={tt} className="flex items-center gap-1 text-[11px] text-gray-700 dark:text-gray-300">
                            <input
                              type="checkbox"
                              checked={ftTaskTypes.has(tt)}
                              onChange={(e) => {
                                const next = new Set(ftTaskTypes)
                                if (e.target.checked) next.add(tt); else next.delete(tt)
                                setFtTaskTypes(next)
                              }}
                              className="rounded text-primary-600"
                            />
                            {tt}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Training Data</label>
                      <div className="mt-0.5 flex flex-wrap gap-2">
                        {(['corrections', 'rules', 'glossary'] as const).map(item => (
                          <label key={item} className="flex items-center gap-1 text-[11px] text-gray-700 dark:text-gray-300">
                            <input
                              type="checkbox"
                              checked={ftInclude.has(item)}
                              onChange={(e) => {
                                const next = new Set(ftInclude)
                                if (e.target.checked) next.add(item); else next.delete(item)
                                setFtInclude(next)
                              }}
                              className="rounded text-primary-600"
                            />
                            {item}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Domain (optional)</label>
                      <select
                        value={ftDomain}
                        onChange={(e) => setFtDomain(e.target.value)}
                        className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      >
                        <option value="">All domains (cross-domain)</option>
                        {domainList.map(d => (
                          <option key={d.filename} value={d.filename}>{d.name}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">
                        Min confidence: {Math.round(ftMinConf * 100)}%
                      </label>
                      <input
                        type="range"
                        min={0} max={100} step={5}
                        value={ftMinConf * 100}
                        onChange={(e) => setFtMinConf(Number(e.target.value) / 100)}
                        className="mt-0.5 w-full"
                      />
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={async () => {
                          if (!ftName || !ftProvider || !ftBaseModel || ftTaskTypes.size === 0) return
                          setFtSubmitting(true)
                          try {
                            const ftResult = await apolloClient.mutate({
                              mutation: START_FINE_TUNE_JOB,
                              variables: {
                                input: {
                                  name: ftName,
                                  provider: ftProvider,
                                  baseModel: ftBaseModel,
                                  taskTypes: Array.from(ftTaskTypes),
                                  domain: ftDomain || undefined,
                                  include: Array.from(ftInclude),
                                  minConfidence: ftMinConf,
                                },
                              },
                            })
                            const job = toFineTuneJob(ftResult.data.startFineTuneJob)
                            setFtJobs([job, ...ftJobs])
                            setFtShowForm(false)
                            setFtName('')
                          } catch (err) {
                            console.error('Fine-tune start failed:', err)
                          } finally {
                            setFtSubmitting(false)
                          }
                        }}
                        disabled={ftSubmitting || !ftName || !ftProvider || !ftBaseModel || ftTaskTypes.size === 0}
                        className="flex-1 px-3 py-1.5 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded disabled:opacity-50 transition-colors"
                      >
                        {ftSubmitting ? 'Starting...' : 'Start Training'}
                      </button>
                      <button
                        onClick={() => setFtShowForm(false)}
                        className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => setFtShowForm(true)}
                    className="w-full px-3 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 border border-dashed border-primary-300 dark:border-primary-700 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-md transition-colors flex items-center justify-center gap-2"
                  >
                    <PlusIcon className="w-4 h-4" />
                    New Fine-Tune Job
                  </button>
                )}

                {ftJobs.length === 0 && !ftShowForm && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No fine-tuning jobs yet.</p>
                )}
              </div>
            )}
          </div>
        )}
      </AccordionSection>
      )}

      {/* Regression Tests (in Improvement sub-group) */}
      {sessionId && (
        <AccordionSection
          id="regression"
          title="Regression Tests"
          icon={<BeakerIcon className="w-4 h-4" />}
          count={regressionQuestionCount}
        >
          <RegressionPanel sessionId={sessionId} />
        </AccordionSection>
      )}

      </>
      )}
    </>
  )
}
