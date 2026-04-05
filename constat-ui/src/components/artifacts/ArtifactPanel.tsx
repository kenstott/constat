// Copyright (c) 2025 Kenneth Stott
// Canary: 8d6bb15b-26f6-419a-9871-1f70a0ec2f1e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Artifact Panel container — thin orchestrator

import { useEffect, useState, type ReactNode } from 'react'
import {
  ChevronRightIcon,
  EnvelopeIcon,
} from '@heroicons/react/24/outline'
import { useSessionContext } from '@/contexts/SessionContext'
import { useArtifactContext } from '@/contexts/ArtifactContext'
import { useAuth } from '@/contexts/AuthContext'
import { useReactiveVar } from '@apollo/client'
import { activeDeepLinkVar, consumeDeepLink, expandedSectionsVar, toggleSection, expandSection } from '@/graphql/ui-state'
import { useGlossaryData } from '@/hooks/useGlossaryData'
import { useSkills } from '@/hooks/useLearnings'
import { useArtifacts } from '@/hooks/useArtifacts'
import GlossaryPanel from './GlossaryPanel'
import type { ObjectivesEntry, DomainTreeNode } from '@/types/api'
import { apolloClient } from '@/graphql/client'
import {
  OBJECTIVES_QUERY,
  toObjectivesEntry,
} from '@/graphql/operations/state'
import {
  ADD_DATABASE, ADD_API,
  UPLOAD_DOCUMENTS, ADD_DOCUMENT_URI, ADD_FILE_REF, ADD_EMAIL_SOURCE,
} from '@/graphql/operations/sources'
import { ADD_FACT } from '@/graphql/operations/data'
import {
  DOMAIN_TREE_QUERY,
  toDomainTreeNode,
} from '@/graphql/operations/domains'
import { EMAIL_OAUTH_PROVIDERS_QUERY } from '@/graphql/operations/auth'
import {
  ResultsSection,
  SourcesSection,
  ReasoningSection,
  ObjectivesSection,
  ImprovementsSection,
  CodeLogSection,
} from './sections'
import { PersonalResourcePicker } from './PersonalResourcePicker'
import { AccountManager } from './AccountManager'
import { AddDatabaseModal } from './AddDatabaseModal'

type ModalType = 'database' | 'api' | 'document' | 'email' | 'fact' | 'rule' | 'personal' | 'accounts' | null

function TopLevelSection({ id, title, count, loading, children }: {
  id: string; title: string; count?: number; loading?: boolean; children: ReactNode
}) {
  const expanded = useReactiveVar(expandedSectionsVar)
  const isOpen = expanded.includes(id)
  return (
    <>
      <button
        onClick={() => toggleSection(id)}
        className="w-full px-4 py-2.5 bg-gray-50 dark:bg-gray-800/80 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors"
      >
        <span className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wider flex items-center gap-1.5">
          {title}
          {count !== undefined && <span className="text-gray-400 dark:text-gray-500 font-normal">({count})</span>}
          {loading && <span className="inline-block w-2.5 h-2.5 border-[1.5px] border-gray-400 border-t-transparent rounded-full animate-spin" />}
        </span>
        <ChevronRightIcon className={`w-3.5 h-3.5 text-gray-400 transition-transform ${isOpen ? 'rotate-90' : ''}`} />
      </button>
      {isOpen && children}
    </>
  )
}

export function ArtifactPanel() {
  const { session, messages, sessionReady } = useSessionContext()
  const pendingDeepLink = useReactiveVar(activeDeepLinkVar)
  const {
    stepCodes,
    inferenceCodes,
    scratchpadEntries,
    sessionDDL,
    supersededStepNumbers,
    promptContext,
    taskRouting,
    allAgents,
    fetchPromptContext,
    fetchTaskRouting,
    fetchAllAgents,
    fetchScratchpad,
    fetchDDL,
    createSkill,
    updateSkill,
    deleteSkill,
    draftSkill,
    updateSystemPrompt,
    addRule,
    updateRule,
    deleteRule,
    deleteLearning,
    ingestingSource,
    ingestProgress,
  } = useArtifactContext()
  const { loading: configLoading } = useSkills()
  const { artifacts } = useArtifacts()
  const { loading: glossaryLoading } = useGlossaryData(session?.session_id || '')
  const { canSee: canSeeSection, canWrite } = useAuth()

  // Visibility calculations
  const sourcesVisible = canSeeSection('databases') || canSeeSection('apis') || canSeeSection('documents') || canSeeSection('facts')
  const hasRouting = taskRouting && Object.keys(taskRouting).length > 0
  const reasoningVisible = canSeeSection('system_prompt') || hasRouting || canSeeSection('agents') || canSeeSection('skills') || canSeeSection('learnings') || canSeeSection('code') || canSeeSection('inference_code')
  const configVisible = canSeeSection('system_prompt') || hasRouting || canSeeSection('agents') || canSeeSection('skills')
  const improvementVisible = canSeeSection('learnings')

  // Modal state
  const [showModal, setShowModal] = useState<ModalType>(null)
  const [modalInput, setModalInput] = useState({ name: '', value: '', uri: '', type: '', persist: false })
  // Document modal state
  const [docSourceType, setDocSourceType] = useState<'uri' | 'files'>('files')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [docFollowLinks, setDocFollowLinks] = useState(false)
  const [docMaxDepth, setDocMaxDepth] = useState(2)
  const [docMaxDocuments, setDocMaxDocuments] = useState(20)
  const [docSameDomainOnly, setDocSameDomainOnly] = useState(true)
  const [docContentType, setDocContentType] = useState('auto')
  const [docShowAdvanced, setDocShowAdvanced] = useState(false)
  // Email modal state
  const [emailProvider, setEmailProvider] = useState<'google' | 'microsoft' | 'other' | null>(null)
  const [emailHost, setEmailHost] = useState('')
  const [emailUsername, setEmailUsername] = useState('')
  const [emailPassword, setEmailPassword] = useState('')
  const [emailAuthType, setEmailAuthType] = useState<'basic' | 'oauth2'>('basic')
  const [emailMailbox, setEmailMailbox] = useState('INBOX')
  const [emailSince, setEmailSince] = useState('')
  const [emailMaxMessages, setEmailMaxMessages] = useState(500)
  const [emailIncludeHeaders, setEmailIncludeHeaders] = useState(true)
  const [emailExtractAttachments, setEmailExtractAttachments] = useState(true)
  const [emailOAuth2ClientId, setEmailOAuth2ClientId] = useState('')
  const [emailOAuth2ClientSecret, setEmailOAuth2ClientSecret] = useState('')
  const [emailOAuth2TenantId, setEmailOAuth2TenantId] = useState('')
  const [emailOAuthProviders, setEmailOAuthProviders] = useState<{ google: boolean; microsoft: boolean } | null>(null)
  const [emailOAuthToken, setEmailOAuthToken] = useState<{ provider: string; email: string; refresh_token: string } | null>(null)
  const [emailShowAdvanced, setEmailShowAdvanced] = useState(false)

  // Objectives state
  const [objectives, setObjectives] = useState<ObjectivesEntry[]>([])

  // Domain list state
  const [domainList, setDomainList] = useState<{ filename: string; name: string }[]>([])
  const [movingFact, setMovingFact] = useState<string | null>(null)

  // Deep link handling
  useEffect(() => {
    if (!pendingDeepLink || !session) return
    const link = pendingDeepLink

    // Source deep links (table/api/document): expand Context group, let SourcesSection consume
    if (link.type === 'table' || link.type === 'document' || link.type === 'api') {
      expandSection('context')
      // Don't consume — SourcesSection will handle expand + scroll + consume
      return
    }

    // Non-source deep links: consume here
    consumeDeepLink()

    if (link.type === 'glossary_term') {
      expandSection('context')
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const el = document.getElementById(`glossary-term-${link.termName}`)
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          } else {
            document.getElementById('section-glossary')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
          }
        })
      })
    }
  }, [pendingDeepLink, session])

  // Fetch data when session is ready
  useEffect(() => {
    if (session && sessionReady) {
      const sid = session.session_id
      fetchAllAgents(sid)
      apolloClient.query({ query: DOMAIN_TREE_QUERY, fetchPolicy: 'network-only' }).then(({ data }) => {
        const nodes = data.domainTree.map(toDomainTreeNode)
        const collect = (ns: DomainTreeNode[]): { filename: string; name: string }[] =>
          ns.flatMap((n) => [{ filename: n.filename, name: n.name }, ...collect(n.children)])
        setDomainList(collect(nodes))
      }).catch(() => {})
      const timer = setTimeout(() => {
        fetchPromptContext(sid)
        fetchTaskRouting(sid)
        fetchScratchpad(sid)
        fetchDDL(sid)
        apolloClient.query({ query: OBJECTIVES_QUERY, variables: { sessionId: sid }, fetchPolicy: 'network-only' }).then(({ data }) => setObjectives(data.objectives.map(toObjectivesEntry))).catch(() => {})
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [session, sessionReady, fetchAllAgents, fetchPromptContext, fetchTaskRouting, fetchScratchpad, fetchDDL])

  // Fetch OAuth providers when email modal opens
  useEffect(() => {
    if (showModal === 'email') {
      apolloClient.query({ query: EMAIL_OAUTH_PROVIDERS_QUERY })
        .then((r) => setEmailOAuthProviders(r.data.emailOAuthProviders))
        .catch(() => setEmailOAuthProviders(null))
    }
  }, [showModal])

  // Listen for OAuth email popup completion
  useEffect(() => {
    const handleOAuthMessage = (event: MessageEvent) => {
      if (event.data?.type === 'oauth-email-complete') {
        const { provider, email, refresh_token } = event.data
        setEmailOAuthToken({ provider, email, refresh_token })
        if (email) setEmailUsername(email)
        if (provider === 'google') setEmailHost('imap.gmail.com')
        else if (provider === 'microsoft') setEmailHost('outlook.office365.com')
      }
    }
    window.addEventListener('message', handleOAuthMessage)
    return () => window.removeEventListener('message', handleOAuthMessage)
  }, [])

  // Modal handlers
  const openModal = (type: ModalType) => {
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
    setDocSourceType('files')
    setSelectedFiles([])
    setShowModal(type)
  }

  const handleAddFact = async () => {
    if (!session || !modalInput.name || !modalInput.value) return
    await apolloClient.mutate({ mutation: ADD_FACT, variables: { sessionId: session.session_id, name: modalInput.name, value: modalInput.value, persist: modalInput.persist } })
    apolloClient.refetchQueries({ include: ['Facts'] })
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddApi = async () => {
    if (!session || !modalInput.name || !modalInput.uri) return
    await apolloClient.mutate({ mutation: ADD_API, variables: { sessionId: session.session_id, input: { name: modalInput.name, baseUrl: modalInput.uri, type: modalInput.type || 'rest' } } })
    apolloClient.refetchQueries({ include: ['DataSources'] })
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddDocument = async () => {
    if (!session) return

    if (docSourceType === 'files') {
      if (selectedFiles.length === 0) return
      setUploading(true)
      try {
        await apolloClient.mutate({ mutation: UPLOAD_DOCUMENTS, variables: { sessionId: session.session_id, files: selectedFiles } })
        apolloClient.refetchQueries({ include: ['DataSources'] })
        setShowModal(null)
        setSelectedFiles([])
        setDocSourceType('files')
      } finally {
        setUploading(false)
      }
    } else {
      if (!modalInput.name || !modalInput.uri) return
      setUploading(true)
      try {
        if (modalInput.uri.startsWith('http://') || modalInput.uri.startsWith('https://')) {
          await apolloClient.mutate({ mutation: ADD_DOCUMENT_URI, variables: { sessionId: session.session_id, input: { name: modalInput.name, url: modalInput.uri, followLinks: docFollowLinks, maxDepth: docMaxDepth, maxDocuments: docMaxDocuments, sameDomainOnly: docSameDomainOnly, type: docContentType } } })
        } else {
          await apolloClient.mutate({ mutation: ADD_FILE_REF, variables: { sessionId: session.session_id, input: { name: modalInput.name, uri: modalInput.uri } } })
        }
        apolloClient.refetchQueries({ include: ['DataSources'] })
        setShowModal(null)
      } catch (err) {
        console.error('Failed to add document:', err)
        alert(`Failed to add document: ${err instanceof Error ? err.message : 'Unknown error'}`)
      } finally {
        setUploading(false)
      }
    }
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddEmail = async () => {
    if (!session || !modalInput.name) return
    const effectiveHost = emailOAuthToken
      ? (emailOAuthToken.provider === 'google' ? 'imap.gmail.com' : 'outlook.office365.com')
      : emailHost
    const effectiveUsername = emailOAuthToken ? emailOAuthToken.email : emailUsername
    if (!effectiveHost || !effectiveUsername) return
    setUploading(true)
    try {
      const scheme = effectiveHost.includes('://') ? '' : 'imaps://'
      await apolloClient.mutate({ mutation: ADD_EMAIL_SOURCE, variables: { sessionId: session.session_id, input: {
        name: modalInput.name,
        url: `${scheme}${effectiveHost}`,
        username: effectiveUsername,
        password: emailProvider === 'other' && emailAuthType === 'basic' ? emailPassword : undefined,
        authType: emailOAuthToken ? 'oauth2_refresh' : emailAuthType,
        mailbox: emailMailbox,
        since: emailSince || undefined,
        maxMessages: emailMaxMessages,
        includeHeaders: emailIncludeHeaders,
        extractAttachments: emailExtractAttachments,
        oauth2ClientId: emailProvider === 'other' && emailAuthType === 'oauth2' ? emailOAuth2ClientId : undefined,
        oauth2ClientSecret: emailProvider === 'other' && emailAuthType === 'oauth2' ? emailOAuth2ClientSecret : undefined,
        oauth2TenantId: emailProvider === 'other' && emailAuthType === 'oauth2' ? emailOAuth2TenantId || undefined : undefined,
        oauth2RefreshToken: emailOAuthToken?.refresh_token || undefined,
      } } })
      apolloClient.refetchQueries({ include: ['DataSources'] })
      setShowModal(null)
      setEmailOAuthToken(null)
      setEmailProvider(null)
    } catch (err) {
      console.error('Failed to add email source:', err)
      alert(`Failed to add email source: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setUploading(false)
    }
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddRule = async () => {
    if (!modalInput.value.trim()) return
    await addRule(modalInput.value.trim())
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

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
      {/* Personal Resource Picker */}
      <PersonalResourcePicker
        isOpen={showModal === 'personal'}
        onClose={() => setShowModal(null)}
        sessionId={session.session_id}
      />

      {/* Account Manager */}
      <AccountManager
        isOpen={showModal === 'accounts'}
        onClose={() => setShowModal(null)}
      />

      {/* Add Modal */}
      {showModal && showModal !== 'personal' && showModal !== 'accounts' && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-80 shadow-xl">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
              Add {showModal === 'fact' ? 'Fact' : showModal === 'database' ? 'Database' : showModal === 'api' ? 'API' : showModal === 'rule' ? 'Rule' : showModal === 'email' ? 'Email Source' : 'Document'}
            </h3>
            <div className="space-y-3">
              {showModal === 'database' ? (
                <AddDatabaseModal
                  onAdd={async (name, uri, type) => {
                    if (!session) return
                    await apolloClient.mutate({ mutation: ADD_DATABASE, variables: { sessionId: session.session_id, input: { name, uri, type } } })
                    apolloClient.refetchQueries({ include: ['DataSources'] })
                    setShowModal(null)
                  }}
                  onCancel={() => setShowModal(null)}
                  uploading={uploading}
                />
              ) : showModal === 'email' ? (
                <div className="space-y-3">
                  {/* Step 1: Choose provider */}
                  {!emailProvider && (
                    <div className="space-y-2">
                      <p className="text-sm text-gray-600 dark:text-gray-400">Choose your email provider:</p>
                      <div className="space-y-2">
                        <button
                            type="button"
                            onClick={() => setEmailProvider('google')}
                            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                          >
                            <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 24 24">
                              <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/>
                              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                              <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                            </svg>
                            Gmail
                          </button>
                        <button
                            type="button"
                            onClick={() => setEmailProvider('microsoft')}
                            className="w-full flex items-center gap-3 px-3 py-2.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                          >
                            <svg className="w-5 h-5 flex-shrink-0" viewBox="0 0 23 23">
                              <path fill="#f35325" d="M1 1h10v10H1z"/>
                              <path fill="#81bc06" d="M12 1h10v10H12z"/>
                              <path fill="#05a6f0" d="M1 12h10v10H1z"/>
                              <path fill="#ffba08" d="M12 12h10v10H12z"/>
                            </svg>
                            Microsoft 365 / Outlook
                          </button>
                        <button
                          type="button"
                          onClick={() => setEmailProvider('other')}
                          className="w-full flex items-center gap-3 px-3 py-2.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                        >
                          <EnvelopeIcon className="w-5 h-5 flex-shrink-0 text-gray-400" />
                          Other (IMAP)
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Step 2: Provider-specific form */}
                  {emailProvider && (
                    <div className="space-y-2">
                      {/* Back button */}
                      <button
                        type="button"
                        onClick={() => { setEmailProvider(null); setEmailOAuthToken(null); setEmailHost(''); setEmailUsername(''); setEmailPassword('') }}
                        className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 flex items-center gap-1"
                      >
                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" /></svg>
                        Back
                      </button>

                      <input
                        type="text"
                        placeholder="Name (e.g., company_inbox)"
                        value={modalInput.name || ''}
                        onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />

                      {/* Google / Microsoft: OAuth sign-in */}
                      {(emailProvider === 'google' || emailProvider === 'microsoft') && (
                        <>
                          {emailOAuthToken ? (
                            <div className="flex items-center gap-2 p-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
                              <svg className="w-4 h-4 text-green-600 dark:text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                              <span className="text-sm text-green-700 dark:text-green-300">
                                {emailOAuthToken.email}
                              </span>
                              <button
                                type="button"
                                onClick={() => setEmailOAuthToken(null)}
                                className="ml-auto text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                              >
                                Change
                              </button>
                            </div>
                          ) : emailOAuthProviders?.[emailProvider] ? (
                            <button
                              type="button"
                              onClick={() => {
                                const url = `/api/oauth/email/authorize?provider=${emailProvider}&session_id=${session?.session_id || ''}`
                                window.open(url, 'oauth-email', 'width=500,height=600,scrollbars=yes')
                              }}
                              className="w-full flex items-center justify-center gap-2 px-3 py-2.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
                            >
                              Sign in with {emailProvider === 'google' ? 'Google' : 'Microsoft'}
                            </button>
                          ) : (
                            <p className="text-sm text-amber-600 dark:text-amber-400">
                              {emailProvider === 'google' ? 'Google' : 'Microsoft'} OAuth not configured on server. Ask your administrator to set {emailProvider === 'google' ? 'GOOGLE_EMAIL_CLIENT_ID' : 'MICROSOFT_EMAIL_CLIENT_ID'}.
                            </p>
                          )}
                        </>
                      )}

                      {/* Manual IMAP fields: shown for "other" or when OAuth is not configured for Google/Microsoft */}
                      {(emailProvider === 'other' || ((emailProvider === 'google' || emailProvider === 'microsoft') && !emailOAuthProviders?.[emailProvider] && !emailOAuthToken)) && (
                        <>
                          <input
                            type="text"
                            placeholder="IMAP Host (e.g., imap.example.com)"
                            value={emailHost}
                            onChange={(e) => setEmailHost(e.target.value)}
                            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          />
                          <input
                            type="text"
                            placeholder="Email / Username"
                            value={emailUsername}
                            onChange={(e) => setEmailUsername(e.target.value)}
                            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                          />
                          <div className="flex gap-4">
                            <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                              <input type="radio" name="emailAuth" checked={emailAuthType === 'basic'} onChange={() => setEmailAuthType('basic')} className="text-primary-600" />
                              Password
                            </label>
                            <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                              <input type="radio" name="emailAuth" checked={emailAuthType === 'oauth2'} onChange={() => setEmailAuthType('oauth2')} className="text-primary-600" />
                              OAuth2
                            </label>
                          </div>
                          {emailAuthType === 'basic' ? (
                            <input
                              type="password"
                              placeholder="Password / App Password"
                              value={emailPassword}
                              onChange={(e) => setEmailPassword(e.target.value)}
                              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                            />
                          ) : (
                            <div className="space-y-2">
                              <input
                                type="text"
                                placeholder="OAuth2 Client ID"
                                value={emailOAuth2ClientId}
                                onChange={(e) => setEmailOAuth2ClientId(e.target.value)}
                                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                              />
                              <input
                                type="password"
                                placeholder="OAuth2 Client Secret / Refresh Token"
                                value={emailOAuth2ClientSecret}
                                onChange={(e) => setEmailOAuth2ClientSecret(e.target.value)}
                                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                              />
                              <input
                                type="text"
                                placeholder="Tenant ID (Azure AD only)"
                                value={emailOAuth2TenantId}
                                onChange={(e) => setEmailOAuth2TenantId(e.target.value)}
                                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                              />
                            </div>
                          )}
                        </>
                      )}

                      {/* Advanced options (shared across all providers) */}
                      <button
                        type="button"
                        onClick={() => setEmailShowAdvanced(!emailShowAdvanced)}
                        className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                      >
                        {emailShowAdvanced ? 'Hide' : 'Show'} advanced options
                      </button>

                      {emailShowAdvanced && (
                        <div className="space-y-2 pl-2 border-l-2 border-gray-200 dark:border-gray-600">
                          <div className="flex items-center gap-2">
                            <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Mailbox</label>
                            <input
                              type="text"
                              value={emailMailbox}
                              onChange={(e) => setEmailMailbox(e.target.value)}
                              className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Since date</label>
                            <input
                              type="date"
                              value={emailSince}
                              onChange={(e) => setEmailSince(e.target.value)}
                              className="flex-1 px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                            />
                          </div>
                          <div className="flex items-center gap-2">
                            <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Max messages</label>
                            <select value={emailMaxMessages} onChange={(e) => setEmailMaxMessages(Number(e.target.value))} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                              {[50, 100, 200, 500, 1000].map(n => <option key={n} value={n}>{n}</option>)}
                            </select>
                          </div>
                          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <input type="checkbox" checked={emailIncludeHeaders} onChange={(e) => setEmailIncludeHeaders(e.target.checked)} className="rounded text-primary-600" />
                            Include email headers
                          </label>
                          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <input type="checkbox" checked={emailExtractAttachments} onChange={(e) => setEmailExtractAttachments(e.target.checked)} className="rounded text-primary-600" />
                            Extract attachments
                          </label>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : showModal === 'rule' ? (
                <textarea
                  placeholder="Enter the rule text..."
                  value={modalInput.value || ''}
                  onChange={(e) => setModalInput({ ...modalInput, value: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
                  rows={3}
                />
              ) : showModal === 'document' ? (
                <>
                  {/* Document source type selector */}
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                      <input
                        type="radio"
                        name="docSourceType"
                        checked={docSourceType === 'uri'}
                        onChange={() => setDocSourceType('uri')}
                        className="text-primary-600"
                      />
                      From URI
                    </label>
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                      <input
                        type="radio"
                        name="docSourceType"
                        checked={docSourceType === 'files'}
                        onChange={() => setDocSourceType('files')}
                        className="text-primary-600"
                      />
                      From Files
                    </label>
                  </div>

                  {docSourceType === 'uri' ? (
                    <div className="space-y-2">
                      <input
                        type="text"
                        placeholder="Name"
                        value={modalInput.name || ''}
                        onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                      <input
                        type="text"
                        placeholder="URI (file:// or http://)"
                        value={modalInput.uri || ''}
                        onChange={(e) => setModalInput({ ...modalInput, uri: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />

                      {/* Crawling options (only for HTTP URLs) */}
                      {(modalInput.uri || '').startsWith('http') && (
                        <>
                          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <input
                              type="checkbox"
                              checked={docFollowLinks}
                              onChange={(e) => setDocFollowLinks(e.target.checked)}
                              className="rounded text-primary-600"
                            />
                            Follow links (crawl linked pages)
                          </label>

                          {docFollowLinks && (
                            <div className="pl-6 space-y-2">
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Max depth</label>
                                <select value={docMaxDepth} onChange={(e) => setDocMaxDepth(Number(e.target.value))} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                                </select>
                              </div>
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Max documents</label>
                                <select value={docMaxDocuments} onChange={(e) => setDocMaxDocuments(Number(e.target.value))} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  {[5, 10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                                </select>
                              </div>
                              <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                                <input
                                  type="checkbox"
                                  checked={docSameDomainOnly}
                                  onChange={(e) => setDocSameDomainOnly(e.target.checked)}
                                  className="rounded text-primary-600"
                                />
                                Same domain only
                              </label>
                            </div>
                          )}

                          <button
                            type="button"
                            onClick={() => setDocShowAdvanced(!docShowAdvanced)}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                          >
                            {docShowAdvanced ? 'Hide' : 'Show'} advanced options
                          </button>

                          {docShowAdvanced && (
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Content type</label>
                                <select value={docContentType} onChange={(e) => setDocContentType(e.target.value)} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  <option value="auto">Auto</option>
                                  <option value="html">HTML</option>
                                  <option value="pdf">PDF</option>
                                  <option value="markdown">Markdown</option>
                                  <option value="text">Text</option>
                                </select>
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  ) : (
                    <>
                      <input
                        type="file"
                        multiple
                        accept=".md,.txt,.pdf,.docx,.html,.htm,.pptx,.xlsx,.csv,.tsv,.parquet,.json"
                        onChange={(e) => setSelectedFiles(Array.from(e.target.files || []))}
                        className="w-full text-sm text-gray-600 dark:text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-sm file:bg-primary-50 file:text-primary-700 dark:file:bg-primary-900/30 dark:file:text-primary-400 hover:file:bg-primary-100 dark:hover:file:bg-primary-900/50 cursor-pointer"
                      />
                      {selectedFiles.length > 0 && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected:
                          <ul className="mt-1 space-y-0.5">
                            {selectedFiles.map((f, i) => (
                              <li key={i} className="truncate">{f.name}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : (
                <>
                  <input
                    type="text"
                    placeholder="Name"
                    value={modalInput.name || ''}
                    onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                  {showModal === 'fact' ? (
                    <>
                      <input
                        type="text"
                        placeholder="Value"
                        value={modalInput.value || ''}
                        onChange={(e) => setModalInput({ ...modalInput, value: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <input
                          type="checkbox"
                          checked={modalInput.persist}
                          onChange={(e) => setModalInput({ ...modalInput, persist: e.target.checked })}
                          className="rounded border-gray-300 dark:border-gray-600"
                        />
                        Save for future sessions
                      </label>
                    </>
                  ) : (
                    <input
                      type="text"
                      placeholder="URI / Path"
                      value={modalInput.uri || ''}
                      onChange={(e) => setModalInput({ ...modalInput, uri: e.target.value })}
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  )}
                </>
              )}
              {showModal === 'api' && (
                <select
                  value={modalInput.type || ''}
                  onChange={(e) => setModalInput({ ...modalInput, type: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <option value="">Type (optional)</option>
                  <option value="rest">REST</option>
                  <option value="graphql">GraphQL</option>
                  <option value="openapi">OpenAPI</option>
                </select>
              )}
            </div>
            {showModal !== 'database' && (
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => { setShowModal(null); setEmailOAuthToken(null); setEmailProvider(null) }}
                className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (showModal === 'fact') handleAddFact()
                  else if (showModal === 'document') handleAddDocument()
                  else if (showModal === 'email') handleAddEmail()
                  else if (showModal === 'rule') handleAddRule()
                  else if (showModal === 'api') handleAddApi()
                }}
                disabled={uploading || (showModal === 'document' && docSourceType === 'files' && selectedFiles.length === 0) || (showModal === 'email' && (!emailProvider || !modalInput.name || (emailProvider === 'other' && (!emailHost || !emailUsername)) || ((emailProvider === 'google' || emailProvider === 'microsoft') && !emailOAuthToken && (emailOAuthProviders?.[emailProvider] || (!emailHost || !emailUsername)))))}
                className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {uploading && (
                  <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                )}
                {uploading ? 'Uploading...' : 'Add'}
              </button>
            </div>
            )}
          </div>
        </div>
      )}

      {/* ═══════════════ ARTIFACTS ═══════════════ */}
      <TopLevelSection id="artifacts" title="Artifacts">
        <ResultsSection
          messages={messages}
          stepCodes={stepCodes}
          supersededStepNumbers={supersededStepNumbers}
        />
      </TopLevelSection>

      {/* ═══════════════ DEBUG ═══════════════ */}
      {reasoningVisible && (
      <TopLevelSection id="debug" title="Debug" count={stepCodes.length + inferenceCodes.length}>
        <CodeLogSection
          stepCodes={stepCodes}
          inferenceCodes={inferenceCodes}
          scratchpadEntries={scratchpadEntries}
          sessionDDL={sessionDDL}
          supersededStepNumbers={supersededStepNumbers}
          onRefreshDDL={() => { if (session) fetchDDL(session.session_id) }}
          session={session}
          canSeeSection={canSeeSection}
          artifacts={artifacts}
        />
      </TopLevelSection>
      )}

      {/* ═══════════════ CONTEXT ═══════════════ */}
      <TopLevelSection id="context" title="Context" loading={glossaryLoading}>
        {sourcesVisible && (
          <SourcesSection
            sessionId={session.session_id}
            sourcesVisible={sourcesVisible}
            canSeeSection={canSeeSection}
            canWrite={canWrite}
            onOpenModal={openModal}
            ingestingSource={ingestingSource}
            ingestProgress={ingestProgress}
            domainList={domainList}
            movingFact={movingFact}
            setMovingFact={setMovingFact}
          />
        )}

        {canSeeSection('glossary') && session && (
        <div id="section-glossary" className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 bg-white dark:bg-gray-800">
          <GlossaryPanel sessionId={session.session_id} />
        </div>
        )}

        {reasoningVisible && (
          <>
            <ReasoningSection
              sessionId={session.session_id}
              reasoningVisible={reasoningVisible}
              configVisible={configVisible}
              canSeeSection={canSeeSection}
              canWrite={canWrite}
              configLoading={configLoading}
              promptContext={promptContext}
              taskRouting={taskRouting}
              allAgents={allAgents}
              domainList={domainList}
              createSkill={createSkill}
              updateSkill={updateSkill}
              deleteSkill={deleteSkill}
              draftSkill={draftSkill}
              updateSystemPrompt={updateSystemPrompt}
              fetchAllAgents={fetchAllAgents}
            />

            <ObjectivesSection objectives={objectives} />

            <ImprovementsSection
              sessionId={session.session_id}
              improvementVisible={improvementVisible}
              canWrite={canWrite('learnings')}
              canSeeSection={canSeeSection}
              domainList={domainList}
              addRule={addRule}
              updateRule={updateRule}
              deleteRule={deleteRule}
              deleteLearning={deleteLearning}
              openModal={(type: 'rule') => openModal(type)}
            />
          </>
        )}
      </TopLevelSection>
    </div>
  )
}
