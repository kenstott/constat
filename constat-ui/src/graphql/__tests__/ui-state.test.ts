import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock localStorage before module import
const localStorageMock: Record<string, string> = {}
vi.stubGlobal('localStorage', {
  getItem: vi.fn((key: string) => localStorageMock[key] ?? null),
  setItem: vi.fn((key: string, value: string) => { localStorageMock[key] = value }),
  removeItem: vi.fn((key: string) => { delete localStorageMock[key] }),
  clear: vi.fn(() => { for (const k of Object.keys(localStorageMock)) delete localStorageMock[k] }),
})

// Dynamic import so each test file gets its own module instance
// We rely on top-level import since makeVar state is mutable across tests
import {
  showArtifactPanelVar,
  showProofPanelVar,
  showGlossaryPanelVar,
  menuOpenVar,
  conversationPanelHiddenVar,
  artifactPanelHiddenVar,
  artifactPanelWidthVar,
  toggleArtifactPanel,
  showArtifactPanel,
  toggleConversationPanel,
  setArtifactPanelWidth,
  expandedSectionsVar,
  toggleSection,
  expandSection,
  expandSections,
  pathToDeepLink,
  activeDeepLinkVar,
  setDeepLink,
  consumeDeepLink,
  fullscreenArtifactVar,
  openFullscreenArtifact,
  closeFullscreenArtifact,
  publicSessionIdVar,
  clearPublicSession,
  themeVar,
  setTheme,
  briefModeVar,
  toggleBriefMode,
  uiModeVar,
  enterReasonChainMode,
  exitReasonChainMode,
  resultsShowPublishedOnlyVar,
  setResultsShowPublishedOnly,
  glossaryGeneratingVar,
  glossaryGenerationStageVar,
  glossaryGenerationPercentVar,
  setGlossaryGenerating,
  setGlossaryProgress,
  glossaryTaxonomySuggestionsVar,
  stepCodesVar,
  inferenceCodesVar,
  scratchpadEntriesVar,
  selectedArtifactVar,
  selectedTableVar,
  supersededStepNumbersVar,
  ingestingSourceVar,
  ingestProgressVar,
  addStepCode,
  addInferenceCode,
  clearInferenceCodes,
  markStepsSuperseded,
  truncateFromStep,
  clearArtifactState,
  sourcesCollapsedVar,
  collapsedResultStepsVar,
  toggleResultStep,
  expandResultStep,
  toastsVar,
  addToast,
  dismissToast,
  proofFactsVar,
  isProvingVar,
  isPlanningCompleteVar,
  isProofPanelOpenVar,
  proofSummaryVar,
  isSummaryGeneratingVar,
  hasCompletedProofVar,
  openProofPanel,
  closeProofPanel,
  toggleProofPanel,
  clearProofFacts,
  exportFacts,
  importFacts,
  handleFactEvent,
  type DeepLink,
  type FactNode,
} from '@/graphql/ui-state'

// ---------- helpers ----------

function resetPanelVars() {
  showArtifactPanelVar(true)
  showProofPanelVar(false)
  showGlossaryPanelVar(false)
  menuOpenVar(false)
  conversationPanelHiddenVar(false)
  artifactPanelHiddenVar(true)
  artifactPanelWidthVar(400)
}

function resetProofVars() {
  proofFactsVar(new Map())
  isProvingVar(false)
  isPlanningCompleteVar(false)
  isProofPanelOpenVar(false)
  proofSummaryVar(null)
  isSummaryGeneratingVar(false)
  hasCompletedProofVar(false)
}

// ---------- Tests ----------

describe('Panel visibility reactive vars — defaults', () => {
  it('showArtifactPanelVar defaults to true', () => {
    expect(showArtifactPanelVar()).toBe(true)
  })

  it('showProofPanelVar defaults to false', () => {
    expect(showProofPanelVar()).toBe(false)
  })

  it('showGlossaryPanelVar defaults to false', () => {
    expect(showGlossaryPanelVar()).toBe(false)
  })

  it('menuOpenVar defaults to false', () => {
    expect(menuOpenVar()).toBe(false)
  })
})

describe('toggleArtifactPanel', () => {
  beforeEach(resetPanelVars)

  it('hides artifact panel when currently shown', () => {
    artifactPanelHiddenVar(false)
    toggleArtifactPanel()
    expect(artifactPanelHiddenVar()).toBe(true)
    // When hiding artifact, conversation should be shown
    expect(conversationPanelHiddenVar()).toBe(false)
  })

  it('shows artifact panel when currently hidden', () => {
    artifactPanelHiddenVar(true)
    toggleArtifactPanel()
    expect(artifactPanelHiddenVar()).toBe(false)
  })
})

describe('showArtifactPanel', () => {
  beforeEach(resetPanelVars)

  it('shows artifact panel when hidden', () => {
    artifactPanelHiddenVar(true)
    showArtifactPanel()
    expect(artifactPanelHiddenVar()).toBe(false)
  })

  it('no-op when already visible', () => {
    artifactPanelHiddenVar(false)
    showArtifactPanel()
    expect(artifactPanelHiddenVar()).toBe(false)
  })
})

describe('toggleConversationPanel', () => {
  beforeEach(resetPanelVars)

  it('hides conversation panel and shows artifact', () => {
    conversationPanelHiddenVar(false)
    toggleConversationPanel()
    expect(conversationPanelHiddenVar()).toBe(true)
    expect(artifactPanelHiddenVar()).toBe(false)
  })

  it('shows conversation panel when hidden', () => {
    conversationPanelHiddenVar(true)
    toggleConversationPanel()
    expect(conversationPanelHiddenVar()).toBe(false)
  })
})

describe('setArtifactPanelWidth', () => {
  it('updates the width var and persists', () => {
    setArtifactPanelWidth(600)
    expect(artifactPanelWidthVar()).toBe(600)
    expect(localStorage.setItem).toHaveBeenCalled()
    // Reset
    artifactPanelWidthVar(400)
  })
})

describe('Expanded sections', () => {
  beforeEach(() => {
    expandedSectionsVar(['artifacts'])
  })

  it('toggleSection adds a section', () => {
    toggleSection('context')
    expect(expandedSectionsVar()).toContain('context')
  })

  it('toggleSection removes an existing section', () => {
    toggleSection('artifacts')
    expect(expandedSectionsVar()).not.toContain('artifacts')
  })

  it('expandSection adds if missing', () => {
    expandSection('databases')
    expect(expandedSectionsVar()).toContain('databases')
  })

  it('expandSection no-op if already present', () => {
    expandSection('artifacts')
    expect(expandedSectionsVar().filter(s => s === 'artifacts').length).toBe(1)
  })

  it('expandSections adds multiple new sections', () => {
    expandSections(['databases', 'documents'])
    expect(expandedSectionsVar()).toContain('databases')
    expect(expandedSectionsVar()).toContain('documents')
  })

  it('expandSections no-op when all already present', () => {
    const before = [...expandedSectionsVar()]
    expandSections(['artifacts'])
    expect(expandedSectionsVar()).toEqual(before)
  })
})

describe('pathToDeepLink', () => {
  it('parses /db/<db>/<table>', () => {
    const link = pathToDeepLink('/db/mydb/users')
    expect(link).toEqual({ type: 'table', dbName: 'mydb', tableName: 'users' })
  })

  it('parses /apis/<name>', () => {
    const link = pathToDeepLink('/apis/my-api')
    expect(link).toEqual({ type: 'api', apiName: 'my-api' })
  })

  it('parses /doc/<name>', () => {
    const link = pathToDeepLink('/doc/readme')
    expect(link).toEqual({ type: 'document', documentName: 'readme' })
  })

  it('parses /glossary/<term>', () => {
    const link = pathToDeepLink('/glossary/revenue')
    expect(link).toEqual({ type: 'glossary_term', termName: 'revenue' })
  })

  it('parses /s/<sessionId>', () => {
    const link = pathToDeepLink('/s/abc-123')
    expect(link).toEqual({ type: 'session', sessionId: 'abc-123' })
  })

  it('returns null for empty path', () => {
    expect(pathToDeepLink('/')).toBeNull()
    expect(pathToDeepLink('')).toBeNull()
  })

  it('returns null for unknown prefix', () => {
    expect(pathToDeepLink('/unknown/path')).toBeNull()
  })

  it('returns null for too-short db path', () => {
    expect(pathToDeepLink('/db/mydb')).toBeNull()
  })

  it('returns null for /apis with no name', () => {
    expect(pathToDeepLink('/apis')).toBeNull()
  })

  it('returns null for /doc with no name', () => {
    expect(pathToDeepLink('/doc')).toBeNull()
  })

  it('returns null for /glossary with no term', () => {
    expect(pathToDeepLink('/glossary')).toBeNull()
  })

  it('returns null for /s with no id', () => {
    expect(pathToDeepLink('/s')).toBeNull()
  })

  it('decodes URI components', () => {
    const link = pathToDeepLink('/doc/my%20document')
    expect(link).toEqual({ type: 'document', documentName: 'my document' })
  })
})

describe('Deep link management', () => {
  beforeEach(() => {
    activeDeepLinkVar(null)
  })

  it('setDeepLink sets value with timestamp', () => {
    const link: DeepLink = { type: 'table', dbName: 'db', tableName: 'tbl' }
    setDeepLink(link)
    const stored = activeDeepLinkVar()
    expect(stored?.type).toBe('table')
    expect(stored?._ts).toBeDefined()
  })

  it('consumeDeepLink returns and clears', () => {
    setDeepLink({ type: 'api', apiName: 'test' })
    const consumed = consumeDeepLink()
    expect(consumed?.type).toBe('api')
    expect(activeDeepLinkVar()).toBeNull()
  })

  it('consumeDeepLink returns null when empty', () => {
    expect(consumeDeepLink()).toBeNull()
  })
})

describe('Fullscreen artifact', () => {
  beforeEach(() => fullscreenArtifactVar(null))

  it('openFullscreenArtifact sets the value', () => {
    openFullscreenArtifact({ type: 'artifact', id: 1, name: 'test' })
    expect(fullscreenArtifactVar()?.id).toBe(1)
  })

  it('closeFullscreenArtifact clears the value', () => {
    openFullscreenArtifact({ type: 'table', name: 'x' })
    closeFullscreenArtifact()
    expect(fullscreenArtifactVar()).toBeNull()
  })
})

describe('Public session', () => {
  it('clearPublicSession sets to null', () => {
    publicSessionIdVar('some-id')
    clearPublicSession()
    expect(publicSessionIdVar()).toBeNull()
  })
})

describe('Theme', () => {
  beforeEach(() => {
    document.documentElement.classList.remove('dark')
    themeVar('system')
  })

  it('setTheme dark adds dark class', () => {
    setTheme('dark')
    expect(themeVar()).toBe('dark')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
  })

  it('setTheme light removes dark class', () => {
    document.documentElement.classList.add('dark')
    setTheme('light')
    expect(themeVar()).toBe('light')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('setTheme system uses matchMedia (light preference)', () => {
    window.matchMedia = vi.fn().mockReturnValue({ matches: false }) as any
    setTheme('system')
    expect(themeVar()).toBe('system')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })

  it('setTheme system uses matchMedia (dark preference)', () => {
    window.matchMedia = vi.fn().mockReturnValue({ matches: true }) as any
    setTheme('system')
    expect(themeVar()).toBe('system')
    expect(document.documentElement.classList.contains('dark')).toBe(true)
    // Clean up
    document.documentElement.classList.remove('dark')
  })
})

describe('Brief mode', () => {
  beforeEach(() => briefModeVar(false))

  it('toggleBriefMode flips value', () => {
    toggleBriefMode()
    expect(briefModeVar()).toBe(true)
    toggleBriefMode()
    expect(briefModeVar()).toBe(false)
  })
})

describe('UI mode', () => {
  beforeEach(() => {
    uiModeVar('exploratory')
    conversationPanelHiddenVar(true)
  })

  it('enterReasonChainMode sets mode and shows conversation', () => {
    enterReasonChainMode()
    expect(uiModeVar()).toBe('reason-chain')
    expect(conversationPanelHiddenVar()).toBe(false)
  })

  it('exitReasonChainMode resets to exploratory', () => {
    enterReasonChainMode()
    exitReasonChainMode()
    expect(uiModeVar()).toBe('exploratory')
  })
})

describe('Results filter', () => {
  it('setResultsShowPublishedOnly updates var and localStorage', () => {
    setResultsShowPublishedOnly(false)
    expect(resultsShowPublishedOnlyVar()).toBe(false)
    expect(localStorage.setItem).toHaveBeenCalledWith('constat-results-filter', 'all')

    setResultsShowPublishedOnly(true)
    expect(resultsShowPublishedOnlyVar()).toBe(true)
    expect(localStorage.setItem).toHaveBeenCalledWith('constat-results-filter', 'published')
  })
})

describe('Glossary generation', () => {
  beforeEach(() => {
    glossaryGeneratingVar(false)
    glossaryGenerationStageVar(null)
    glossaryGenerationPercentVar(0)
  })

  it('setGlossaryGenerating(true) sets var', () => {
    setGlossaryGenerating(true)
    expect(glossaryGeneratingVar()).toBe(true)
  })

  it('setGlossaryGenerating(false) resets stage and percent', () => {
    glossaryGenerationStageVar('analyzing')
    glossaryGenerationPercentVar(50)
    setGlossaryGenerating(false)
    expect(glossaryGeneratingVar()).toBe(false)
    expect(glossaryGenerationStageVar()).toBeNull()
    expect(glossaryGenerationPercentVar()).toBe(0)
  })

  it('setGlossaryProgress updates stage and percent', () => {
    setGlossaryProgress('embedding', 75)
    expect(glossaryGenerationStageVar()).toBe('embedding')
    expect(glossaryGenerationPercentVar()).toBe(75)
  })

  it('glossaryTaxonomySuggestionsVar defaults to empty array', () => {
    expect(glossaryTaxonomySuggestionsVar()).toEqual([])
  })
})

describe('Step codes', () => {
  beforeEach(() => {
    stepCodesVar([])
    scratchpadEntriesVar([])
    supersededStepNumbersVar(new Set())
  })

  it('addStepCode adds a new step', () => {
    addStepCode(1, 'Goal 1', 'SELECT 1')
    expect(stepCodesVar()).toHaveLength(1)
    expect(stepCodesVar()[0]).toEqual({ step_number: 1, goal: 'Goal 1', code: 'SELECT 1', model: undefined })
  })

  it('addStepCode updates existing step', () => {
    addStepCode(1, 'Goal 1', 'SELECT 1')
    addStepCode(1, 'Goal 1 updated', 'SELECT 2', 'gpt-4')
    expect(stepCodesVar()).toHaveLength(1)
    expect(stepCodesVar()[0].code).toBe('SELECT 2')
    expect(stepCodesVar()[0].model).toBe('gpt-4')
  })

  it('addStepCode sorts by step number', () => {
    addStepCode(3, 'Goal 3', 'code3')
    addStepCode(1, 'Goal 1', 'code1')
    addStepCode(2, 'Goal 2', 'code2')
    expect(stepCodesVar().map(s => s.step_number)).toEqual([1, 2, 3])
  })

  it('markStepsSuperseded marks all current steps', () => {
    addStepCode(1, 'g', 'c')
    addStepCode(2, 'g', 'c')
    markStepsSuperseded()
    expect(supersededStepNumbersVar().has(1)).toBe(true)
    expect(supersededStepNumbersVar().has(2)).toBe(true)
  })

  it('truncateFromStep removes steps >= fromStep', () => {
    addStepCode(1, 'g1', 'c1')
    addStepCode(2, 'g2', 'c2')
    addStepCode(3, 'g3', 'c3')
    truncateFromStep(2)
    expect(stepCodesVar()).toHaveLength(1)
    expect(stepCodesVar()[0].step_number).toBe(1)
  })
})

describe('Inference codes', () => {
  beforeEach(() => inferenceCodesVar([]))

  it('addInferenceCode adds new inference code', () => {
    addInferenceCode({ inference_id: 'a', name: 'n', operation: 'op', code: 'c', attempt: 1 })
    expect(inferenceCodesVar()).toHaveLength(1)
  })

  it('addInferenceCode replaces existing by inference_id', () => {
    addInferenceCode({ inference_id: 'a', name: 'n', operation: 'op', code: 'c1', attempt: 1 })
    addInferenceCode({ inference_id: 'a', name: 'n', operation: 'op', code: 'c2', attempt: 2 })
    expect(inferenceCodesVar()).toHaveLength(1)
    expect(inferenceCodesVar()[0].code).toBe('c2')
  })

  it('clearInferenceCodes empties the list', () => {
    addInferenceCode({ inference_id: 'a', name: 'n', operation: 'op', code: 'c', attempt: 1 })
    clearInferenceCodes()
    expect(inferenceCodesVar()).toEqual([])
  })
})

describe('clearArtifactState', () => {
  it('resets all artifact-related vars', () => {
    addStepCode(1, 'g', 'c')
    addInferenceCode({ inference_id: 'x', name: 'n', operation: 'o', code: 'c', attempt: 1 })
    selectedArtifactVar({ content: 'x', is_binary: false } as any)
    selectedTableVar('tbl')
    ingestingSourceVar('file.csv')
    ingestProgressVar({ current: 5, total: 10 })

    clearArtifactState()

    expect(stepCodesVar()).toEqual([])
    expect(inferenceCodesVar()).toEqual([])
    expect(scratchpadEntriesVar()).toEqual([])
    expect(selectedArtifactVar()).toBeNull()
    expect(selectedTableVar()).toBeNull()
    expect(supersededStepNumbersVar().size).toBe(0)
    expect(ingestingSourceVar()).toBeNull()
    expect(ingestProgressVar()).toBeNull()
  })
})

describe('Collapsed result steps', () => {
  beforeEach(() => collapsedResultStepsVar(new Set()))

  it('toggleResultStep adds a step', () => {
    toggleResultStep(1)
    expect(collapsedResultStepsVar().has(1)).toBe(true)
  })

  it('toggleResultStep removes existing step', () => {
    toggleResultStep(1)
    toggleResultStep(1)
    expect(collapsedResultStepsVar().has(1)).toBe(false)
  })

  it('expandResultStep removes collapsed step', () => {
    toggleResultStep(5)
    expandResultStep(5)
    expect(collapsedResultStepsVar().has(5)).toBe(false)
  })

  it('expandResultStep no-op if not collapsed', () => {
    expandResultStep(99)
    expect(collapsedResultStepsVar().has(99)).toBe(false)
  })
})

describe('Toast notifications', () => {
  beforeEach(() => toastsVar([]))

  it('addToast adds a toast with default type', () => {
    addToast('Hello')
    const toasts = toastsVar()
    expect(toasts).toHaveLength(1)
    expect(toasts[0].message).toBe('Hello')
    expect(toasts[0].type).toBe('success')
    expect(toasts[0].id).toBeDefined()
  })

  it('addToast with specific type', () => {
    addToast('Error occurred', 'error')
    expect(toastsVar()[0].type).toBe('error')
  })

  it('dismissToast removes by id', () => {
    addToast('A')
    addToast('B')
    const toasts = toastsVar()
    dismissToast(toasts[0].id)
    expect(toastsVar()).toHaveLength(1)
    expect(toastsVar()[0].message).toBe('B')
  })

  it('auto-removes after timeout', async () => {
    vi.useFakeTimers()
    toastsVar([])
    addToast('Auto remove')
    expect(toastsVar()).toHaveLength(1)
    vi.advanceTimersByTime(3000)
    expect(toastsVar()).toHaveLength(0)
    vi.useRealTimers()
  })
})

describe('Proof panel', () => {
  beforeEach(resetProofVars)

  it('openProofPanel sets true', () => {
    openProofPanel()
    expect(isProofPanelOpenVar()).toBe(true)
  })

  it('closeProofPanel sets false', () => {
    openProofPanel()
    closeProofPanel()
    expect(isProofPanelOpenVar()).toBe(false)
  })

  it('toggleProofPanel toggles value', () => {
    expect(isProofPanelOpenVar()).toBe(false)
    toggleProofPanel()
    expect(isProofPanelOpenVar()).toBe(true)
    toggleProofPanel()
    expect(isProofPanelOpenVar()).toBe(false)
  })
})

describe('clearProofFacts', () => {
  beforeEach(resetProofVars)

  it('resets all proof state', () => {
    proofFactsVar(new Map([['a', { id: 'a', name: 'a', status: 'resolved', dependencies: [] } as FactNode]]))
    isProvingVar(true)
    isPlanningCompleteVar(true)
    proofSummaryVar('summary')
    hasCompletedProofVar(true)

    clearProofFacts()

    expect(proofFactsVar().size).toBe(0)
    expect(isProvingVar()).toBe(false)
    expect(isPlanningCompleteVar()).toBe(false)
    expect(proofSummaryVar()).toBeNull()
    expect(hasCompletedProofVar()).toBe(false)
  })
})

describe('exportFacts / importFacts', () => {
  beforeEach(resetProofVars)

  it('exportFacts returns array from map', () => {
    const fact: FactNode = { id: 'f1', name: 'f1', status: 'resolved', dependencies: [] }
    proofFactsVar(new Map([['f1', fact]]))
    const exported = exportFacts()
    expect(exported).toHaveLength(1)
    expect(exported[0].id).toBe('f1')
  })

  it('importFacts restores facts map and sets states', () => {
    uiModeVar('reason-chain')
    const facts: FactNode[] = [
      { id: 'a', name: 'a', status: 'resolved', dependencies: [] },
      { id: 'b', name: 'b', status: 'pending', dependencies: ['a'] },
    ]
    importFacts(facts, 'test summary')

    expect(proofFactsVar().size).toBe(2)
    expect(isProvingVar()).toBe(false)
    expect(isProofPanelOpenVar()).toBe(true) // reason-chain mode
    expect(isPlanningCompleteVar()).toBe(true)
    expect(proofSummaryVar()).toBe('test summary')
    expect(hasCompletedProofVar()).toBe(true)

    // Clean up
    uiModeVar('exploratory')
  })

  it('importFacts with no summary sets null', () => {
    importFacts([{ id: 'x', name: 'x', status: 'resolved', dependencies: [] }])
    expect(proofSummaryVar()).toBeNull()
  })

  it('importFacts with empty facts sets hasCompletedProof false', () => {
    importFacts([])
    expect(hasCompletedProofVar()).toBe(false)
  })
})

describe('handleFactEvent', () => {
  beforeEach(resetProofVars)

  it('proof_start resets proof state', () => {
    proofFactsVar(new Map([['old', { id: 'old', name: 'old', status: 'resolved', dependencies: [] } as FactNode]]))
    handleFactEvent('proof_start', { fact_name: '' })
    expect(proofFactsVar().size).toBe(0)
    expect(isProvingVar()).toBe(true)
    expect(isPlanningCompleteVar()).toBe(false)
    expect(proofSummaryVar()).toBeNull()
  })

  it('dag_execution_start sets planning complete and enters reason-chain', () => {
    handleFactEvent('dag_execution_start', { fact_name: '' })
    expect(isPlanningCompleteVar()).toBe(true)
    expect(isProofPanelOpenVar()).toBe(true)
    expect(uiModeVar()).toBe('reason-chain')
    // Clean up
    uiModeVar('exploratory')
  })

  it('proof_complete sets isProving false and hasCompleted true', () => {
    isProvingVar(true)
    handleFactEvent('proof_complete', { fact_name: '' })
    expect(isProvingVar()).toBe(false)
    expect(hasCompletedProofVar()).toBe(true)
    expect(isSummaryGeneratingVar()).toBe(true)
    // Clean up
    isSummaryGeneratingVar(false)
  })

  it('proof_summary_ready sets summary', () => {
    isSummaryGeneratingVar(true)
    handleFactEvent('proof_summary_ready', { fact_name: '', summary: 'Final answer is 42' })
    expect(proofSummaryVar()).toBe('Final answer is 42')
    expect(isSummaryGeneratingVar()).toBe(false)
  })

  it('proof_summary_ready with empty summary', () => {
    isSummaryGeneratingVar(true)
    handleFactEvent('proof_summary_ready', { fact_name: '', summary: '' })
    expect(proofSummaryVar()).toBeNull()
    expect(isSummaryGeneratingVar()).toBe(false)
  })

  it('fact_start creates/updates a fact node', () => {
    handleFactEvent('fact_start', {
      fact_name: 'revenue',
      fact_description: 'Total revenue',
      dependencies: ['sales', 'returns'],
    })
    const fact = proofFactsVar().get('revenue')
    expect(fact).toBeDefined()
    expect(fact!.status).toBe('pending')
    expect(fact!.description).toBe('Total revenue')
    expect(fact!.dependencies).toEqual(['sales', 'returns'])
    expect(isProvingVar()).toBe(true)
  })

  it('fact_planning sets status to planning', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('fact_planning', { fact_name: 'rev' })
    expect(proofFactsVar().get('rev')!.status).toBe('planning')
  })

  it('fact_executing sets status and formula', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('fact_executing', { fact_name: 'rev', formula: 'SUM(sales)' })
    const fact = proofFactsVar().get('rev')!
    expect(fact.status).toBe('executing')
    expect(fact.formula).toBe('SUM(sales)')
  })

  it('fact_resolved sets full result data', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('fact_resolved', {
      fact_name: 'rev',
      value: 1000,
      source: 'database',
      confidence: 0.95,
      tier: 1,
      strategy: 'direct',
      dependencies: ['sales'],
      elapsed_ms: 150,
      attempt: 1,
    })
    const fact = proofFactsVar().get('rev')!
    expect(fact.status).toBe('resolved')
    expect(fact.value).toBe(1000)
    expect(fact.source).toBe('database')
    expect(fact.confidence).toBe(0.95)
    expect(fact.tier).toBe(1)
    expect(fact.strategy).toBe('direct')
    expect(fact.elapsed_ms).toBe(150)
  })

  it('fact_failed sets status and reason', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('fact_failed', { fact_name: 'rev', reason: 'timeout' })
    const fact = proofFactsVar().get('rev')!
    expect(fact.status).toBe('failed')
    expect(fact.reason).toBe('timeout')
  })

  it('fact_blocked sets status and reason', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('fact_blocked', { fact_name: 'rev', reason: 'dependency failed' })
    const fact = proofFactsVar().get('rev')!
    expect(fact.status).toBe('blocked')
    expect(fact.reason).toBe('dependency failed')
  })

  it('inference_code updates existing fact code', () => {
    handleFactEvent('fact_start', { fact_name: 'rev', dependencies: [] })
    handleFactEvent('inference_code', { fact_name: 'rev', code: 'SELECT SUM(*)', attempt: 2 })
    const fact = proofFactsVar().get('rev')!
    expect(fact.code).toBe('SELECT SUM(*)')
    expect(fact.attempt).toBe(2)
  })

  it('inference_code ignores if fact not found', () => {
    handleFactEvent('inference_code', { fact_name: 'nonexistent', code: 'x', attempt: 1 })
    expect(proofFactsVar().has('nonexistent')).toBe(false)
  })

  it('inference_code ignores if no fact_name', () => {
    handleFactEvent('inference_code', { fact_name: '', code: 'x', attempt: 1 })
    expect(proofFactsVar().size).toBe(0)
  })

  it('ignores events with no fact_name for standard events', () => {
    handleFactEvent('fact_start', { fact_name: '' })
    expect(proofFactsVar().size).toBe(0)
  })

  it('creates a new fact node if not present for standard events', () => {
    handleFactEvent('fact_resolved', {
      fact_name: 'new_fact',
      value: 42,
    })
    const fact = proofFactsVar().get('new_fact')
    expect(fact).toBeDefined()
    expect(fact!.status).toBe('resolved')
    expect(fact!.value).toBe(42)
  })
})

describe('sourcesCollapsedVar', () => {
  it('defaults based on localStorage', () => {
    // Default localStorage mock returns null, so 'true' comparison is false
    expect(typeof sourcesCollapsedVar()).toBe('boolean')
  })
})

describe('initPreferences', () => {
  it('is exported and callable', async () => {
    const mod = await import('@/graphql/ui-state')
    expect(typeof mod.initPreferences).toBe('function')
  })
})
