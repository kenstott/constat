import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock dependencies
vi.mock('@/api/sessions', () => ({
  listArtifacts: vi.fn(),
  listTables: vi.fn(),
  listFacts: vi.fn(),
  listEntities: vi.fn(),
  listLearnings: vi.fn(),
  listStepCodes: vi.fn(),
  listInferenceCodes: vi.fn(),
  getScratchpad: vi.fn(),
  getDDL: vi.fn(),
  listDatabases: vi.fn(),
  listDataSources: vi.fn(),
  getPromptContext: vi.fn(),
  getSessionRouting: vi.fn(),
  getArtifact: vi.fn(),
  persistFact: vi.fn(),
  forgetFact: vi.fn(),
  deleteArtifact: vi.fn(),
  deleteTable: vi.fn(),
  toggleArtifactStar: vi.fn(),
  toggleTableStar: vi.fn(),
  addRule: vi.fn(),
  updateRule: vi.fn(),
  deleteRule: vi.fn(),
  deleteLearning: vi.fn(),
  getMyPermissions: vi.fn(),
  updateSystemPrompt: vi.fn(),
}))

vi.mock('@/api/skills', () => ({
  listSkills: vi.fn(),
  createSkill: vi.fn(),
  updateSkillContent: vi.fn(),
  deleteSkill: vi.fn(),
  setActiveSkills: vi.fn(),
  draftSkill: vi.fn(),
}))

vi.mock('@/store/authStore', () => ({
  isAuthDisabled: true,
  useAuthStore: {
    getState: () => ({
      getToken: vi.fn().mockResolvedValue(null),
      logout: vi.fn(),
    }),
  },
}))

import { useArtifactStore } from '../artifactStore'
import * as sessionsApi from '@/api/sessions'

describe('useArtifactStore', () => {
  beforeEach(() => {
    useArtifactStore.getState().clear()
    vi.clearAllMocks()
  })

  describe('initial state', () => {
    it('has empty arrays and null selections', () => {
      const state = useArtifactStore.getState()
      expect(state.artifacts).toEqual([])
      expect(state.tables).toEqual([])
      expect(state.facts).toEqual([])
      expect(state.entities).toEqual([])
      expect(state.learnings).toEqual([])
      expect(state.rules).toEqual([])
      expect(state.databases).toEqual([])
      expect(state.apis).toEqual([])
      expect(state.documents).toEqual([])
      expect(state.stepCodes).toEqual([])
      expect(state.inferenceCodes).toEqual([])
      expect(state.scratchpadEntries).toEqual([])
      expect(state.sessionDDL).toBe('')
      expect(state.promptContext).toBeNull()
      expect(state.selectedArtifact).toBeNull()
      expect(state.selectedTable).toBeNull()
    })

    it('has correct default loading states', () => {
      const state = useArtifactStore.getState()
      expect(state.loading).toBe(false)
      expect(state.sourcesLoading).toBe(true)
      expect(state.factsLoading).toBe(true)
      expect(state.learningsLoading).toBe(true)
      expect(state.configLoading).toBe(true)
      expect(state.error).toBeNull()
    })

    it('has default user permissions', () => {
      const state = useArtifactStore.getState()
      expect(state.userPermissions).toEqual({
        isAdmin: false,
        persona: 'viewer',
        visibility: {},
        writes: {},
      })
    })
  })

  describe('fetchArtifacts', () => {
    it('sets loading and populates artifacts on success', async () => {
      const mockArtifacts = [{ id: 1, name: 'chart1', artifact_type: 'chart' }]
      vi.mocked(sessionsApi.listArtifacts).mockResolvedValue({ artifacts: mockArtifacts })

      await useArtifactStore.getState().fetchArtifacts('sess-1')

      const state = useArtifactStore.getState()
      expect(state.artifacts).toEqual(mockArtifacts)
      expect(state.loading).toBe(false)
      expect(state.error).toBeNull()
    })

    it('sets error on failure', async () => {
      vi.mocked(sessionsApi.listArtifacts).mockRejectedValue(new Error('network fail'))

      await useArtifactStore.getState().fetchArtifacts('sess-1')

      const state = useArtifactStore.getState()
      expect(state.loading).toBe(false)
      expect(state.error).toContain('network fail')
    })
  })

  describe('fetchTables', () => {
    it('populates tables on success', async () => {
      const mockTables = [{ name: 'sales', row_count: 100 }]
      vi.mocked(sessionsApi.listTables).mockResolvedValue({ tables: mockTables })

      await useArtifactStore.getState().fetchTables('sess-1')

      expect(useArtifactStore.getState().tables).toEqual(mockTables)
    })
  })

  describe('fetchFacts', () => {
    it('sets factsLoading and populates facts', async () => {
      const mockFacts = [{ name: 'total_sales', value: '1000' }]
      vi.mocked(sessionsApi.listFacts).mockResolvedValue({ facts: mockFacts })

      await useArtifactStore.getState().fetchFacts('sess-1')

      const state = useArtifactStore.getState()
      expect(state.facts).toEqual(mockFacts)
      expect(state.factsLoading).toBe(false)
    })
  })

  describe('clear', () => {
    it('resets all state to defaults', async () => {
      // First populate some data
      vi.mocked(sessionsApi.listArtifacts).mockResolvedValue({
        artifacts: [{ id: 1, name: 'x', artifact_type: 'chart' }],
      })
      await useArtifactStore.getState().fetchArtifacts('sess-1')
      expect(useArtifactStore.getState().artifacts).toHaveLength(1)

      // Clear
      useArtifactStore.getState().clear()

      const state = useArtifactStore.getState()
      expect(state.artifacts).toEqual([])
      expect(state.tables).toEqual([])
      expect(state.facts).toEqual([])
      expect(state.selectedArtifact).toBeNull()
      expect(state.error).toBeNull()
      expect(state.sourcesLoading).toBe(true)
      expect(state.factsLoading).toBe(true)
    })
  })

  describe('clearQueryResults', () => {
    it('clears query results but keeps data sources', async () => {
      // Populate artifacts
      vi.mocked(sessionsApi.listArtifacts).mockResolvedValue({
        artifacts: [{ id: 1, name: 'x', artifact_type: 'chart' }],
      })
      await useArtifactStore.getState().fetchArtifacts('sess-1')

      // Manually set some entities (data source context)
      useArtifactStore.setState({ entities: [{ name: 'Customer', type: 'entity' }] as never[] })

      useArtifactStore.getState().clearQueryResults()

      const state = useArtifactStore.getState()
      expect(state.artifacts).toEqual([])
      expect(state.tables).toEqual([])
      expect(state.stepCodes).toEqual([])
      // entities should be preserved
      expect(state.entities).toHaveLength(1)
    })
  })

  describe('selectTable', () => {
    it('sets selectedTable', () => {
      useArtifactStore.getState().selectTable('sales')
      expect(useArtifactStore.getState().selectedTable).toBe('sales')
    })

    it('clears selectedTable with null', () => {
      useArtifactStore.getState().selectTable('sales')
      useArtifactStore.getState().selectTable(null)
      expect(useArtifactStore.getState().selectedTable).toBeNull()
    })
  })

  describe('addTable', () => {
    it('adds a new table', () => {
      useArtifactStore.getState().addTable({ name: 'orders', row_count: 50 } as never)
      expect(useArtifactStore.getState().tables).toHaveLength(1)
      expect(useArtifactStore.getState().tables[0].name).toBe('orders')
    })

    it('updates existing table by name', () => {
      useArtifactStore.getState().addTable({ name: 'orders', row_count: 50 } as never)
      useArtifactStore.getState().addTable({ name: 'orders', row_count: 100 } as never)
      expect(useArtifactStore.getState().tables).toHaveLength(1)
      expect((useArtifactStore.getState().tables[0] as { row_count: number }).row_count).toBe(100)
    })
  })

  describe('addArtifact', () => {
    it('adds a new artifact', () => {
      useArtifactStore.getState().addArtifact({ id: 1, name: 'chart', artifact_type: 'chart' } as never)
      expect(useArtifactStore.getState().artifacts).toHaveLength(1)
    })

    it('updates existing artifact by id', () => {
      useArtifactStore.getState().addArtifact({ id: 1, name: 'chart-v1', artifact_type: 'chart' } as never)
      useArtifactStore.getState().addArtifact({ id: 1, name: 'chart-v2', artifact_type: 'chart' } as never)
      expect(useArtifactStore.getState().artifacts).toHaveLength(1)
      expect(useArtifactStore.getState().artifacts[0].name).toBe('chart-v2')
    })
  })

  describe('addStepCode', () => {
    it('adds step codes in sorted order', () => {
      useArtifactStore.getState().addStepCode(3, 'step 3', 'code3')
      useArtifactStore.getState().addStepCode(1, 'step 1', 'code1')
      useArtifactStore.getState().addStepCode(2, 'step 2', 'code2')
      const codes = useArtifactStore.getState().stepCodes
      expect(codes.map((c) => c.step_number)).toEqual([1, 2, 3])
    })

    it('updates existing step code', () => {
      useArtifactStore.getState().addStepCode(1, 'goal-v1', 'code-v1')
      useArtifactStore.getState().addStepCode(1, 'goal-v2', 'code-v2')
      expect(useArtifactStore.getState().stepCodes).toHaveLength(1)
      expect(useArtifactStore.getState().stepCodes[0].goal).toBe('goal-v2')
    })
  })

  describe('truncateFromStep', () => {
    it('removes items from step N onwards', () => {
      useArtifactStore.getState().addStepCode(1, 'g1', 'c1')
      useArtifactStore.getState().addStepCode(2, 'g2', 'c2')
      useArtifactStore.getState().addStepCode(3, 'g3', 'c3')

      useArtifactStore.getState().truncateFromStep(2)

      expect(useArtifactStore.getState().stepCodes).toHaveLength(1)
      expect(useArtifactStore.getState().stepCodes[0].step_number).toBe(1)
    })
  })

  describe('markStepsSuperseded', () => {
    it('marks current step numbers as superseded', () => {
      useArtifactStore.getState().addStepCode(1, 'g1', 'c1')
      useArtifactStore.getState().addStepCode(2, 'g2', 'c2')

      useArtifactStore.getState().markStepsSuperseded()

      const superseded = useArtifactStore.getState().supersededStepNumbers
      expect(superseded.has(1)).toBe(true)
      expect(superseded.has(2)).toBe(true)
    })
  })

  describe('addInferenceCode', () => {
    it('adds inference code', () => {
      useArtifactStore.getState().addInferenceCode({
        inference_id: 'i1',
        name: 'inf1',
        operation: 'compute',
        code: 'SELECT 1',
        attempt: 1,
      })
      expect(useArtifactStore.getState().inferenceCodes).toHaveLength(1)
    })

    it('replaces inference code with same id', () => {
      useArtifactStore.getState().addInferenceCode({
        inference_id: 'i1',
        name: 'inf1',
        operation: 'compute',
        code: 'SELECT 1',
        attempt: 1,
      })
      useArtifactStore.getState().addInferenceCode({
        inference_id: 'i1',
        name: 'inf1',
        operation: 'compute',
        code: 'SELECT 2',
        attempt: 2,
      })
      expect(useArtifactStore.getState().inferenceCodes).toHaveLength(1)
      expect(useArtifactStore.getState().inferenceCodes[0].code).toBe('SELECT 2')
    })
  })

  describe('clearInferenceCodes', () => {
    it('clears all inference codes', () => {
      useArtifactStore.getState().addInferenceCode({
        inference_id: 'i1',
        name: 'inf1',
        operation: 'compute',
        code: 'SELECT 1',
        attempt: 1,
      })
      useArtifactStore.getState().clearInferenceCodes()
      expect(useArtifactStore.getState().inferenceCodes).toEqual([])
    })
  })
})
