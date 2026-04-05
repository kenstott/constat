import { describe, it, expect, vi } from 'vitest'

// Mock idb for entityCache (transitive dependency)
vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve({
    get: vi.fn(() => undefined),
    put: vi.fn(),
    delete: vi.fn(),
  })),
}))

// Mock CachePersistor to avoid IDB hanging in full test suite
vi.mock('apollo3-cache-persist', () => ({
  CachePersistor: class {
    persist() { return Promise.resolve() }
    purge() { return Promise.resolve() }
    restore() { return Promise.resolve() }
  },
}))

// Mock graphql-ws to prevent WebSocket connection attempts
vi.mock('graphql-ws', () => ({
  createClient: vi.fn(() => ({
    subscribe: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    terminate: vi.fn(),
  })),
}))

// Mock GraphQLWsLink to prevent WebSocket setup
vi.mock('@apollo/client/link/subscriptions', () => ({
  GraphQLWsLink: class {
    request() { return null }
  },
}))

// indexedDB stub provided by test-setup.ts

describe('Apollo client exports', () => {
  it('apolloClient is an ApolloClient instance', { timeout: 30000 }, async () => {
    const { ApolloClient } = await import('@apollo/client')
    const { apolloClient } = await import('@/graphql/client')
    expect(apolloClient).toBeInstanceOf(ApolloClient)
  })

  it('cache has type policies configured', async () => {
    const { cache } = await import('@/graphql/client')
    const policies = (cache as any).policies
    expect(policies).toBeDefined()
  })

  it('cachePersistor is defined with persist and purge', async () => {
    const { cachePersistor } = await import('@/graphql/client')
    expect(cachePersistor).toBeDefined()
    expect(typeof cachePersistor.persist).toBe('function')
    expect(typeof cachePersistor.purge).toBe('function')
  })
})

describe('cache-policies', () => {
  it('defines keyFields for known types', async () => {
    const { typePolicies } = await import('@/graphql/cache-policies')
    expect(typePolicies.GlossaryTermType?.keyFields).toEqual(['name', 'domain'])
    expect(typePolicies.EntityRelationshipType?.keyFields).toEqual(['id'])
    expect(typePolicies.SessionType?.keyFields).toEqual(['sessionId'])
    expect(typePolicies.TableInfoType?.keyFields).toEqual(['name'])
    expect(typePolicies.ArtifactInfoType?.keyFields).toEqual(['id'])
    expect(typePolicies.FactInfoType?.keyFields).toEqual(['name'])
    expect(typePolicies.DomainInfoType?.keyFields).toEqual(['filename'])
    expect(typePolicies.RuleInfoType?.keyFields).toEqual(['id'])
    expect(typePolicies.LearningInfoType?.keyFields).toEqual(['id'])
  })

  it('Query fields use merge: false', async () => {
    const { typePolicies } = await import('@/graphql/cache-policies')
    const queryFields = (typePolicies.Query as any).fields
    expect(queryFields.tables.merge).toBe(false)
    expect(queryFields.artifacts.merge).toBe(false)
    expect(queryFields.facts.merge).toBe(false)
    expect(queryFields.entities.merge).toBe(false)
    expect(queryFields.sessions.merge).toBe(false)
    expect(queryFields.domains.merge).toBe(false)
    expect(queryFields.learnings.merge).toBe(false)
  })
})
