// Copyright (c) 2025 Kenneth Stott
// Canary: 2d456a1a-7f6a-44d5-b321-ba7d5f090922
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'

// vi.mock is hoisted — factory must be self-contained
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

describe('Apollo client', () => {
  it('exists and is an ApolloClient instance', { timeout: 30000 }, async () => {
    const { ApolloClient } = await import('@apollo/client')
    const { apolloClient } = await import('@/graphql/client')
    expect(apolloClient).toBeInstanceOf(ApolloClient)
  })

  it('cache is defined with type policies', async () => {
    const { cache } = await import('@/graphql/client')
    expect(cache).toBeDefined()
    expect((cache as any).policies).toBeDefined()
  })
})

describe('UI state reactive variables', () => {
  it('showArtifactPanelVar defaults to true', async () => {
    const { showArtifactPanelVar } = await import('@/graphql/ui-state')
    expect(showArtifactPanelVar()).toBe(true)
  })

  it('showProofPanelVar defaults to false', async () => {
    const { showProofPanelVar } = await import('@/graphql/ui-state')
    expect(showProofPanelVar()).toBe(false)
  })

  it('showGlossaryPanelVar defaults to false', async () => {
    const { showGlossaryPanelVar } = await import('@/graphql/ui-state')
    expect(showGlossaryPanelVar()).toBe(false)
  })

  it('reactive vars can be updated', async () => {
    const { showArtifactPanelVar } = await import('@/graphql/ui-state')
    showArtifactPanelVar(false)
    expect(showArtifactPanelVar()).toBe(false)
    showArtifactPanelVar(true)
  })
})
