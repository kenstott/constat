// Copyright (c) 2025 Kenneth Stott
// Canary: c1fb89f5-490a-4c7a-aa5e-d93ebe4ac213
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import '@testing-library/jest-dom'
import { vi } from 'vitest'

// Mock CachePersistor globally — the real impl opens IDB at module scope, which hangs in the full suite
vi.mock('apollo3-cache-persist', () => ({
  CachePersistor: class {
    persist() { return Promise.resolve() }
    purge() { return Promise.resolve() }
    restore() { return Promise.resolve() }
  },
}))

// Mock graphql-ws to prevent WebSocket connection attempts in test environment
vi.mock('graphql-ws', () => ({
  createClient: vi.fn(() => ({
    subscribe: vi.fn(),
    dispose: vi.fn(),
    on: vi.fn(),
    terminate: vi.fn(),
  })),
}))

// Global indexedDB stub — required because client.ts uses IDBStorage at module scope
if (typeof globalThis.indexedDB === 'undefined') {
  const makeAutoResolveProxy = (target: any, triggerProp: string, preCallback?: () => void) =>
    new Proxy(target, {
      set(t, prop, value) {
        t[prop] = value
        if (prop === triggerProp && value) {
          preCallback?.()
          value()
        }
        return true
      },
      get(t, prop) { return t[prop] },
    })

  const makeFakeRequest = (result: any = null) => {
    const req: any = { result }
    // Auto-trigger onsuccess when it's set
    return makeAutoResolveProxy(req, 'onsuccess')
  }

  const fakeStore = {
    get: vi.fn(() => makeFakeRequest(null)),
    put: vi.fn(() => makeFakeRequest()),
    delete: vi.fn(() => makeFakeRequest()),
  }
  const makeFakeTransaction = () => {
    // Auto-trigger oncomplete when it's set (IDBStorage.setItem waits for tx.oncomplete)
    const tx: any = { objectStore: vi.fn(() => fakeStore) }
    return makeAutoResolveProxy(tx, 'oncomplete')
  }

  const fakeDB = {
    createObjectStore: vi.fn(),
    close: vi.fn(),
    transaction: vi.fn(() => makeFakeTransaction()),
  }

  vi.stubGlobal('indexedDB', {
    open: vi.fn(() => {
      const req: any = { result: fakeDB }
      return makeAutoResolveProxy(req, 'onsuccess', () => req.onupgradeneeded?.())
    }),
    deleteDatabase: vi.fn(() => makeAutoResolveProxy({}, 'onsuccess')),
  })
}
