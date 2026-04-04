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

// Global indexedDB stub — required because client.ts uses IDBStorage at module scope
if (typeof globalThis.indexedDB === 'undefined') {
  const fakeStore = {
    get: vi.fn(() => ({ onsuccess: null, onerror: null, result: null })),
    put: vi.fn(() => ({ onsuccess: null, onerror: null })),
    delete: vi.fn(() => ({ onsuccess: null, onerror: null })),
  }
  const fakeDB = {
    createObjectStore: vi.fn(),
    close: vi.fn(),
    transaction: vi.fn(() => ({ objectStore: vi.fn(() => fakeStore) })),
  }
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

  vi.stubGlobal('indexedDB', {
    open: vi.fn(() => {
      const req: any = { result: fakeDB }
      return makeAutoResolveProxy(req, 'onsuccess', () => req.onupgradeneeded?.())
    }),
    deleteDatabase: vi.fn(() => makeAutoResolveProxy({}, 'onsuccess')),
  })
}
