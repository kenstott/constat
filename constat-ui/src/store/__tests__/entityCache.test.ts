// Copyright (c) 2025 Kenneth Stott
// Canary: c835e258-287d-4331-9fea-a71cbafb7427
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockDB = {
  get: vi.fn(),
  put: vi.fn(),
  delete: vi.fn(),
}

vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve(mockDB)),
}))

describe('entityCache', () => {
  beforeEach(() => {
    vi.resetModules()
    mockDB.get.mockReset()
    mockDB.put.mockReset()
    mockDB.delete.mockReset()
  })

  describe('getCachedEntry', () => {
    it('returns cached entry when it exists', async () => {
      const entry = { state: { e: {}, g: {}, r: {}, k: {} }, version: 5 }
      mockDB.get.mockResolvedValue(entry)
      const { getCachedEntry } = await import('../entityCache')
      const result = await getCachedEntry('session-1')
      expect(result).toEqual(entry)
      expect(mockDB.get).toHaveBeenCalledWith('sessions', 'session-1')
    })

    it('returns null when no entry exists', async () => {
      mockDB.get.mockResolvedValue(undefined)
      const { getCachedEntry } = await import('../entityCache')
      const result = await getCachedEntry('missing')
      expect(result).toBeNull()
    })
  })

  describe('setCachedEntry', () => {
    it('stores entry with session id as key', async () => {
      mockDB.put.mockResolvedValue(undefined)
      const state = { e: {}, g: {}, r: {}, k: {} } as any
      const { setCachedEntry } = await import('../entityCache')
      await setCachedEntry('session-2', state, 3)
      expect(mockDB.put).toHaveBeenCalledWith('sessions', { state, version: 3 }, 'session-2')
    })
  })

  describe('clearCachedState', () => {
    it('deletes entry by session id', async () => {
      mockDB.delete.mockResolvedValue(undefined)
      const { clearCachedState } = await import('../entityCache')
      await clearCachedState('session-3')
      expect(mockDB.delete).toHaveBeenCalledWith('sessions', 'session-3')
    })
  })

  describe('getDB singleton', () => {
    it('reuses the same db promise on multiple calls', async () => {
      const { openDB } = await import('idb')
      const callsBefore = (openDB as any).mock.calls.length
      const { getCachedEntry } = await import('../entityCache')
      mockDB.get.mockResolvedValue(undefined)
      await getCachedEntry('a')
      await getCachedEntry('b')
      // openDB should have been called exactly once for this import
      expect((openDB as any).mock.calls.length - callsBefore).toBe(1)
    })
  })
})
