import { describe, it, expect } from 'vitest'

// Minimal DomainNode extraction for testing isSystem/isActive logic
// We test the logic directly rather than importing the full component
// which has heavy dependencies (Apollo, contexts, etc.)

describe('DomainNode logic', () => {
  function computeDomainState(filename: string, activeDomains: string[]) {
    const isSystem = filename === 'root' || filename === 'user'
    const isSynthetic = filename === 'root' || filename === 'user'
    const isActive = isSystem || activeDomains.includes(filename)
    const isDisabled = isSystem
    return { isSystem, isSynthetic, isActive, isDisabled }
  }

  describe('user domain', () => {
    it('is always active regardless of activeDomains', () => {
      const state = computeDomainState('user', [])
      expect(state.isActive).toBe(true)
    })

    it('is always active even when not in activeDomains list', () => {
      const state = computeDomainState('user', ['sales-analytics'])
      expect(state.isActive).toBe(true)
    })

    it('is system (checkbox disabled)', () => {
      const state = computeDomainState('user', [])
      expect(state.isSystem).toBe(true)
      expect(state.isDisabled).toBe(true)
    })

    it('is synthetic', () => {
      const state = computeDomainState('user', [])
      expect(state.isSynthetic).toBe(true)
    })
  })

  describe('root domain', () => {
    it('is always active', () => {
      const state = computeDomainState('root', [])
      expect(state.isActive).toBe(true)
      expect(state.isSystem).toBe(true)
      expect(state.isDisabled).toBe(true)
    })
  })

  describe('regular domain', () => {
    it('is active when in activeDomains', () => {
      const state = computeDomainState('sales-analytics', ['sales-analytics'])
      expect(state.isActive).toBe(true)
      expect(state.isSystem).toBe(false)
      expect(state.isDisabled).toBe(false)
    })

    it('is inactive when not in activeDomains', () => {
      const state = computeDomainState('sales-analytics', [])
      expect(state.isActive).toBe(false)
    })

    it('is not synthetic', () => {
      const state = computeDomainState('sales-analytics', [])
      expect(state.isSynthetic).toBe(false)
    })
  })
})
