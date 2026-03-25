import { describe, it, expect } from 'vitest'
import { inflateToGlossaryTerms, type CompactState } from '@/store/entityCacheKeys'

describe('inflateToGlossaryTerms', () => {
  it('maps compact glossary terms to GlossaryTerm[]', () => {
    const compact: CompactState = {
      e: {
        ent1: { a: 'norway', b: 'Norway', c: 'concept', d: 'GPE', e: 'hr-reporting' },
      },
      g: {
        ghi789: { a: 'order', b: 'Order', c: 'A customer purchase', d: 'reviewed', e: null, f: ['purchase'] },
        ghi790: { a: 'customer', b: 'Customer', c: null, d: 'draft', e: 'ghi789', f: [] },
      },
      r: {
        rel1: { a: 'customer', b: 'places', c: 'order', d: 0.95 },
      },
      k: {
        norway: ['sweden', 'denmark'],
      },
    }

    const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(compact)
    expect(terms).toHaveLength(2)
    expect(totalDefined).toBe(1)
    expect(totalSelfDescribing).toBe(1)

    const order = terms.find((t) => t.name === 'order')!
    expect(order).toBeDefined()
    expect(order.display_name).toBe('Order')
    expect(order.definition).toBe('A customer purchase')
    expect(order.status).toBe('reviewed')
    expect(order.parent_id).toBeNull()
    expect(order.aliases).toEqual(['purchase'])
    expect(order.cardinality).toBe('unknown')
    expect(order.glossary_status).toBe('defined')
    expect(order.connected_resources).toEqual([])

    const customer = terms.find((t) => t.name === 'customer')!
    expect(customer).toBeDefined()
    expect(customer.definition).toBeNull()
    expect(customer.parent_id).toBe('ghi789')
    expect(customer.aliases).toEqual([])
    expect(customer.glossary_status).toBe('self_describing')
  })

  it('returns empty for empty glossary block', () => {
    const compact: CompactState = {
      e: {},
      g: {},
      r: {},
      k: {},
    }
    const result = inflateToGlossaryTerms(compact)
    expect(result.terms).toEqual([])
    expect(result.totalDefined).toBe(0)
    expect(result.totalSelfDescribing).toBe(0)
  })

  it('handles state with only entities (no glossary terms)', () => {
    const compact: CompactState = {
      e: {
        ent1: { a: 'norway', b: 'Norway', c: 'concept', d: 'GPE', e: 'hr-reporting' },
      },
      g: {},
      r: {},
      k: {},
    }
    const result = inflateToGlossaryTerms(compact)
    expect(result.terms).toEqual([])
    expect(result.totalDefined).toBe(0)
    expect(result.totalSelfDescribing).toBe(0)
  })
})
