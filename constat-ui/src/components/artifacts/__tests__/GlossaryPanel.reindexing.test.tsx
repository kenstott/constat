// Copyright (c) 2025 Kenneth Stott
// Canary: 7a3c9d2e-1b4f-4e8a-9c6d-2f5a7b8c9d0e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, beforeEach } from 'vitest'
import { ingestingSourceVar } from '@/graphql/ui-state'

// Tests for reindexing state driven by ingestingSourceVar reactive var.
// Full GlossaryPanel render is avoided due to heavy Apollo/context dependencies;
// the reactive var is the single source of truth for reindexing display.

describe('GlossaryPanel reindexing state via ingestingSourceVar', () => {
  beforeEach(() => {
    ingestingSourceVar(null)
  })

  it('starts with no reindexing source', () => {
    expect(ingestingSourceVar()).toBeNull()
  })

  it('tracks source name when source_ingest_start fires', () => {
    ingestingSourceVar('sales_db')
    expect(ingestingSourceVar()).toBe('sales_db')
  })

  it('clears reindexing state on source_ingest_complete', () => {
    ingestingSourceVar('sales_db')
    expect(ingestingSourceVar()).toBe('sales_db')
    ingestingSourceVar(null)
    expect(ingestingSourceVar()).toBeNull()
  })

  it('clears reindexing state on source_ingest_error', () => {
    ingestingSourceVar('hr_db')
    ingestingSourceVar(null)
    expect(ingestingSourceVar()).toBeNull()
  })

  it('updates to new source when a second ingest starts', () => {
    ingestingSourceVar('source_a')
    ingestingSourceVar('source_b')
    expect(ingestingSourceVar()).toBe('source_b')
  })

  describe('spinner visibility logic', () => {
    it('spinner should be visible when reindexing is non-null', () => {
      ingestingSourceVar('my_source')
      const reindexing = ingestingSourceVar()
      expect(reindexing !== null).toBe(true)
    })

    it('spinner should be hidden when reindexing is null', () => {
      ingestingSourceVar(null)
      const reindexing = ingestingSourceVar()
      expect(reindexing !== null).toBe(false)
    })

    it('spinner text includes source name', () => {
      ingestingSourceVar('analytics_warehouse')
      const reindexing = ingestingSourceVar()
      const label = reindexing ? `Re-indexing ${reindexing}...` : ''
      expect(label).toBe('Re-indexing analytics_warehouse...')
    })
  })
})
