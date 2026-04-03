import { describe, it, expect } from 'vitest'
import {
  toTableData,
  toTableVersions,
  toArtifactVersions,
  toTableInfo,
  toArtifact,
  toArtifactContent,
  toFact,
  toEntity,
} from '../data'

describe('toTableData', () => {
  it('maps GraphQL table data to snake_case', () => {
    const result = toTableData({
      name: 'users',
      columns: ['id', 'name'],
      data: [[1, 'Alice']],
      totalRows: 100,
      page: 1,
      pageSize: 50,
      hasMore: true,
    })
    expect(result).toEqual({
      name: 'users',
      columns: ['id', 'name'],
      data: [[1, 'Alice']],
      total_rows: 100,
      page: 1,
      page_size: 50,
      has_more: true,
    })
  })
})

describe('toTableVersions', () => {
  it('maps version data with all fields', () => {
    const result = toTableVersions({
      name: 'orders',
      currentVersion: 3,
      versions: [
        { version: 1, stepNumber: 2, rowCount: 10, createdAt: '2024-01-01' },
        { version: 2, stepNumber: 5, rowCount: 20, createdAt: null },
      ],
    })
    expect(result).toEqual({
      name: 'orders',
      current_version: 3,
      versions: [
        { version: 1, step_number: 2, row_count: 10, created_at: '2024-01-01' },
        { version: 2, step_number: 5, row_count: 20, created_at: undefined },
      ],
    })
  })

  it('handles null versions array', () => {
    const result = toTableVersions({ name: 'empty', currentVersion: 1, versions: null })
    expect(result.versions).toEqual([])
  })

  it('handles undefined versions array', () => {
    const result = toTableVersions({ name: 'empty', currentVersion: 1 })
    expect(result.versions).toEqual([])
  })
})

describe('toArtifactVersions', () => {
  it('maps artifact version data', () => {
    const result = toArtifactVersions({
      name: 'chart.png',
      currentVersion: 2,
      versions: [
        { id: 1, version: 1, stepNumber: 3, attempt: 1, createdAt: '2024-01-01' },
        { id: 2, version: 2, stepNumber: 5, attempt: 2, createdAt: null },
      ],
    })
    expect(result).toEqual({
      name: 'chart.png',
      current_version: 2,
      versions: [
        { id: 1, version: 1, step_number: 3, attempt: 1, created_at: '2024-01-01' },
        { id: 2, version: 2, step_number: 5, attempt: 2, created_at: undefined },
      ],
    })
  })

  it('handles null versions', () => {
    const result = toArtifactVersions({ name: 'x', currentVersion: 1, versions: null })
    expect(result.versions).toEqual([])
  })
})

describe('toTableInfo', () => {
  it('maps full table info', () => {
    const result = toTableInfo({
      name: 'products',
      rowCount: 500,
      stepNumber: 3,
      columns: ['id', 'price'],
      isStarred: true,
      isView: true,
      roleId: 'role-1',
      version: 2,
      versionCount: 3,
    })
    expect(result).toEqual({
      name: 'products',
      row_count: 500,
      step_number: 3,
      columns: ['id', 'price'],
      is_starred: true,
      is_view: true,
      role_id: 'role-1',
      version: 2,
      version_count: 3,
    })
  })

  it('applies defaults for nullable fields', () => {
    const result = toTableInfo({
      name: 'minimal',
      rowCount: 0,
      stepNumber: 1,
      columns: [],
    })
    expect(result.is_starred).toBe(false)
    expect(result.is_view).toBe(false)
    expect(result.role_id).toBeUndefined()
    expect(result.version).toBe(1)
    expect(result.version_count).toBe(1)
  })
})

describe('toArtifact', () => {
  it('maps full artifact info', () => {
    const result = toArtifact({
      id: 1,
      name: 'report.html',
      artifactType: 'html',
      stepNumber: 4,
      title: 'Sales Report',
      description: 'Q4 report',
      mimeType: 'text/html',
      createdAt: '2024-01-15',
      isStarred: true,
      metadata: { key: 'val' },
      roleId: 'r1',
      version: 3,
      versionCount: 5,
    })
    expect(result).toEqual({
      id: 1,
      name: 'report.html',
      artifact_type: 'html',
      step_number: 4,
      title: 'Sales Report',
      description: 'Q4 report',
      mime_type: 'text/html',
      created_at: '2024-01-15',
      is_starred: true,
      is_key_result: true,
      metadata: { key: 'val' },
      role_id: 'r1',
      version: 3,
      version_count: 5,
    })
  })

  it('applies defaults for nullable fields', () => {
    const result = toArtifact({
      id: 2,
      name: 'x',
      artifactType: 'text',
      stepNumber: 1,
      mimeType: 'text/plain',
    })
    expect(result.title).toBeUndefined()
    expect(result.description).toBeUndefined()
    expect(result.created_at).toBeUndefined()
    expect(result.is_starred).toBe(false)
    expect(result.is_key_result).toBe(false)
    expect(result.metadata).toBeUndefined()
    expect(result.role_id).toBeUndefined()
    expect(result.version).toBe(1)
    expect(result.version_count).toBe(1)
  })
})

describe('toArtifactContent', () => {
  it('maps artifact content', () => {
    const result = toArtifactContent({
      id: 5,
      name: 'data.json',
      artifactType: 'json',
      stepNumber: 2,
      content: '{"a":1}',
      mimeType: 'application/json',
      isBinary: false,
    })
    expect(result).toEqual({
      id: 5,
      name: 'data.json',
      artifact_type: 'json',
      step_number: 2,
      content: '{"a":1}',
      mime_type: 'application/json',
      is_binary: false,
    })
  })

  it('defaults stepNumber to 0 when null', () => {
    const result = toArtifactContent({
      id: 6,
      name: 'x',
      artifactType: 'text',
      stepNumber: null,
      content: '',
      mimeType: 'text/plain',
      isBinary: false,
    })
    expect(result.step_number).toBe(0)
  })
})

describe('toFact', () => {
  it('maps full fact', () => {
    const result = toFact({
      name: 'revenue',
      value: 1000000,
      source: 'analysis',
      reasoning: 'Calculated from Q4',
      confidence: 0.95,
      isPersisted: true,
      roleId: 'analyst',
      domain: 'finance',
    })
    expect(result).toEqual({
      name: 'revenue',
      value: 1000000,
      source: 'analysis',
      reasoning: 'Calculated from Q4',
      confidence: 0.95,
      is_persisted: true,
      role_id: 'analyst',
      domain: 'finance',
    })
  })

  it('applies defaults for nullable fields', () => {
    const result = toFact({
      name: 'basic',
      value: 'test',
      source: 'user',
    })
    expect(result.reasoning).toBeUndefined()
    expect(result.confidence).toBeUndefined()
    expect(result.is_persisted).toBe(false)
    expect(result.role_id).toBeUndefined()
    expect(result.domain).toBeUndefined()
  })
})

describe('toEntity', () => {
  it('maps full entity with references', () => {
    const result = toEntity({
      id: 'e1',
      name: 'Acme Corp',
      type: 'organization',
      types: ['organization', 'company'],
      sources: ['doc1', 'doc2'],
      metadata: { industry: 'tech' },
      references: [
        { document: 'doc1', section: 'intro', mentions: 3, mentionText: 'Acme Corp' },
        { document: 'doc2', section: null, mentions: null, mentionText: null },
      ],
      mentionCount: 5,
      originalName: 'ACME CORP',
      relatedEntities: ['e2', 'e3'],
    })
    expect(result).toEqual({
      id: 'e1',
      name: 'Acme Corp',
      type: 'organization',
      types: ['organization', 'company'],
      sources: ['doc1', 'doc2'],
      metadata: { industry: 'tech' },
      references: [
        { document: 'doc1', section: 'intro', mentions: 3, mention_text: 'Acme Corp' },
        { document: 'doc2', section: undefined, mentions: 0, mention_text: undefined },
      ],
      mention_count: 5,
      original_name: 'ACME CORP',
      related_entities: ['e2', 'e3'],
    })
  })

  it('applies defaults for nullable fields', () => {
    const result = toEntity({
      id: 'e2',
      name: 'Widget',
      type: 'product',
    })
    expect(result.types).toEqual([])
    expect(result.sources).toEqual([])
    expect(result.metadata).toEqual({})
    expect(result.references).toEqual([])
    expect(result.mention_count).toBe(0)
    expect(result.original_name).toBeUndefined()
    expect(result.related_entities).toBeUndefined()
  })
})
