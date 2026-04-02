// Copyright (c) 2025 Kenneth Stott
// Canary: 7170e77b-2a1f-454e-aa46-a4c347324518
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const TABLES_QUERY = gql`
  query Tables($sessionId: String!) {
    tables(sessionId: $sessionId) {
      tables {
        name
        rowCount
        stepNumber
        columns
        isStarred
        isView
        roleId
        version
        versionCount
      }
      total
    }
  }
`

export const TABLE_DATA_QUERY = gql`
  query TableData($sessionId: String!, $tableName: String!, $page: Int, $pageSize: Int) {
    tableData(sessionId: $sessionId, tableName: $tableName, page: $page, pageSize: $pageSize) {
      name
      columns
      data
      totalRows
      page
      pageSize
      hasMore
    }
  }
`

export const TABLE_VERSIONS_QUERY = gql`
  query TableVersions($sessionId: String!, $name: String!) {
    tableVersions(sessionId: $sessionId, name: $name) {
      name
      currentVersion
      versions {
        version
        stepNumber
        rowCount
        createdAt
      }
    }
  }
`

export const TABLE_VERSION_DATA_QUERY = gql`
  query TableVersionData($sessionId: String!, $name: String!, $version: Int!, $page: Int, $pageSize: Int) {
    tableVersionData(sessionId: $sessionId, name: $name, version: $version, page: $page, pageSize: $pageSize) {
      name
      columns
      data
      totalRows
      page
      pageSize
      hasMore
    }
  }
`

export const ARTIFACTS_QUERY = gql`
  query Artifacts($sessionId: String!) {
    artifacts(sessionId: $sessionId) {
      artifacts {
        id
        name
        artifactType
        stepNumber
        title
        description
        mimeType
        createdAt
        isStarred
        metadata
        roleId
        version
        versionCount
      }
      total
    }
  }
`

export const ARTIFACT_QUERY = gql`
  query Artifact($sessionId: String!, $id: Int!) {
    artifact(sessionId: $sessionId, id: $id) {
      id
      name
      artifactType
      content
      mimeType
      isBinary
    }
  }
`

export const ARTIFACT_VERSIONS_QUERY = gql`
  query ArtifactVersions($sessionId: String!, $id: Int!) {
    artifactVersions(sessionId: $sessionId, id: $id) {
      name
      currentVersion
      versions {
        id
        version
        stepNumber
        attempt
        createdAt
      }
    }
  }
`

export const FACTS_QUERY = gql`
  query Facts($sessionId: String!) {
    facts(sessionId: $sessionId) {
      facts {
        name
        value
        source
        reasoning
        confidence
        isPersisted
        roleId
        domain
      }
      total
    }
  }
`

export const ENTITIES_QUERY = gql`
  query Entities($sessionId: String!, $entityType: String) {
    entities(sessionId: $sessionId, entityType: $entityType) {
      entities {
        id
        name
        type
        types
        sources
        metadata
        references {
          document
          section
          mentions
          mentionText
        }
        mentionCount
        originalName
        relatedEntities
      }
      total
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const DELETE_TABLE = gql`
  mutation DeleteTable($sessionId: String!, $name: String!) {
    deleteTable(sessionId: $sessionId, name: $name) {
      status
      name
    }
  }
`

export const TOGGLE_TABLE_STAR = gql`
  mutation ToggleTableStar($sessionId: String!, $name: String!) {
    toggleTableStar(sessionId: $sessionId, name: $name) {
      name
      isStarred
    }
  }
`

export const DELETE_ARTIFACT = gql`
  mutation DeleteArtifact($sessionId: String!, $id: Int!) {
    deleteArtifact(sessionId: $sessionId, id: $id) {
      status
      name
    }
  }
`

export const TOGGLE_ARTIFACT_STAR = gql`
  mutation ToggleArtifactStar($sessionId: String!, $id: Int!) {
    toggleArtifactStar(sessionId: $sessionId, id: $id) {
      name
      isStarred
    }
  }
`

export const ADD_FACT = gql`
  mutation AddFact($sessionId: String!, $name: String!, $value: JSON!, $persist: Boolean) {
    addFact(sessionId: $sessionId, name: $name, value: $value, persist: $persist) {
      status
      fact {
        name
        value
        source
        isPersisted
      }
    }
  }
`

export const EDIT_FACT = gql`
  mutation EditFact($sessionId: String!, $factName: String!, $value: JSON!) {
    editFact(sessionId: $sessionId, factName: $factName, value: $value) {
      status
      fact {
        name
        value
        source
      }
    }
  }
`

export const PERSIST_FACT = gql`
  mutation PersistFact($sessionId: String!, $factName: String!) {
    persistFact(sessionId: $sessionId, factName: $factName) {
      status
    }
  }
`

export const FORGET_FACT = gql`
  mutation ForgetFact($sessionId: String!, $factName: String!) {
    forgetFact(sessionId: $sessionId, factName: $factName) {
      status
    }
  }
`

export const MOVE_FACT = gql`
  mutation MoveFact($sessionId: String!, $factName: String!, $toDomain: String!) {
    moveFact(sessionId: $sessionId, factName: $factName, toDomain: $toDomain) {
      status
      factName
      toDomain
    }
  }
`

export const ADD_ENTITY_TO_GLOSSARY = gql`
  mutation AddEntityToGlossary($sessionId: String!, $entityId: String!) {
    addEntityToGlossary(sessionId: $sessionId, entityId: $entityId) {
      status
      entityId
      note
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toTableData(gql: any) {
  return {
    name: gql.name,
    columns: gql.columns,
    data: gql.data,
    total_rows: gql.totalRows,
    page: gql.page,
    page_size: gql.pageSize,
    has_more: gql.hasMore,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toTableVersions(gql: any) {
  return {
    name: gql.name,
    current_version: gql.currentVersion,
    versions: (gql.versions ?? []).map((v: any) => ({
      version: v.version,
      step_number: v.stepNumber ?? undefined,
      row_count: v.rowCount,
      created_at: v.createdAt ?? undefined,
    })),
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toArtifactVersions(gql: any) {
  return {
    name: gql.name,
    current_version: gql.currentVersion,
    versions: (gql.versions ?? []).map((v: any) => ({
      id: v.id,
      version: v.version,
      step_number: v.stepNumber,
      attempt: v.attempt,
      created_at: v.createdAt ?? undefined,
    })),
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toTableInfo(gql: any) {
  return {
    name: gql.name,
    row_count: gql.rowCount,
    step_number: gql.stepNumber,
    columns: gql.columns,
    is_starred: gql.isStarred ?? false,
    is_view: gql.isView ?? false,
    role_id: gql.roleId ?? undefined,
    version: gql.version ?? 1,
    version_count: gql.versionCount ?? 1,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toArtifact(gql: any) {
  return {
    id: gql.id,
    name: gql.name,
    artifact_type: gql.artifactType,
    step_number: gql.stepNumber,
    title: gql.title ?? undefined,
    description: gql.description ?? undefined,
    mime_type: gql.mimeType,
    created_at: gql.createdAt ?? undefined,
    is_starred: gql.isStarred ?? false,
    is_key_result: gql.isStarred ?? false,  // unified: starred = key result
    metadata: gql.metadata ?? undefined,
    role_id: gql.roleId ?? undefined,
    version: gql.version ?? 1,
    version_count: gql.versionCount ?? 1,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toArtifactContent(gql: any) {
  return {
    id: gql.id,
    name: gql.name,
    artifact_type: gql.artifactType,
    step_number: gql.stepNumber ?? 0,
    content: gql.content,
    mime_type: gql.mimeType,
    is_binary: gql.isBinary,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toFact(gql: any) {
  return {
    name: gql.name,
    value: gql.value,
    source: gql.source,
    reasoning: gql.reasoning ?? undefined,
    confidence: gql.confidence ?? undefined,
    is_persisted: gql.isPersisted ?? false,
    role_id: gql.roleId ?? undefined,
    domain: gql.domain ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toEntity(gql: any) {
  return {
    id: gql.id,
    name: gql.name,
    type: gql.type,
    types: gql.types ?? [],
    sources: gql.sources ?? [],
    metadata: gql.metadata ?? {},
    references: (gql.references ?? []).map((r: any) => ({
      document: r.document,
      section: r.section ?? undefined,
      mentions: r.mentions ?? 0,
      mention_text: r.mentionText ?? undefined,
    })),
    mention_count: gql.mentionCount ?? 0,
    original_name: gql.originalName ?? undefined,
    related_entities: gql.relatedEntities ?? undefined,
  }
}
