// Copyright (c) 2025 Kenneth Stott
// Canary: 319ce6ba-1cde-4cdd-b49f-c74c49e6582c
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Public (unauthenticated) queries ----------------------------------------

export const PUBLIC_SESSION_QUERY = gql`
  query PublicSession($sessionId: String!) {
    publicSession(sessionId: $sessionId) {
      sessionId
      summary
      status
    }
  }
`

export const PUBLIC_MESSAGES_QUERY = gql`
  query PublicMessages($sessionId: String!) {
    publicMessages(sessionId: $sessionId) {
      messages {
        id
        type
        content
        timestamp
        stepNumber
        isFinalInsight
        stepDurationMs
        role
        skills
      }
    }
  }
`

export const PUBLIC_ARTIFACTS_QUERY = gql`
  query PublicArtifacts($sessionId: String!) {
    publicArtifacts(sessionId: $sessionId) {
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
      version
      versionCount
    }
  }
`

export const PUBLIC_ARTIFACT_QUERY = gql`
  query PublicArtifact($sessionId: String!, $artifactId: Int!) {
    publicArtifact(sessionId: $sessionId, artifactId: $artifactId) {
      id
      name
      artifactType
      content
      mimeType
      isBinary
    }
  }
`

export const PUBLIC_TABLES_QUERY = gql`
  query PublicTables($sessionId: String!) {
    publicTables(sessionId: $sessionId) {
      name
      rowCount
      stepNumber
      columns
      isStarred
      version
      versionCount
    }
  }
`

export const PUBLIC_TABLE_DATA_QUERY = gql`
  query PublicTableData($sessionId: String!, $tableName: String!, $page: Int, $pageSize: Int) {
    publicTableData(sessionId: $sessionId, tableName: $tableName, page: $page, pageSize: $pageSize) {
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

export const PUBLIC_PROOF_FACTS_QUERY = gql`
  query PublicProofFacts($sessionId: String!) {
    publicProofFacts(sessionId: $sessionId) {
      facts {
        id
        name
        description
        status
        value
        source
        confidence
        tier
        strategy
        formula
        reason
        dependencies
        elapsedMs
      }
      summary
    }
  }
`
