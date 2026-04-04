// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const HANDBOOK_QUERY = gql`
  query Handbook($sessionId: String!, $domain: String) {
    handbook(sessionId: $sessionId, domain: $domain) {
      domain
      generatedAt
      summary
      sections
    }
  }
`

export const HANDBOOK_SECTION_QUERY = gql`
  query HandbookSection($sessionId: String!, $section: String!, $domain: String) {
    handbookSection(sessionId: $sessionId, section: $section, domain: $domain) {
      title
      content {
        key
        display
        metadata
        editable
      }
      lastUpdated
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const UPDATE_HANDBOOK_ENTRY = gql`
  mutation UpdateHandbookEntry(
    $sessionId: String!
    $section: String!
    $key: String!
    $fieldName: String!
    $newValue: String!
    $reason: String
  ) {
    updateHandbookEntry(
      sessionId: $sessionId
      section: $section
      key: $key
      fieldName: $fieldName
      newValue: $newValue
      reason: $reason
    ) {
      status
      section
      key
    }
  }
`
