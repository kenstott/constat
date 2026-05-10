// Copyright (c) 2025 Kenneth Stott
// Canary: 1e7f7ab8-fae9-41ba-8182-155207eba214
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

export const GLOSSARY_CHANGED_SUBSCRIPTION = gql`
  subscription GlossaryChanged($sessionId: String!) {
    glossaryChanged(sessionId: $sessionId) {
      sessionId
      action
      termName
      stage
      percent
      termsCount
      durationMs
      error
      term {
        name
        displayName
        definition
        domain
        domainPath
        parentId
        parentVerb
        aliases
        semanticType
        cardinality
        status
        provenance
        glossaryStatus
        entityId
        glossaryId
        nerType
        tags
        ignored
        canonicalSource
        spanningDomains
      }
    }
  }
`
