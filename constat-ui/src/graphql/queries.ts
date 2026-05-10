// Copyright (c) 2025 Kenneth Stott
// Canary: ce422167-a122-432e-a31b-e55f4d325472
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

export const GLOSSARY_QUERY = gql`
  query Glossary($sessionId: String!, $scope: String, $domain: String) {
    glossary(sessionId: $sessionId, scope: $scope, domain: $domain) {
      terms {
        name
        displayName
        definition
        domain
        domainPath
        parentId
        parentVerb
        aliases
        semanticType
        nerType
        cardinality
        status
        provenance
        glossaryStatus
        entityId
        glossaryId
        tags
        ignored
        canonicalSource
        spanningDomains
      }
      totalDefined
      totalSelfDescribing
      clusters
    }
  }
`

export const GLOSSARY_TERM_QUERY = gql`
  query GlossaryTerm($sessionId: String!, $name: String!) {
    glossaryTerm(sessionId: $sessionId, name: $name) {
      name
      displayName
      definition
      domain
      domainPath
      parentId
      parentVerb
      parent {
        name
        displayName
      }
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
      connectedResources {
        entityName
        entityType
        sources {
          documentName
          source
          section
          url
        }
      }
      children {
        name
        displayName
        parentVerb
      }
      relationships {
        id
        subject
        verb
        object
        confidence
        userEdited
      }
      clusterSiblings
      spanningDomains
    }
  }
`
