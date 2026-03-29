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
