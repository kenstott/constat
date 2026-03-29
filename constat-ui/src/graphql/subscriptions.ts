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
