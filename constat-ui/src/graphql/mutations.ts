import { gql } from '@apollo/client'

export const CREATE_GLOSSARY_TERM_MUTATION = gql`
  mutation CreateGlossaryTerm($sessionId: String!, $input: GlossaryTermInput!) {
    createGlossaryTerm(sessionId: $sessionId, input: $input) {
      name
      displayName
      definition
      domain
      aliases
      semanticType
      status
      provenance
      glossaryStatus
      glossaryId
    }
  }
`

export const UPDATE_GLOSSARY_TERM_MUTATION = gql`
  mutation UpdateGlossaryTerm($sessionId: String!, $name: String!, $input: GlossaryTermUpdateInput!) {
    updateGlossaryTerm(sessionId: $sessionId, name: $name, input: $input) {
      name
      displayName
      definition
      domain
      domainPath
      parentId
      parentVerb
      aliases
      semanticType
      status
      provenance
      tags
      ignored
      canonicalSource
    }
  }
`

export const DELETE_GLOSSARY_TERM_MUTATION = gql`
  mutation DeleteGlossaryTerm($sessionId: String!, $name: String!, $domain: String) {
    deleteGlossaryTerm(sessionId: $sessionId, name: $name, domain: $domain)
  }
`

export const CREATE_RELATIONSHIP_MUTATION = gql`
  mutation CreateRelationship($sessionId: String!, $subject: String!, $verb: String!, $object: String!) {
    createRelationship(sessionId: $sessionId, subject: $subject, verb: $verb, object: $object) {
      id
      subject
      verb
      object
      confidence
      userEdited
    }
  }
`

export const UPDATE_RELATIONSHIP_MUTATION = gql`
  mutation UpdateRelationship($sessionId: String!, $relId: String!, $verb: String!) {
    updateRelationship(sessionId: $sessionId, relId: $relId, verb: $verb) {
      id
      subject
      verb
      object
      confidence
      userEdited
    }
  }
`

export const DELETE_RELATIONSHIP_MUTATION = gql`
  mutation DeleteRelationship($sessionId: String!, $relId: String!) {
    deleteRelationship(sessionId: $sessionId, relId: $relId)
  }
`

export const APPROVE_RELATIONSHIP_MUTATION = gql`
  mutation ApproveRelationship($sessionId: String!, $relId: String!) {
    approveRelationship(sessionId: $sessionId, relId: $relId) {
      id
      subject
      verb
      object
      confidence
      userEdited
    }
  }
`

export const DELETE_GLOSSARY_TERMS_MUTATION = gql`
  mutation DeleteGlossaryTerms($sessionId: String!, $names: [String!]!) {
    deleteGlossaryTerms(sessionId: $sessionId, names: $names)
  }
`

export const BULK_UPDATE_STATUS_MUTATION = gql`
  mutation BulkUpdateStatus($sessionId: String!, $names: [String!]!, $newStatus: String!) {
    bulkUpdateStatus(sessionId: $sessionId, names: $names, newStatus: $newStatus)
  }
`

export const GENERATE_GLOSSARY_MUTATION = gql`
  mutation GenerateGlossary($sessionId: String!, $phases: JSON) {
    generateGlossary(sessionId: $sessionId, phases: $phases) {
      status
      message
    }
  }
`

export const DRAFT_DEFINITION_MUTATION = gql`
  mutation DraftDefinition($sessionId: String!, $name: String!) {
    draftDefinition(sessionId: $sessionId, name: $name) {
      name
      draft
    }
  }
`

export const DRAFT_ALIASES_MUTATION = gql`
  mutation DraftAliases($sessionId: String!, $name: String!) {
    draftAliases(sessionId: $sessionId, name: $name) {
      name
      aliases
    }
  }
`

export const DRAFT_TAGS_MUTATION = gql`
  mutation DraftTags($sessionId: String!, $name: String!) {
    draftTags(sessionId: $sessionId, name: $name) {
      name
      tags
    }
  }
`

export const REFINE_DEFINITION_MUTATION = gql`
  mutation RefineDefinition($sessionId: String!, $name: String!) {
    refineDefinition(sessionId: $sessionId, name: $name) {
      name
      before
      after
    }
  }
`

export const SUGGEST_TAXONOMY_MUTATION = gql`
  mutation SuggestTaxonomy($sessionId: String!) {
    suggestTaxonomy(sessionId: $sessionId) {
      suggestions {
        child
        parent
        parentVerb
        confidence
        reason
      }
      message
    }
  }
`

export const RENAME_TERM_MUTATION = gql`
  mutation RenameTerm($sessionId: String!, $name: String!, $newName: String!) {
    renameTerm(sessionId: $sessionId, name: $name, newName: $newName) {
      oldName
      newName
      displayName
      relationshipsUpdated
    }
  }
`
