import { useMutation } from '@apollo/client/react'
import { useGlossaryStore } from '@/store/glossaryStore'
import {
  CREATE_GLOSSARY_TERM_MUTATION,
  UPDATE_GLOSSARY_TERM_MUTATION,
  DELETE_GLOSSARY_TERM_MUTATION,
  DELETE_GLOSSARY_TERMS_MUTATION,
  CREATE_RELATIONSHIP_MUTATION,
  UPDATE_RELATIONSHIP_MUTATION,
  DELETE_RELATIONSHIP_MUTATION,
  APPROVE_RELATIONSHIP_MUTATION,
  BULK_UPDATE_STATUS_MUTATION,
  GENERATE_GLOSSARY_MUTATION,
  DRAFT_DEFINITION_MUTATION,
  DRAFT_ALIASES_MUTATION,
  DRAFT_TAGS_MUTATION,
  REFINE_DEFINITION_MUTATION,
  SUGGEST_TAXONOMY_MUTATION,
  RENAME_TERM_MUTATION,
} from '@/graphql/mutations'

const REFETCH = { refetchQueries: ['Glossary'] }

export function useGlossaryMutations(sessionId: string) {
  const [createTermMut] = useMutation(CREATE_GLOSSARY_TERM_MUTATION, REFETCH)
  const [updateTermMut] = useMutation(UPDATE_GLOSSARY_TERM_MUTATION, REFETCH)
  const [deleteTermMut] = useMutation(DELETE_GLOSSARY_TERM_MUTATION)
  const [deleteTermsMut] = useMutation(DELETE_GLOSSARY_TERMS_MUTATION)
  const [createRelMut] = useMutation(CREATE_RELATIONSHIP_MUTATION, REFETCH)
  const [updateRelMut] = useMutation(UPDATE_RELATIONSHIP_MUTATION, REFETCH)
  const [deleteRelMut] = useMutation(DELETE_RELATIONSHIP_MUTATION, REFETCH)
  const [approveRelMut] = useMutation(APPROVE_RELATIONSHIP_MUTATION, REFETCH)
  const [bulkStatusMut] = useMutation(BULK_UPDATE_STATUS_MUTATION, REFETCH)
  const [generateMut] = useMutation(GENERATE_GLOSSARY_MUTATION)
  const [draftDefMut] = useMutation(DRAFT_DEFINITION_MUTATION)
  const [draftAliasesMut] = useMutation(DRAFT_ALIASES_MUTATION)
  const [draftTagsMut] = useMutation(DRAFT_TAGS_MUTATION)
  const [refineMut] = useMutation(REFINE_DEFINITION_MUTATION, REFETCH)
  const [suggestMut] = useMutation(SUGGEST_TAXONOMY_MUTATION)
  const [renameMut] = useMutation(RENAME_TERM_MUTATION, REFETCH)

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function evictTerms(cache: any, names: string[]) {
    const nameSet = new Set(names.map(n => n.toLowerCase()))
    cache.modify({
      fields: {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        glossary(existing: any, { readField }: any) {
          if (!existing?.terms) return existing
          return {
            ...existing,
            terms: existing.terms.filter(
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              (ref: any) => {
                const name = readField('name', ref)
                return !nameSet.has((name as string)?.toLowerCase())
              }
            ),
          }
        },
      },
    })
  }

  return {
    createTerm: (input: Record<string, unknown>) =>
      createTermMut({ variables: { sessionId, input } }),

    updateTerm: (name: string, input: Record<string, unknown>) =>
      updateTermMut({ variables: { sessionId, name, input } }),

    deleteTerm: (name: string, domain?: string) =>
      deleteTermMut({
        variables: { sessionId, name, ...(domain != null && { domain }) },
        update: (cache) => evictTerms(cache, [name]),
      }),

    deleteTerms: (names: string[]) =>
      deleteTermsMut({
        variables: { sessionId, names },
        update: (cache) => evictTerms(cache, names),
      }),

    createRelationship: (subject: string, verb: string, object: string) =>
      createRelMut({ variables: { sessionId, subject, verb, object } }),

    updateRelationship: (relId: string, verb: string) =>
      updateRelMut({ variables: { sessionId, relId, verb } }),

    deleteRelationship: (relId: string) =>
      deleteRelMut({ variables: { sessionId, relId } }),

    approveRelationship: (relId: string) =>
      approveRelMut({ variables: { sessionId, relId } }),

    bulkUpdateStatus: (names: string[], newStatus: string) =>
      bulkStatusMut({ variables: { sessionId, names, newStatus } }),

    generateGlossary: async (phases?: Record<string, boolean>) => {
      useGlossaryStore.getState().setGenerating(true)
      await generateMut({ variables: { sessionId, phases } })
    },

    draftDefinition: async (name: string) => {
      const { data } = await draftDefMut({ variables: { sessionId, name } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (data as any)?.draftDefinition as { name: string; draft: string } | undefined
    },

    draftAliases: async (name: string) => {
      const { data } = await draftAliasesMut({ variables: { sessionId, name } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (data as any)?.draftAliases as { name: string; aliases: string[] } | undefined
    },

    draftTags: async (name: string) => {
      const { data } = await draftTagsMut({ variables: { sessionId, name } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (data as any)?.draftTags as { name: string; tags: string[] } | undefined
    },

    refineDefinition: async (name: string) => {
      const { data } = await refineMut({ variables: { sessionId, name } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (data as any)?.refineDefinition as { name: string; before: string | null; after: string } | undefined
    },

    suggestTaxonomy: async () => {
      const { data } = await suggestMut({ variables: { sessionId } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const result = (data as any)?.suggestTaxonomy
      if (result?.suggestions) {
        useGlossaryStore.getState().setTaxonomySuggestions(result.suggestions)
      }
      return result
    },

    renameTerm: async (name: string, newName: string) => {
      const { data } = await renameMut({ variables: { sessionId, name, newName } })
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return (data as any)?.renameTerm as { oldName: string; newName: string; displayName: string; relationshipsUpdated: number } | undefined
    },
  }
}
