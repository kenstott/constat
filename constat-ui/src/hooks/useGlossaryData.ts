import { useQuery, useSubscription } from '@apollo/client/react'
import { GLOSSARY_QUERY } from '@/graphql/queries'
import { GLOSSARY_CHANGED_SUBSCRIPTION } from '@/graphql/subscriptions'
import { useGlossaryStore } from '@/store/glossaryStore'
import type { GlossaryTerm } from '@/types/api'
import { useMemo, useRef, useCallback } from 'react'

interface GlossaryQueryResult {
  glossary: {
    terms: GraphQLGlossaryTerm[]
    totalDefined: number
    totalSelfDescribing: number
    clusters: Record<string, string[]> | null
  }
}

interface GlossaryChangeSubscriptionResult {
  glossaryChanged: {
    sessionId: string
    action: string
    termName: string
    stage: string | null
    percent: number | null
    termsCount: number | null
    durationMs: number | null
    error: string | null
    term: GraphQLGlossaryTerm | null
  }
}

interface GraphQLGlossaryTerm {
  name: string
  displayName: string
  definition: string | null
  domain: string | null
  domainPath: string | null
  parentId: string | null
  parentVerb: 'HAS_ONE' | 'HAS_KIND' | 'HAS_MANY' | null
  aliases: string[]
  semanticType: string | null
  nerType: string | null
  cardinality: string
  status: string | null
  provenance: string | null
  glossaryStatus: string
  entityId: string | null
  glossaryId: string | null
  tags: Record<string, unknown> | null
  ignored: boolean | null
  canonicalSource: string | null
  spanningDomains: string[] | null
}

function mapTerm(t: GraphQLGlossaryTerm): GlossaryTerm {
  return {
    name: t.name,
    display_name: t.displayName,
    definition: t.definition,
    domain: t.domain,
    domain_path: t.domainPath,
    parent_id: t.parentId,
    parent_verb: t.parentVerb as GlossaryTerm['parent_verb'],
    aliases: t.aliases ?? [],
    semantic_type: t.semanticType,
    ner_type: t.nerType,
    cardinality: t.cardinality,
    status: t.status as GlossaryTerm['status'],
    provenance: t.provenance as GlossaryTerm['provenance'],
    glossary_status: t.glossaryStatus as GlossaryTerm['glossary_status'],
    entity_id: t.entityId,
    glossary_id: t.glossaryId,
    tags: t.tags ?? undefined,
    ignored: t.ignored ?? false,
    canonical_source: t.canonicalSource,
    spanning_domains: t.spanningDomains ?? undefined,
    connected_resources: [],
  }
}

export function useGlossaryData(sessionId: string, scope?: string, domain?: string) {
  const { data, loading: networkLoading, refetch } = useQuery<GlossaryQueryResult>(GLOSSARY_QUERY, {
    variables: { sessionId, scope, domain },
    fetchPolicy: 'cache-and-network',
    skip: !sessionId,
  })

  // Debounce subscription refetch to avoid DOM thrashing during bulk operations
  const refetchTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const debouncedRefetch = useCallback(() => {
    if (refetchTimer.current) clearTimeout(refetchTimer.current)
    refetchTimer.current = setTimeout(() => { refetch() }, 300)
  }, [refetch])

  useSubscription<GlossaryChangeSubscriptionResult>(GLOSSARY_CHANGED_SUBSCRIPTION, {
    variables: { sessionId },
    skip: !sessionId,
    onData: ({ data: subData }) => {
      const event = subData?.data?.glossaryChanged
      if (!event) return
      const gStore = useGlossaryStore.getState()
      switch (event.action) {
        case 'GENERATION_STARTED':
          gStore.setGenerating(true)
          break
        case 'GENERATION_PROGRESS':
          gStore.setProgress(event.stage ?? '', event.percent ?? 0)
          break
        case 'GENERATION_COMPLETE':
          gStore.setGenerating(false)
          break
        default:
          // CREATED / UPDATED / DELETED — refetch glossary data
          debouncedRefetch()
          break
      }
    },
  })

  const terms = useMemo(
    () => (data?.glossary?.terms ?? []).map(mapTerm),
    [data],
  )

  return {
    terms,
    totalDefined: data?.glossary?.totalDefined ?? 0,
    totalSelfDescribing: data?.glossary?.totalSelfDescribing ?? 0,
    clusters: data?.glossary?.clusters ?? {},
    loading: networkLoading && !data?.glossary,
    refetch,
  }
}
