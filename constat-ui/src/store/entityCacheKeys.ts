import type { GlossaryTerm, GlossaryStatus } from '@/types/api'

// Top-level keys
export const K_ENTITIES = 'e' as const
export const K_GLOSSARY = 'g' as const
export const K_RELATIONSHIPS = 'r' as const
export const K_CLUSTERS = 'k' as const

// Entity short keys
export const EK_NAME = 'a' as const
export const EK_DISPLAY_NAME = 'b' as const
export const EK_SEMANTIC_TYPE = 'c' as const
export const EK_NER_TYPE = 'd' as const
export const EK_DOMAIN_ID = 'e' as const

// Glossary short keys
export const GK_NAME = 'a' as const
export const GK_DISPLAY_NAME = 'b' as const
export const GK_DEFINITION = 'c' as const
export const GK_STATUS = 'd' as const
export const GK_PARENT_ID = 'e' as const
export const GK_ALIASES = 'f' as const
export const GK_DOMAIN = 'g' as const
export const GK_DOMAIN_PATH = 'h' as const
export const GK_PARENT_VERB = 'i' as const
export const GK_GLOSSARY_STATUS = 'j' as const
export const GK_ENTITY_ID = 'k' as const
export const GK_SEMANTIC_TYPE = 'l' as const
export const GK_NER_TYPE = 'm' as const
export const GK_TAGS = 'n' as const
export const GK_IGNORED = 'o' as const
export const GK_CANONICAL_SOURCE = 'p' as const

// Relationship short keys
export const RK_SUBJECT = 'a' as const
export const RK_VERB = 'b' as const
export const RK_OBJECT = 'c' as const
export const RK_CONFIDENCE = 'd' as const
export const RK_USER_EDITED = 'e' as const

export interface CompactEntity {
  [EK_NAME]: string
  [EK_DISPLAY_NAME]: string
  [EK_SEMANTIC_TYPE]: string
  [EK_NER_TYPE]: string
  [EK_DOMAIN_ID]: string
}

export interface CompactGlossaryTerm {
  [GK_NAME]: string
  [GK_DISPLAY_NAME]: string
  [GK_DEFINITION]?: string | null
  [GK_STATUS]?: string | null
  [GK_PARENT_ID]?: string | null
  [GK_ALIASES]?: string[]
  [GK_DOMAIN]?: string | null
  [GK_DOMAIN_PATH]?: string | null
  [GK_PARENT_VERB]?: string | null
  [GK_GLOSSARY_STATUS]?: string | null
  [GK_ENTITY_ID]?: string | null
  [GK_SEMANTIC_TYPE]?: string | null
  [GK_NER_TYPE]?: string | null
  [GK_TAGS]?: Record<string, unknown>
  [GK_IGNORED]?: boolean
  [GK_CANONICAL_SOURCE]?: string | null
}

export interface CompactRelationship {
  [RK_SUBJECT]: string
  [RK_VERB]: string
  [RK_OBJECT]: string
  [RK_CONFIDENCE]: number
  [RK_USER_EDITED]?: boolean
}

export interface CompactState {
  [K_ENTITIES]: Record<string, CompactEntity>
  [K_GLOSSARY]: Record<string, CompactGlossaryTerm>
  [K_RELATIONSHIPS]: Record<string, CompactRelationship>
  [K_CLUSTERS]: Record<string, string[]>
}

export interface InflatedRelationship {
  id: string
  subject: string
  verb: string
  object: string
  confidence: number
  user_edited?: boolean
}

export function inflateRelationships(compact: CompactState): InflatedRelationship[] {
  const rels = compact[K_RELATIONSHIPS] ?? {}
  return Object.entries(rels).map(([id, r]) => ({
    id,
    subject: r[RK_SUBJECT],
    verb: r[RK_VERB],
    object: r[RK_OBJECT],
    confidence: r[RK_CONFIDENCE],
    user_edited: r[RK_USER_EDITED],
  }))
}

export function inflateToGlossaryTerms(compact: CompactState): { terms: GlossaryTerm[], totalDefined: number, totalSelfDescribing: number } {
  const glossary = compact[K_GLOSSARY]
  const clusters = compact[K_CLUSTERS] ?? {}
  const allRels = inflateRelationships(compact)

  // Build relationship index by lowercase name
  const relsByName = new Map<string, InflatedRelationship[]>()
  for (const r of allRels) {
    const sKey = r.subject.toLowerCase()
    const oKey = r.object.toLowerCase()
    if (!relsByName.has(sKey)) relsByName.set(sKey, [])
    relsByName.get(sKey)!.push(r)
    if (sKey !== oKey) {
      if (!relsByName.has(oKey)) relsByName.set(oKey, [])
      relsByName.get(oKey)!.push(r)
    }
  }

  // First pass: build terms and index by entity_id/glossary_id for parent resolution
  let totalDefined = 0
  let totalSelfDescribing = 0
  const termEntries: Array<{ key: string; g: CompactGlossaryTerm }> = []
  const termByEntityId = new Map<string, { name: string; display_name: string }>()

  for (const [key, g] of Object.entries(glossary)) {
    termEntries.push({ key, g })
    const entityId = g[GK_ENTITY_ID]
    if (entityId) {
      termByEntityId.set(entityId, { name: g[GK_NAME], display_name: g[GK_DISPLAY_NAME] })
    }
  }

  // Build children index: parent_id -> [{name, display_name, parent_verb}]
  const childrenByParentId = new Map<string, Array<{ name: string; display_name: string; parent_verb?: string }>>()
  for (const { g } of termEntries) {
    const pid = g[GK_PARENT_ID]
    if (pid) {
      if (!childrenByParentId.has(pid)) childrenByParentId.set(pid, [])
      childrenByParentId.get(pid)!.push({
        name: g[GK_NAME],
        display_name: g[GK_DISPLAY_NAME],
        parent_verb: g[GK_PARENT_VERB] ?? undefined,
      })
    }
  }

  const terms = termEntries.map(({ g }) => {
    const glossary_status = (g[GK_GLOSSARY_STATUS] || (g[GK_DEFINITION] ? 'defined' : 'self_describing')) as GlossaryStatus
    if (glossary_status === 'defined') totalDefined++
    else totalSelfDescribing++
    const name = g[GK_NAME]
    const entityId = g[GK_ENTITY_ID]

    // Resolve parent
    const parentId = g[GK_PARENT_ID]
    const parentInfo = parentId ? termByEntityId.get(parentId) ?? null : null

    // Resolve children (parent_id could match entity_id)
    const children = (entityId ? childrenByParentId.get(entityId) : undefined) ?? []

    // Relationships involving this term
    const relationships = relsByName.get(name.toLowerCase()) ?? []

    return {
      name,
      display_name: g[GK_DISPLAY_NAME],
      definition: g[GK_DEFINITION] ?? null,
      domain: g[GK_DOMAIN] ?? null,
      domain_path: g[GK_DOMAIN_PATH] ?? null,
      parent_id: parentId ?? null,
      parent_verb: g[GK_PARENT_VERB] as GlossaryTerm['parent_verb'] ?? undefined,
      parent: parentInfo,
      aliases: g[GK_ALIASES] ?? [],
      semantic_type: g[GK_SEMANTIC_TYPE] ?? null,
      cardinality: 'unknown',
      status: g[GK_STATUS] as GlossaryTerm['status'],
      glossary_status,
      entity_id: entityId ?? null,
      glossary_id: entityId ?? null,
      ner_type: g[GK_NER_TYPE] ?? null,
      tags: g[GK_TAGS] ?? {},
      ignored: g[GK_IGNORED] ?? false,
      connected_resources: [],
      cluster_siblings: clusters[name] ?? undefined,
      children,
      relationships,
      canonical_source: g[GK_CANONICAL_SOURCE] ?? null,
    } as GlossaryTerm
  })
  return { terms, totalDefined, totalSelfDescribing }
}
