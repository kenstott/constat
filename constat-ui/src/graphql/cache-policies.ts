// Copyright (c) 2025 Kenneth Stott
// Canary: f7952342-194d-404c-a56a-4fe229eabdcb
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import type { TypePolicies } from '@apollo/client'

export const typePolicies: TypePolicies = {
  // Glossary (existing)
  GlossaryTermType: { keyFields: ['name', 'domain'] },
  EntityRelationshipType: { keyFields: ['id'] },

  // Sessions
  SessionType: { keyFields: ['sessionId'] },

  // Tables — keyed by name (unique within session)
  TableInfoType: { keyFields: ['name'] },

  // Artifacts — keyed by id
  ArtifactInfoType: { keyFields: ['id'] },

  // Facts — keyed by name
  FactInfoType: { keyFields: ['name'] },

  // Entities — keyed by name + entityType
  EntityInfoType: { keyFields: ['name', 'entityType'] },

  // Domains — keyed by filename
  DomainInfoType: { keyFields: ['filename'] },

  // Databases — keyed by name
  SessionDatabaseType: { keyFields: ['name'] },

  // APIs — keyed by name
  SessionApiType: { keyFields: ['name'] },

  // Documents — keyed by name
  SessionDocumentType: { keyFields: ['name'] },

  // Rules — keyed by id
  RuleInfoType: { keyFields: ['id'] },

  // Learnings — keyed by id
  LearningInfoType: { keyFields: ['id'] },

  // Query: replace arrays on refetch (don't merge)
  Query: {
    fields: {
      tables: { merge: false },
      artifacts: { merge: false },
      facts: { merge: false },
      entities: { merge: false },
      sessions: { merge: false },
      domains: { merge: false },
      learnings: { merge: false },
    },
  },
}
