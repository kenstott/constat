// Copyright (c) 2025 Kenneth Stott
// Canary: a0654525-3db9-49e2-89af-5f814308b399
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'
import type { Session } from '@/types/api'

// Map GQL camelCase response to frontend snake_case Session type
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toSession(gql: any): Session {
  return {
    session_id: gql.sessionId,
    user_id: gql.userId,
    status: gql.status?.toLowerCase() ?? 'idle',
    created_at: gql.createdAt,
    last_activity: gql.lastActivity,
    current_query: gql.currentQuery ?? undefined,
    summary: gql.summary ?? undefined,
    active_domains: gql.activeDomains ?? [],
    tables_count: gql.tablesCount ?? 0,
    artifacts_count: gql.artifactsCount ?? 0,
    shared_with: gql.sharedWith ?? [],
    is_public: gql.isPublic ?? false,
  }
}

// -- Fragments ---------------------------------------------------------------

const SESSION_FIELDS = gql`
  fragment SessionFields on SessionType {
    sessionId
    userId
    status
    createdAt
    lastActivity
    currentQuery
    summary
    activeDomains
    tablesCount
    artifactsCount
    sharedWith
    isPublic
  }
`

// -- Queries -----------------------------------------------------------------

export const SESSIONS_QUERY = gql`
  ${SESSION_FIELDS}
  query Sessions {
    sessions {
      sessions {
        ...SessionFields
      }
      total
    }
  }
`

export const SESSION_QUERY = gql`
  ${SESSION_FIELDS}
  query Session($sessionId: String!) {
    session(sessionId: $sessionId) {
      ...SessionFields
    }
  }
`

export const SESSION_SHARES_QUERY = gql`
  query SessionShares($sessionId: String!) {
    sessionShares(sessionId: $sessionId)
  }
`

export const ACTIVE_DOMAINS_QUERY = gql`
  query ActiveDomains($sessionId: String!) {
    activeDomains(sessionId: $sessionId)
  }
`

// -- Mutations ---------------------------------------------------------------

export const CREATE_SESSION = gql`
  ${SESSION_FIELDS}
  mutation CreateSession($sessionId: String!, $userId: String) {
    createSession(sessionId: $sessionId, userId: $userId) {
      ...SessionFields
    }
  }
`

export const DELETE_SESSION = gql`
  mutation DeleteSession($sessionId: String!) {
    deleteSession(sessionId: $sessionId)
  }
`

export const TOGGLE_PUBLIC_SHARING = gql`
  mutation TogglePublicSharing($sessionId: String!, $public: Boolean!) {
    togglePublicSharing(sessionId: $sessionId, public: $public) {
      status
      public
      shareUrl
    }
  }
`

export const SHARE_SESSION = gql`
  mutation ShareSession($sessionId: String!, $email: String!) {
    shareSession(sessionId: $sessionId, email: $email) {
      status
      shareUrl
    }
  }
`

export const REMOVE_SHARE = gql`
  mutation RemoveShare($sessionId: String!, $sharedUserId: String!) {
    removeShare(sessionId: $sessionId, sharedUserId: $sharedUserId)
  }
`

export const RESET_CONTEXT = gql`
  mutation ResetContext($sessionId: String!) {
    resetContext(sessionId: $sessionId)
  }
`

export const SET_ACTIVE_DOMAINS = gql`
  mutation SetActiveDomains($sessionId: String!, $domains: [String!]!) {
    setActiveDomains(sessionId: $sessionId, domains: $domains)
  }
`
